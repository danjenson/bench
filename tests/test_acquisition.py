import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import tarfile
from threading import Thread
import zipfile

import pytest

from bench.acquisition import download, extract_archive
from bench.storage import file_digest, write_json


@pytest.mark.parametrize("archive_type", ["tar", "zip"])
@pytest.mark.parametrize("member", ["../outside", "/absolute"])
def test_archive_traversal_is_rejected(tmp_path, archive_type, member):
    path = tmp_path / f"bad.{archive_type}"
    if archive_type == "zip":
        with zipfile.ZipFile(path, "w") as stream:
            stream.writestr(member, b"bad")
    else:
        with tarfile.open(path, "w") as stream:
            entry = tarfile.TarInfo(member)
            entry.size = 3
            stream.addfile(entry, io.BytesIO(b"bad"))
    with pytest.raises(ValueError, match="escapes"):
        extract_archive(path, tmp_path / "staging")
    assert not (tmp_path / "outside").exists()


def test_archive_links_are_rejected(tmp_path):
    path = tmp_path / "link.tar"
    with tarfile.open(path, "w") as stream:
        entry = tarfile.TarInfo("escape")
        entry.type = tarfile.SYMTYPE
        entry.linkname = "../outside"
        stream.addfile(entry)
    with pytest.raises(ValueError, match="Unsupported"):
        extract_archive(path, tmp_path / "staging")


def test_download_resume_checksum_and_server_ignoring_range(tmp_path):
    payload = b"verified dataset payload" * 100
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(dict(self.headers))
            offset = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
            resume = offset and self.path == "/resume"
            self.send_response(206 if resume else 200)
            self.send_header("ETag", '"edition-1"')
            if resume:
                self.send_header("Content-Range", f"bytes {offset}-{len(payload)-1}/{len(payload)}")
            self.end_headers()
            self.wfile.write(payload[offset:] if resume else payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    checksum = {"algorithm": "sha256", "value": hashlib.sha256(payload).hexdigest()}
    try:
        for route in ("resume", "restart"):
            destination = tmp_path / f"{route}.zip"
            url = f"http://127.0.0.1:{server.server_port}/{route}"
            destination.with_suffix(".zip.part").write_bytes(payload[:100])
            write_json(destination.with_suffix(".zip.part.json"), {"url": url, "etag": '"edition-1"'})
            download(url, destination, checksum)
            assert destination.read_bytes() == payload
            assert requests[-1]["Range"] == "bytes=100-"
        bad = tmp_path / "bad.zip"
        with pytest.raises(ValueError, match="Checksum mismatch"):
            download(url, bad, {"algorithm": "sha256", "value": "0" * 64})
        assert not bad.exists()
        assert not bad.with_suffix(".zip.part").exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_preparation_validates_staged_images_before_publish(clevr_data, tmp_path):
    import importlib
    from PIL import Image

    root, source = clevr_data
    preparation = importlib.import_module("bench.prepare")
    recipe = preparation.source_recipes()["clevr"]
    seed = tmp_path / "seed"
    source.rename(seed)
    image = next((seed / "images" / "train").iterdir())
    image.write_bytes(b"invalid image in a checksum-valid archive")
    archive = root / recipe["archive"]
    recipe["download_bytes"], recipe["peak_bytes"] = 10000, 100000
    recipe["checksum"] = {"algorithm": "sha256", "value": ""}

    def pack():
        with zipfile.ZipFile(archive, "w") as stream:
            for path in seed.rglob("*"):
                if path.is_file():
                    stream.write(path, str(Path(recipe["archive_root"]) / path.relative_to(seed)))
        recipe["checksum"]["value"] = file_digest(archive)

    from pathlib import Path

    pack()
    assert not preparation.prepare(names=["clevr-count"], data_root=root)["ready"]
    assert not source.exists() and archive.exists()
    assert not (root / ".bench" / "prepared" / "clevr-count-v1" / "manifest.json").exists()
    Image.new("RGB", (32, 24)).save(image)
    pack()
    assert preparation.prepare(names=["clevr-count"], data_root=root)["ready"]
    assert source.is_dir() and archive.exists()  # User-supplied archive is retained.
