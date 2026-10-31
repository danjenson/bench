"""Measure the fresh worker process, then verify outside its timed interval."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def main(path):
    path = Path(path).resolve()
    assert sha(path) == os.environ['BENCH_EXTENSION_REQUEST_SHA256']
    request = json.loads(path.read_text())
    assert Path(sys.executable).resolve() == Path(request['python_binary'])
    assert all(sha(p) == expected for p, expected in request['pins'].items())
    output = Path(request['output'])
    results_name = request.get('partial_result_file', 'results.json')
    assert isinstance(results_name, str) and results_name == Path(results_name).name
    results_path = output / results_name
    state_path = path.with_name(path.stem + '-state.json')
    assert not output.exists() and not state_path.exists(), 'Preserve an existing trial'
    state = {'status': 'running', 'action': 'run', 'pid': os.getpid(), 'request': str(path),
             'request_sha256': sha(path), 'started_at_utc': datetime.now(timezone.utc).isoformat()}
    write(state_path, state)
    environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(Path(request['package_root']).parent)}
    try:
        first_result = None
        started = time.monotonic()
        process = subprocess.Popen([sys.executable, '-u', request['worker'], 'run', str(path)], env=environment)
        while process.poll() is None:
            if first_result is None and results_path.exists():
                saved = json.loads(results_path.read_text())
                if saved.get('metrics'):
                    first_result = time.monotonic() - started
            time.sleep(request['first_result_poll_seconds'])
        wall = time.monotonic() - started
        assert process.returncode == 0, f'Worker exited {process.returncode}; inspect original partial output'
        assert results_path.exists()
        if first_result is None:
            assert json.loads(results_path.read_text()).get('metrics')
            first_result = wall
        timing = {'request_sha256': sha(path), 'wall_seconds': wall, 'time_to_first_result_seconds': first_result,
            'first_result_observation_resolution_seconds': request['first_result_poll_seconds'],
            'latency_limit_seconds': request['latency_limit_seconds'], 'passed': wall <= request['latency_limit_seconds'],
            'scope': 'Parent wall clock from fresh child launch through child exit: imports, input checks, checkpoint loading/hashing, CUDA setup, bench.run extraction/fitting/scoring/cleanup, caller checks and process teardown. GPU availability wait and independent CPU verification are excluded. OS file caches are uncontrolled; no model-specific features or probes are reused.'}
        write(output / 'timing.json', timing)
        state['action'] = 'verify'
        write(state_path, state)
        verification_started = time.monotonic()
        subprocess.run([sys.executable, '-u', request['worker'], 'verify', str(path)],
                       env={**environment, 'CUDA_VISIBLE_DEVICES': ''}, check=True)
        state['independent_verification_seconds'] = time.monotonic() - verification_started
        proof = json.loads((output / 'verification.json').read_text())
        assert proof['status'] == 'verified' and proof['request_sha256'] == sha(path)
        assert proof['results_sha256'] == sha(results_path) and proof['timing_sha256'] == sha(output / 'timing.json')
        state.update(status='complete', verified_records=proof['verified_records'],
            results_sha256=sha(results_path), results_file=results_name, timing_sha256=sha(output / 'timing.json'),
            verification_sha256=sha(output / 'verification.json'), wall_seconds=wall, latency_passed=timing['passed'])
        print('Fresh monitoring wall seconds:', wall, 'limit passed:', timing['passed'], flush=True)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write(state_path, state)


if __name__ == '__main__':
    main(sys.argv[1])
