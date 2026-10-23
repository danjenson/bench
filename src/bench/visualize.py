"""Offline annotated task inspection, always from development data."""

import colorsys
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .images import make_canvas, mask_canvas, original_to_canvas
from .prepare import load_prepared
from .protocols import resolve_names
from .storage import atomic_write, contained_path, lock, read_json, resolve_path, write_json
from .tasks.correspondence import eligible_keypoints, pck_radius
from .tasks.segmentation import read_mask


def color(index):
    return tuple(round(value * 255) for value in colorsys.hsv_to_rgb((index * 0.61803398875) % 1, 0.65, 0.95))


def select_examples(rows):
    ordered = sorted(rows, key=lambda row: row["id"])
    chosen, groups = [], set()
    for row in ordered:
        group = (row.get("target"), row.get("category"), row.get("viewpoint_variation"))
        if group not in groups:
            chosen.append(row)
            groups.add(group)
        if len(chosen) == 6:
            return chosen
    chosen_ids = {row["id"] for row in chosen}
    return (chosen + [row for row in ordered if row["id"] not in chosen_ids])[:6]


def explain(task):
    canvas = task["canvas"]
    if task["source"] == "imagenet":
        return f"The model receives a {canvas} × {canvas} center crop after resizing the shorter side to {task['resize_short']} pixels. A linear classifier learns from frozen global features and training labels. Top-1 accuracy is the percentage of images whose predicted class matches the annotation, over this recipe's {task['classes']}-class set."
    if task["kind"] == "segmentation":
        return f"The entire image is fitted into a {canvas} × {canvas} canvas with black padding. A shared linear pixel classifier learns from frozen spatial features and training masks. Logits are sampled geometrically with bilinear interpolation and border replication. Mean IoU is computed from the dataset confusion matrix; padding and official ignore regions contribute to neither training loss nor scoring."
    if task["kind"] == "correspondence":
        return "The model receives the two complete images separately, each fitted into a 448 × 448 canvas. Source-keypoint features are sampled bilinearly, normalized, and matched by cosine similarity to native target feature centers. PCK@0.1 pools correct eligible correspondences across pairs; a match is correct within 10% of the target box's largest original-image side. Boxes and keypoints are scoring annotations unavailable to the model. The nearest native target location defines a separate grid-limited ceiling."
    if task["name"] == "clevr-count":
        return "The model receives the entire scene fitted into a 224 × 224 canvas. A linear classifier learns from frozen global features. The label counts all annotated scene objects (3–10), including objects difficult to see; the classifier target is count minus three. Accuracy is the percentage of scenes classified correctly. Object-center overlays are explanatory annotations, never model input."
    return "The model receives the entire scene fitted into a 224 × 224 canvas. A linear classifier learns to classify the closest object's annotated camera depth from frozen global features. Six bins have inclusive upper boundaries 8, 8.5, 9, 9.5, 10, and 100. This is not the distance between two objects. Accuracy is the percentage of correct bin predictions; depth and object-center overlays never reach the model."


def segmentation_figure(canvas, mask, labels):
    transformed = mask_canvas(mask, canvas)
    image = np.array(canvas.image).astype(np.float32)
    legend = []
    for label in np.unique(transformed):
        if label == 255:
            continue
        rgb = color(int(label))
        selected = transformed == label
        image[selected] = 0.45 * image[selected] + 0.55 * np.asarray(rgb)
        legend.append((rgb, labels[label]))
    # Padding and annotation ignore remain visibly distinct.
    yy, xx = np.indices(transformed.shape)
    ignore = (transformed == 255) & canvas.valid
    image[ignore & ((xx + yy) % 12 < 4)] = (255, 70, 170)
    image[~canvas.valid] = np.where((((xx + yy) // 8) % 2)[~canvas.valid, None], (45, 65, 80), (80, 100, 115))
    return Image.fromarray(image.astype(np.uint8)), legend


def spair_figure(source_image, target_image, source, target, annotation):
    side = source.image.width
    result = Image.new("RGB", (2 * side, side))
    result.paste(source.image)
    result.paste(target.image, (side, 0))
    draw = ImageDraw.Draw(result)
    src, trg, valid = eligible_keypoints(annotation)
    src = original_to_canvas(src, source_image.size, source)
    trg = original_to_canvas(trg, target_image.size, target) + [side, 0]
    box = np.asarray(annotation["trg_bndbox"], dtype=float)
    mapped = original_to_canvas(box.reshape(2, 2), target_image.size, target) + [side, 0]
    draw.rectangle(tuple(mapped.reshape(-1)), outline="white", width=2)
    for index in np.flatnonzero(valid):
        rgb = color(int(index))
        draw.line([tuple(src[index]), tuple(trg[index])], fill=rgb, width=2)
        for x, y in (src[index], trg[index]):
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=rgb, outline="black")
    if valid.any():
        x, y = trg[np.flatnonzero(valid)[0]]
        radius = pck_radius(annotation, 0.1)
        rx = radius * target.resized_size[0] / target_image.width
        ry = radius * target.resized_size[1] / target_image.height
        draw.ellipse((x - rx, y - ry, x + rx, y + ry), outline="white", width=2)
    return result, int(valid.sum())


def render_example(root, task, labels, row, annotation=None):
    with Image.open(contained_path(root, row["image"])) as stream:
        original = stream.convert("RGB")
    canvas = make_canvas(original, task)
    plain = canvas.image
    legend = []
    original_marked = None
    if task["kind"] == "segmentation":
        annotated, legend = segmentation_figure(canvas, read_mask(contained_path(root, row["mask"]), task), labels)
        caption = "Training target: semantic mask. Pink hatching: annotation ignore. Slate hatching: padding. Neither is scored."
    elif task["kind"] == "correspondence":
        with Image.open(contained_path(root, row["target_image"])) as stream:
            target_original = stream.convert("RGB")
        target = make_canvas(target_original, task)
        pair = annotation if annotation is not None else read_json(contained_path(root, row["annotation"]))
        annotated, count = spair_figure(original, target_original, canvas, target, pair)
        plain = Image.new("RGB", annotated.size)
        plain.paste(canvas.image)
        plain.paste(target.image, (canvas.image.width, 0))
        caption = f"{row['category']}; {count} eligible correspondences. White target box and ellipse define scoring only. Viewpoint {pair.get('viewpoint_variation')}; scale {pair.get('scale_variation')}; occlusion {pair.get('occlusion')}."
    elif task["source"] == "clevr":
        annotated = canvas.image.copy()
        draw = ImageDraw.Draw(annotated)
        scene = annotation if annotation is not None else read_json(contained_path(root, row["annotation"]))["scenes"][row["scene_index"]]
        objects = scene["objects"]
        points = original_to_canvas([item["pixel_coords"][:2] for item in objects], original.size, canvas)
        nearest = min(range(len(objects)), key=lambda i: objects[i]["pixel_coords"][2])
        for index, (x, y) in enumerate(points):
            if task["name"] == "clevr-distance" and index != nearest:
                continue
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline="yellow", width=2)
            draw.text((x + 8, y - 7), str(index + 1), fill="yellow", stroke_width=1, stroke_fill="black")
        if task["name"] == "clevr-count":
            caption = f"{len(objects)} annotated objects; classifier target {row['target']}."
        else:
            value = objects[nearest]["pixel_coords"][2]
            thresholds = task["distance_thresholds"]
            lower = thresholds[row["target"] - 1] if row["target"] else "−∞"
            caption = f"Closest camera depth {value:.4f}; bin {row['target']}: ({lower}, {thresholds[row['target']]}]."
    else:
        annotated = canvas.image.copy()
        caption = f"Ground truth: {labels[row['target']]} (classifier target {row['target']}). Label trains the probe; it is never an encoder input."
        original_marked = original.copy()
        corners = np.array([[0, 0, 1], [1, 1, 1]]) @ canvas.to_original.T
        box = (corners[:, :2] * original.size).reshape(-1)
        ImageDraw.Draw(original_marked).rectangle(tuple(box), outline="yellow", width=max(2, original.width // 200))
        original_marked.thumbnail((512, 512))
    return plain, annotated, original_marked, caption, legend


def visualize(*, names, data_root=Path("~/datasets"), output_dir=Path("outputs/bench/visualizations"), datasets=None):
    tasks = resolve_names(names)
    root, output = resolve_path(data_root), resolve_path(output_dir)
    prepared = [(task, *load_prepared(root, task, "development")) for task in tasks]
    with lock(output.parent / f".{output.name}.lock"):
        output.mkdir(parents=True, exist_ok=True)
        sections, provenance = [], []
        for task, manifest, rows in prepared:
            source = Path(manifest["source_root"])
            if (datasets or {}).get(task["source"]) and resolve_path(datasets[task["source"]]) != source:
                raise ValueError(f"Configured {task['source']} path differs from prepared source; rerun prepare with datasets.{task['source']}")
            examples = select_examples(rows)
            scenes = read_json(contained_path(source, examples[0]["annotation"]))["scenes"] if task["source"] == "clevr" else None
            figures = []
            for row in examples:
                plain, annotated, original, caption, legend = render_example(source, task, manifest["labels"], row, scenes[row["scene_index"]] if scenes is not None else None)
                stem = task["name"] + "-" + row["id"][:16]
                plain.save(output / f"{stem}-input.png")
                annotated.save(output / f"{stem}-annotation.png")
                source_view = ""
                if original is not None:
                    original.save(output / f"{stem}-original.png")
                    source_view = f'<details><summary>Original image and supplied crop</summary><img src="{stem}-original.png" alt="Original image with the input crop outlined in yellow"></details>'
                swatches = " ".join(f'<span class="legend"><i style="background:rgb{rgb}"></i>{escape(label)}</span>' for rgb, label in legend)
                figures.append(f'<article><p class="sample">DEVELOPMENT · {row["id"][:16]}</p><div class="views"><figure><figcaption>MODEL INPUT</figcaption><a href="{stem}-input.png"><img src="{stem}-input.png" alt="Exact unannotated model canvas"></a></figure><figure><figcaption>ANNOTATIONS</figcaption><a href="{stem}-annotation.png"><img src="{stem}-annotation.png" alt="Task annotations overlaid on the same canvas"></a></figure></div><p>{escape(caption)}</p><div>{swatches}</div>{source_view}</article>')
            sections.append(f'<section id="{task["name"]}"><div class="task"><h2>{task["name"]}</h2><code>{task["id"]}</code><p>{escape(explain(task))}</p><small>Protocol candidate · {task["canvas"]} px canvas · {task["method"]} readout</small></div><div>{"".join(figures)}</div></section>')
            provenance.append({"task": task["id"], "task_digest": task["digest"], "preparation_digest": manifest["digest"], "sample_ids": [row["id"] for row in examples], "partition": "development"})
        navigation = "".join(f'<a href="#{task["name"]}">{task["name"]}</a>' for task in tasks)
        html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>bench · Task inspection</title><style>
        :root{--ink:#203b50;--paper:#edf3f6;--blue:#005e85;--line:#becdd7;--white:#fff;--muted:#506573}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 "Trebuchet MS",sans-serif}header,main{max-width:1400px;margin:auto;padding:32px}header{padding-top:56px}h1{font:600 clamp(34px,5vw,60px)/1.1 "Palatino Linotype",Palatino,serif;margin:12px 0 20px}h2{font-size:26px;margin:0}p{max-width:75ch}code,.sample,figcaption,small{font-family:"DejaVu Sans Mono",monospace;font-size:12px}nav{display:flex;flex-wrap:wrap;gap:10px 24px;margin-top:28px}a{color:var(--blue);text-underline-offset:4px}a:focus-visible,summary:focus-visible{outline:3px solid var(--blue);outline-offset:4px}section{display:grid;grid-template-columns:290px 1fr;gap:40px;padding:36px 0;border-top:2px solid var(--line)}.task{align-self:start;position:sticky;top:20px}.task code{overflow-wrap:anywhere}article{background:var(--white);padding:20px;margin-bottom:28px;border:1px solid var(--line)}.views{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0;min-width:0}figcaption{padding:0 0 8px;color:var(--muted)}img{display:block;width:100%;height:auto}.sample{color:var(--muted);margin-top:0}.legend{display:inline-flex;align-items:center;gap:5px;margin:4px 12px 4px 0;font-size:12px}.legend i{width:12px;height:12px;display:inline-block}details{margin-top:14px}details img{max-width:512px;margin-top:12px}summary{cursor:pointer;color:var(--blue)}@media(max-width:900px){section{grid-template-columns:1fr;gap:20px}.task{position:static}}@media(max-width:540px){header,main{padding:20px}.views{grid-template-columns:1fr}article{padding:12px}}
        </style><header><small>BENCH / DATA & PROTOCOL INSPECTION</small><h1>What the model sees.<br>What the task measures.</h1><p>Exact supplied canvases beside real annotations. These examples come only from the development partition. Click any image to inspect the full resolution.</p><nav>''' + navigation + '</nav></header><main>' + "".join(sections) + '</main></html>'
        write_json(output / "manifest.json", {"schema_version": "1.0.0", "visualization_revision": "pillow-gallery-1", "tasks": provenance})
        atomic_write(output / "index.html", html.encode())
    print(f"Report: {output / 'index.html'}")
    return output / "index.html"
