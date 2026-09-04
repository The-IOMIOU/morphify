"""Make the LivePortrait warping model loadable by stock onnxruntime.

The published `warping_spade-fix.onnx` uses a node called ``GridSample3D``
in the default ONNX domain. No such operator exists in the standard — it is
a placeholder the exporter emitted because opset 16's ``GridSample`` only
handled 4D tensors, and this model samples a 5D feature volume. Upstream
works around it with a TensorRT plugin and a custom onnxruntime build.

Neither is needed any more: **opset 20 added 5D support to the standard
GridSample**, and onnxruntime 1.20+ implements it. So the fix is to rewrite
the graph rather than rebuild the runtime — rename the node to
``GridSample``, translate the attribute spelling (opset 20 renamed the
interpolation modes), and bump the opset.

That keeps the app on the same stock onnxruntime the face swapper uses,
instead of pinning a custom 1.17 build that would drag the rest of the
pipeline backwards.

    python setup/patch_liveportrait_onnx.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnx  # noqa: E402

from modules.paths import MODELS_DIR  # noqa: E402

SOURCE_NAME = "warping_spade-fix.onnx"
PATCHED_NAME = "warping_spade-ort.onnx"

TARGET_OPSET = 20

# Opset 20 renamed GridSample's interpolation modes.
MODE_MAP = {
    b"bilinear": "linear",
    b"bicubic": "cubic",
    b"nearest": "nearest",
    "bilinear": "linear",
    "bicubic": "cubic",
    "nearest": "nearest",
}


def patch_gridsample(model) -> int:
    """GridSample3D -> standard GridSample, with opset-20 attribute names."""
    changed = 0
    for node in model.graph.node:
        if node.op_type != "GridSample3D":
            continue
        node.op_type = "GridSample"
        node.domain = ""
        for attribute in node.attribute:
            if attribute.name == "mode":
                current = onnx.helper.get_attribute_value(attribute)
                attribute.s = MODE_MAP.get(current, "linear").encode()
        changed += 1
    return changed


def patch_resize_to_4d(model) -> int:
    """Fold 5D Resize nodes that only scale H and W down to 4D.

    onnxruntime's CUDA provider has no 5D Resize, so every one of these ran
    on the CPU with a full round trip of a ~90 MB feature volume — together
    they were the single largest cost in the model. Each of them scales
    [N, C, D, H, W] by [1, 1, 1, 2, 2], leaving the depth axis untouched,
    which makes them exactly equivalent to merging C and D, resizing in 4D,
    and splitting them again. Reshape is free on the GPU, so this moves the
    whole operation onto CUDA without changing the numerics.
    """
    from onnx import numpy_helper

    initializers = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    shapes = {}
    for collection in (model.graph.value_info, model.graph.input, model.graph.output):
        for value in collection:
            dims = value.type.tensor_type.shape.dim
            shapes[value.name] = [d.dim_value if d.dim_value else None for d in dims]

    new_nodes = []
    changed = 0
    for node in model.graph.node:
        scales_name = node.input[2] if node.op_type == "Resize" and len(node.input) > 2 else ""
        scales = initializers.get(scales_name)
        in_shape = shapes.get(node.input[0])

        eligible = (
            node.op_type == "Resize"
            and scales is not None
            and len(scales) == 5
            and float(scales[0]) == 1.0
            and float(scales[1]) == 1.0
            and float(scales[2]) == 1.0      # depth untouched
            and in_shape is not None
            and len(in_shape) == 5
            # Every axis must be known: the replacement reshapes need
            # concrete targets, and a dynamic dim would make them guesses.
            and all(in_shape[i] for i in (1, 2, 3, 4))
        )
        if not eligible:
            new_nodes.append(node)
            continue

        n, c, d, h, w = in_shape
        n = n or 1
        prefix = f"{node.name or node.output[0]}_4d"

        # Both target shapes must be fully concrete: Reshape permits a single
        # -1, and here two axes change. The spatial dims are static and the
        # scale is a constant, so the output size is known exactly.
        out_h = int(round(h * float(scales[3])))
        out_w = int(round(w * float(scales[4])))

        merged_shape = onnx.helper.make_tensor(
            f"{prefix}_merge_shape", onnx.TensorProto.INT64, [4],
            [n, c * d, h, w])
        split_shape = onnx.helper.make_tensor(
            f"{prefix}_split_shape", onnx.TensorProto.INT64, [5],
            [n, c, d, out_h, out_w])
        scales_4d = onnx.helper.make_tensor(
            f"{prefix}_scales", onnx.TensorProto.FLOAT, [4],
            [1.0, 1.0, float(scales[3]), float(scales[4])])
        model.graph.initializer.extend([merged_shape, split_shape, scales_4d])

        new_nodes.append(onnx.helper.make_node(
            "Reshape", [node.input[0], merged_shape.name], [f"{prefix}_merged"],
            name=f"{prefix}_reshape_in"))

        resize_inputs = [f"{prefix}_merged", "", scales_4d.name]
        resized = onnx.helper.make_node(
            "Resize", resize_inputs, [f"{prefix}_resized"],
            name=f"{prefix}_resize")
        # Carry the original interpolation settings across unchanged.
        for attribute in node.attribute:
            resized.attribute.append(attribute)
        new_nodes.append(resized)

        new_nodes.append(onnx.helper.make_node(
            "Reshape", [f"{prefix}_resized", split_shape.name], [node.output[0]],
            name=f"{prefix}_reshape_out"))
        changed += 1

    if changed:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
    return changed


def patch(source: str, destination: str) -> tuple:
    """Rewrite the model. Returns (gridsample_nodes, resize_nodes) changed."""
    model = onnx.load(source)

    grids = patch_gridsample(model)
    resizes = patch_resize_to_4d(model)

    if grids:
        del model.opset_import[:]
        model.opset_import.extend([onnx.helper.make_opsetid("", TARGET_OPSET)])

    # Shape inference has to be redone after rewriting the graph, and the
    # stale value_info would otherwise contradict the new 4D tensors.
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model)

    onnx.save(model, destination, save_as_external_data=False)
    return grids, resizes


def verify(path: str) -> bool:
    """Load the patched model and confirm onnxruntime accepts it."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    for providers in (["CUDAExecutionProvider", "CPUExecutionProvider"],
                      ["CPUExecutionProvider"]):
        try:
            session = ort.InferenceSession(
                path, sess_options=options, providers=providers)
            active = session.get_providers()
            print(f"  loads on {', '.join(active)}")
            for item in session.get_inputs():
                print(f"    IN   {item.name:<12} {item.shape}")
            for item in session.get_outputs():
                print(f"    OUT  {item.name:<12} {item.shape}")
            return True
        except Exception as exc:
            print(f"  {providers[0]}: {str(exc).splitlines()[0][:160]}")
    return False


def main() -> int:
    folder = os.path.join(MODELS_DIR, "liveportrait")
    source = os.path.join(folder, SOURCE_NAME)
    destination = os.path.join(folder, PATCHED_NAME)

    if not os.path.isfile(source):
        print(f"missing {source}")
        return 1

    if os.path.isfile(destination) and os.path.getsize(destination) > 1_000_000:
        print(f"already patched: {destination}")
        return 0 if verify(destination) else 1

    print(f"patching {SOURCE_NAME} -> {PATCHED_NAME}")
    grids, resizes = patch(source, destination)
    print(f"  GridSample3D -> GridSample : {grids} node(s), opset -> {TARGET_OPSET}")
    print(f"  5D Resize    -> 4D Resize  : {resizes} node(s)")
    if not grids and not resizes:
        print("  nothing to change; the model may already be standard")

    return 0 if verify(destination) else 1


if __name__ == "__main__":
    raise SystemExit(main())
