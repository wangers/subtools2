# Ldx 2025.9
"""3d_speaker开源cam++导出onnx必须在torch<2进行
否则torch端与onnx端输出不一致 这可能是模型内所定义算子导致
由于egrecho依赖的lightning依赖torch版本大于2 故cam++无法用命令egrecho pretrained2onnx进行转换
在这里采用进行单独导出
"""

import logging
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from jsonargparse import CLI


def export(
    mdir: str,
    mid: str = "campplus",
    ckpt_name: str = "model_weight.ckpt",
    verify_trace: bool = True,
):
    pprint(locals())
    target_onnx_file = Path(mdir) / "model.onnx"
    state_dict = torch.load(Path(mdir) / ckpt_name, map_location="cpu")
    state_dict = state_dict.get("state_dict", state_dict)  # handle pl ckpt

    keys = list(state_dict.keys())
    src_m = None
    for k in keys:
        if k.startswith("wespk_backbone."):
            src_m = "wespk"
            break
        elif k.startswith("spklab_backbone."):
            src_m = "spklab"
            break

    if src_m == "wespk":
        try:
            import wespeaker
            from wespeaker.cli.hub import Hub
        except ImportError as e:
            error_msg = f"Failed to import wespeaker: {str(e)}. Please ensure it's installed correctly. Refer to: https://github.com/wenet-e2e/wespeaker/blob/master/README.md."
            raise ImportError(error_msg) from e
        model_dir = Hub.get_model(mid)
        model = wespeaker.load_model_pt(model_dir)
    # elif src_m == "spklab":
    #     from spklab_sv_loader import load_model

    #     model = load_model(mid, random_init=True)
    else:
        raise ValueError("failed infer src model (wespeaker) from stat_dict")
    state_dict = {
        k.replace("wespk_backbone.", "").replace("spklab_backbone.", ""): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    input_sample = torch.ones(1, 345, 80)
    torch.onnx.export(
        model,
        input_sample,
        target_onnx_file,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["feats"],
        output_names=["embs"],
        dynamic_axes={"feats": {0: "B", 1: "T"}, "embs": {0: "B"}},
    )
    if verify_trace:
        verify_runtime(model, target_onnx_file, input_sample, atol=1e-5)


def verify_runtime(
    model: torch.nn.Module,
    onnx_model: str,
    input_sample: torch.Tensor,
    atol: float = 1e-5,
):
    """Validates the exported onnx model."""
    import onnxruntime

    logging.info(f"Validating ONNX model {Path(onnx_model).as_posix()}...")

    # onnx runtime session
    onnx_session_opt = onnxruntime.SessionOptions()
    onnx_session_opt.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    )
    device = next(model.parameters()).device
    provider = (
        "CUDAExecutionProvider" if device.type == "cuda" else "CPUExecutionProvider"
    )

    sess = onnxruntime.InferenceSession(
        Path(onnx_model).as_posix(),
        sess_options=onnx_session_opt,
        providers=[provider],
    )

    ort_output_names = [output.name for output in sess.get_outputs()]

    input_sample.to(device)
    ort_input = {"feats": input_sample.cpu().numpy()}
    with torch.no_grad():
        output_example = model.forward(input_sample)
    ort_out = sess.run(ort_output_names, ort_input)[0]

    # print(ort_out, output_example)
    # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    # print(cos(torch.from_numpy(ort_out), output_example))

    expected = output_example.cpu().numpy()
    if not ort_out.shape == expected.shape:
        logging.error(f"[x] embed shape {ort_out.shape} doesn't match {expected.shape}")
        raise ShapeError(
            "Output shapes do not match between reference model and ONNX exported model"
        )
    else:
        logging.info(f"[✓] {ort_out.shape} matches {expected.shape}")
        # Values

    if not np.allclose(expected, ort_out, atol=atol):
        max_diff = np.amax(np.abs(expected - ort_out))
        logging.error(
            f"[x] values not close enough, max diff: {max_diff} (atol: {atol})"
        )
        atol_msg = "The maximum absolute difference between the output of the reference model and the ONNX exported model is not within the set tolerance"
        raise AtolError(atol_msg)
    else:
        logging.info(f"[✓] all values close (atol: {atol})")


class ShapeError(ValueError):
    pass


class AtolError(ValueError):
    pass


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    CLI(export)
