from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

import torch


def export_onnx(model: torch.nn.Module, output_path: str | Path, image_size: int = 320, opset: int = 13) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    sample_input = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model.cpu(),
        sample_input,
        output,
        input_names=["input"],
        output_names=["predictions"],
        opset_version=opset,
        dynamic_axes={"input": {0: "batch"}, "predictions": {0: "batch"}},
    )


def convert_onnx_to_tflite_int8(onnx_path: str | Path, output_dir: str | Path, representative_dir: str | Path | None = None) -> Path:
    """Use onnx2tf CLI when available to create an INT8 TFLite model.

    Full integer quantization depends on TensorFlow and onnx2tf. This function raises a clear
    error if the external converter is not installed.
    """
    onnx_path = Path(onnx_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    cmd = [
        "onnx2tf",
        "-i", str(onnx_path),
        "-o", str(output_dir),
        "-oiqt",
        "-qt", "per-channel",
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("onnx2tf is not installed. Install it or export ONNX only.") from exc
    candidates = sorted(output_dir.rglob("*.tflite"))
    if not candidates:
        raise RuntimeError(f"Conversion finished but no TFLite file was found in {output_dir}")
    return candidates[0]


def export_tflite_c_array(tflite_path: str | Path, cc_path: str | Path, h_path: str | Path, array_name: str) -> None:
    data = Path(tflite_path).read_bytes()
    cc = Path(cc_path)
    hh = Path(h_path)
    cc.parent.mkdir(parents=True, exist_ok=True)
    hh.parent.mkdir(parents=True, exist_ok=True)
    guard = f"{array_name.upper()}_H_"
    hh.write_text(
        f"#ifndef {guard}\n#define {guard}\n\n#include <cstdint>\n\nextern const unsigned char {array_name}[];\nextern const unsigned int {array_name}_len;\n\n#endif\n",
        encoding="utf-8",
    )
    lines = [f'#include "{hh.name}"', "", f"const unsigned char {array_name}[] = {{"]
    for i in range(0, len(data), 12):
        chunk = data[i:i + 12]
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    lines.append("};")
    lines.append(f"const unsigned int {array_name}_len = {len(data)};")
    cc.write_text("\n".join(lines) + "\n", encoding="utf-8")
