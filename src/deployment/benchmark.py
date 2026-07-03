from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import numpy as np


def benchmark_tflite_host(tflite_path: str | Path, image_size: int = 320, runs: int = 50) -> Dict[str, float]:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for host-side TFLite benchmarking.") from exc
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    dtype = input_details["dtype"]
    if np.issubdtype(dtype, np.integer):
        sample_input = np.random.randint(-128, 127, input_details["shape"], dtype=dtype)
    else:
        sample_input = np.random.random(input_details["shape"]).astype(dtype)
    for _ in range(5):
        interpreter.set_tensor(input_details["index"], sample_input)
        interpreter.invoke()
    start = time.perf_counter()
    for _ in range(runs):
        interpreter.set_tensor(input_details["index"], sample_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details["index"])
    elapsed = time.perf_counter() - start
    latency_ms = elapsed * 1000.0 / runs
    return {"latency_ms_host": latency_ms, "fps_host": 1000.0 / latency_ms}
