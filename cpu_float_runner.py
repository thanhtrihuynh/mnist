from pathlib import Path
import time

import numpy as np

FRAC_BITS = 12
SCALE = 1 << FRAC_BITS

class CPUFloatRunner:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.load_model()

    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Thiếu model .keras: {self.model_path}")

        try:
            import tensorflow as tf
        except Exception as exc:
            raise ImportError(
                "Backend cpu_float cần TensorFlow. Cài bằng: python -m pip install -r requirements_cpu_float.txt"
            ) from exc

        self.tf = tf
        self.model = tf.keras.models.load_model(self.model_path)

    @staticmethod
    def topk_info(logits):
        logits_np = np.asarray(logits)
        order = np.argsort(logits_np)[::-1]
        top1 = int(order[0])
        top2 = int(order[1])
        margin = float(logits_np[top1] - logits_np[top2])
        return top1, top2, margin

    def predict(self, input_q12):
        start = time.perf_counter()

        arr = np.asarray(input_q12, dtype=np.float32).reshape(1, 28, 28, 1)
        arr = np.clip(arr / SCALE, 0.0, 1.0)

        logits = self.model.predict(arr, verbose=0)[0]
        latency_ms = (time.perf_counter() - start) * 1000.0

        top1, top2, margin = self.topk_info(logits)

        return {
            "backend": "cpu_float",
            "predicted_label": top1,
            "top1_class": top1,
            "top1_logit": float(logits[top1]),
            "top2_class": top2,
            "top2_logit": float(logits[top2]),
            "margin": margin,
            "logits": [float(x) for x in logits.tolist()],
            "latency_ms": latency_ms,
            "throughput_fps": 1000.0 / latency_ms if latency_ms > 0 else 0.0,
            "note": "TensorFlow/Keras float model"
        }
