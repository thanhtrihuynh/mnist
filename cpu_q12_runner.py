from pathlib import Path
import time

import numpy as np

FRAC_BITS = 12

IN_H = 28
IN_W = 28
IN_C = 1

CONV1_OUT_C = 16
CONV1_K = 3

CONV2_OUT_C = 32
CONV2_K = 3

POOL2_OUT_H = 5
POOL2_OUT_W = 5

FLATTEN_SIZE = 800
FC1_OUT = 64
NUM_CLASSES = 10

class CPUQ12Runner:
    def __init__(self, weight_dir, weight_format="hls"):
        self.weight_dir = Path(weight_dir)
        self.weight_format = weight_format
        self.load_weights()

    def _load_txt(self, name, shape):
        path = self.weight_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Thiếu file weight: {path}")

        arr = np.loadtxt(path, dtype=np.int32)
        expected = int(np.prod(shape))

        if arr.size != expected:
            raise ValueError(f"{name}: cần {expected} số, hiện có {arr.size} số.")

        return arr.reshape(shape).astype(np.int16)

    def load_weights(self):
        if self.weight_format == "hls":
            self.conv1_w = self._load_txt("conv1_weight_q12.txt", (CONV1_OUT_C, IN_C, CONV1_K, CONV1_K))
            self.conv2_w = self._load_txt("conv2_weight_q12.txt", (CONV2_OUT_C, CONV1_OUT_C, CONV2_K, CONV2_K))
            self.fc1_w = self._load_txt("fc1_weight_q12.txt", (FC1_OUT, FLATTEN_SIZE))
            self.fc2_w = self._load_txt("fc2_weight_q12.txt", (NUM_CLASSES, FC1_OUT))

        elif self.weight_format == "keras":
            conv1_w_keras = self._load_txt("conv1_weight_q12.txt", (CONV1_K, CONV1_K, IN_C, CONV1_OUT_C))
            conv2_w_keras = self._load_txt("conv2_weight_q12.txt", (CONV2_K, CONV2_K, CONV1_OUT_C, CONV2_OUT_C))
            fc1_w_keras = self._load_txt("fc1_weight_q12.txt", (FLATTEN_SIZE, FC1_OUT))
            fc2_w_keras = self._load_txt("fc2_weight_q12.txt", (FC1_OUT, NUM_CLASSES))

            self.conv1_w = np.transpose(conv1_w_keras, (3, 2, 0, 1)).astype(np.int16)
            self.conv2_w = np.transpose(conv2_w_keras, (3, 2, 0, 1)).astype(np.int16)
            self.fc1_w = np.transpose(fc1_w_keras, (1, 0)).astype(np.int16)
            self.fc2_w = np.transpose(fc2_w_keras, (1, 0)).astype(np.int16)

        else:
            raise ValueError("Q12_WEIGHT_FORMAT chỉ được là 'hls' hoặc 'keras'.")

        self.conv1_b = self._load_txt("conv1_bias_q12.txt", (CONV1_OUT_C,))
        self.conv2_b = self._load_txt("conv2_bias_q12.txt", (CONV2_OUT_C,))
        self.fc1_b = self._load_txt("fc1_bias_q12.txt", (FC1_OUT,))
        self.fc2_b = self._load_txt("fc2_bias_q12.txt", (NUM_CLASSES,))

    @staticmethod
    def sat_int16(x):
        return np.clip(x, -32768, 32767).astype(np.int16)

    @staticmethod
    def relu_int16(x):
        return np.maximum(x, 0).astype(np.int16)

    def conv2d_q12(self, input_arr, weight, bias):
        out_c, in_c, k, _ = weight.shape
        _, h, w = input_arr.shape

        out_h = h - k + 1
        out_w = w - k + 1
        output = np.zeros((out_c, out_h, out_w), dtype=np.int16)

        for oc in range(out_c):
            for oh in range(out_h):
                for ow in range(out_w):
                    acc = int(bias[oc]) << FRAC_BITS

                    for ic in range(in_c):
                        for kh in range(k):
                            for kw in range(k):
                                acc += int(input_arr[ic, oh + kh, ow + kw]) * int(weight[oc, ic, kh, kw])

                    output[oc, oh, ow] = self.sat_int16(acc >> FRAC_BITS)

        return output

    @staticmethod
    def maxpool2x2_q12(input_arr):
        c, h, w = input_arr.shape
        output = np.zeros((c, h // 2, w // 2), dtype=np.int16)

        for ch in range(c):
            for oh in range(h // 2):
                for ow in range(w // 2):
                    window = input_arr[ch, oh * 2:oh * 2 + 2, ow * 2:ow * 2 + 2]
                    output[ch, oh, ow] = np.max(window)

        return output

    def dense_q12(self, input_vec, weight, bias, relu=False, output_int32=False):
        out_dim, in_dim = weight.shape
        output = np.zeros((out_dim,), dtype=np.int32 if output_int32 else np.int16)

        for o in range(out_dim):
            acc = int(bias[o]) << FRAC_BITS

            for i in range(in_dim):
                acc += int(input_vec[i]) * int(weight[o, i])

            val = acc >> FRAC_BITS

            if relu:
                val = max(0, val)

            if output_int32:
                output[o] = np.int32(val)
            else:
                output[o] = self.sat_int16(val)

        return output

    @staticmethod
    def topk_info(logits):
        logits_np = np.asarray(logits)
        order = np.argsort(logits_np)[::-1]
        top1 = int(order[0])
        top2 = int(order[1])
        margin = int(logits_np[top1] - logits_np[top2])
        return top1, top2, margin

    def predict(self, input_q12):
        start = time.perf_counter()

        x = np.asarray(input_q12, dtype=np.int16).reshape(1, 28, 28)

        c1 = self.conv2d_q12(x, self.conv1_w, self.conv1_b)
        r1 = self.relu_int16(c1)
        p1 = self.maxpool2x2_q12(r1)

        c2 = self.conv2d_q12(p1, self.conv2_w, self.conv2_b)
        r2 = self.relu_int16(c2)
        p2 = self.maxpool2x2_q12(r2)

        # Khớp Keras Flatten channels_last: H -> W -> C.
        flat = np.transpose(p2, (1, 2, 0)).reshape(-1)

        fc1 = self.dense_q12(flat, self.fc1_w, self.fc1_b, relu=True, output_int32=False)
        logits = self.dense_q12(fc1, self.fc2_w, self.fc2_b, relu=False, output_int32=True)

        latency_ms = (time.perf_counter() - start) * 1000.0

        top1, top2, margin = self.topk_info(logits)

        return {
            "backend": "cpu_q12",
            "predicted_label": top1,
            "top1_class": top1,
            "top1_logit": int(logits[top1]),
            "top2_class": top2,
            "top2_logit": int(logits[top2]),
            "margin": margin,
            "logits": [int(x) for x in logits.tolist()],
            "latency_ms": latency_ms,
            "throughput_fps": 1000.0 / latency_ms if latency_ms > 0 else 0.0,
            "note": "NumPy fixed-point Q4.12 golden model"
        }
