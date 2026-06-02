class PynqRunner:
    def __init__(self, bit_path, hwh_path, backend_name="fpga_hls"):
        self.bit_path = str(bit_path)
        self.hwh_path = str(hwh_path)
        self.backend_name = backend_name
        self.overlay = None
        self._load_overlay()

    def _load_overlay(self):
        try:
            from pynq import Overlay
        except Exception as exc:
            raise ImportError("Backend FPGA phải chạy trên PYNQ-Z2 có thư viện pynq.") from exc

        self.overlay = Overlay(self.bit_path)

    def predict(self, input_q12):
        raise NotImplementedError(
            "PynqRunner là khung tích hợp. Sau khi có cnn_hls.bit/.hwh hoặc cnn_rtl.bit/.hwh, "
            "cần chỉnh register offset và buffer theo IP thật."
        )
