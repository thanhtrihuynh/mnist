from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"

MODEL_PATH = BASE_DIR / "models" / "best_overall_fpga_mnist_model.keras"

WEIGHT_Q12_DIR = BASE_DIR / "weights" / "q12"

TEST_Q12_DIR = BASE_DIR / "test_images" / "q12"
TEST_UINT8_DIR = BASE_DIR / "test_images" / "uint8"

BITSTREAM_DIR = BASE_DIR / "bitstreams"
HLS_BIT_PATH = BITSTREAM_DIR / "cnn_hls.bit"
HLS_HWH_PATH = BITSTREAM_DIR / "cnn_hls.hwh"
RTL_BIT_PATH = BITSTREAM_DIR / "cnn_rtl.bit"
RTL_HWH_PATH = BITSTREAM_DIR / "cnn_rtl.hwh"

BENCHMARK_CSV_PATH = RESULT_DIR / "benchmark_results.csv"

ALLOWED_EXTENSIONS = {".txt", ".csv", ".mem", ".png", ".jpg", ".jpeg", ".bmp"}

# Đã bỏ mock theo yêu cầu.
BACKENDS = ["cpu_q12", "cpu_float", "fpga_hls", "fpga_rtl"]
DEFAULT_BACKEND = "cpu_q12"

# Dùng "hls" nếu file weight đã transpose theo dạng:
# conv: [out_c, in_c, kh, kw], dense: [out, in]
# Dùng "keras" nếu file weight còn dạng:
# conv: [kh, kw, in_c, out_c], dense: [in, out]
Q12_WEIGHT_FORMAT = "hls"

DRAW_DATASET_DIR = BASE_DIR / "draw_dataset"
DRAW_DATASET_RAW_DIR = BASE_DIR / "draw_dataset_raw"
DRAW_DATASET_CSV_PATH = RESULT_DIR / "draw_dataset_metadata.csv"
