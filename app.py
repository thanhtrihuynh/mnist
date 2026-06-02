from pathlib import Path
import csv
import datetime
import json
import uuid

from flask import Flask, render_template, request, jsonify, send_file

from config import (
    UPLOAD_DIR, RESULT_DIR, BENCHMARK_CSV_PATH,
    MODEL_PATH, WEIGHT_Q12_DIR, TEST_Q12_DIR, TEST_UINT8_DIR,
    HLS_BIT_PATH, HLS_HWH_PATH, RTL_BIT_PATH, RTL_HWH_PATH,
    BACKENDS, DEFAULT_BACKEND, ALLOWED_EXTENSIONS, Q12_WEIGHT_FORMAT,
    DRAW_DATASET_DIR, DRAW_DATASET_RAW_DIR, DRAW_DATASET_CSV_PATH
)
from input_utils import load_input_as_q12, load_data_url_image_as_q12, data_url_to_raw_and_mnist_images, infer_label_from_filename, summarize_input
from fpga_runner import PynqRunner

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

_runner_cache = {}

def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def get_runner(backend):
    if backend in _runner_cache:
        return _runner_cache[backend]

    if backend == "cpu_q12":
        from cpu_q12_runner import CPUQ12Runner
        runner = CPUQ12Runner(WEIGHT_Q12_DIR, Q12_WEIGHT_FORMAT)

    elif backend == "cpu_float":
        from cpu_float_runner import CPUFloatRunner
        runner = CPUFloatRunner(MODEL_PATH)

    elif backend == "fpga_hls":
        runner = PynqRunner(HLS_BIT_PATH, HLS_HWH_PATH, "fpga_hls")

    elif backend == "fpga_rtl":
        runner = PynqRunner(RTL_BIT_PATH, RTL_HWH_PATH, "fpga_rtl")

    else:
        raise ValueError(f"Backend không hợp lệ: {backend}")

    _runner_cache[backend] = runner
    return runner

def ensure_benchmark_header():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if not BENCHMARK_CSV_PATH.exists():
        with open(BENCHMARK_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "backend", "input_file", "true_label", "predicted_label",
                "correct", "top1_class", "top1_logit", "top2_class", "top2_logit",
                "margin", "latency_ms", "throughput_fps", "logits", "note"
            ])

def append_benchmark(row):
    ensure_benchmark_header()
    with open(BENCHMARK_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def list_sample_files():
    files = []

    for folder, kind in [(TEST_Q12_DIR, "q12"), (TEST_UINT8_DIR, "uint8")]:
        if folder.exists():
            for path in sorted(folder.glob("*")):
                if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
                    files.append({
                        "name": path.name,
                        "kind": kind,
                        "path": str(path.relative_to(Path(__file__).resolve().parent))
                    })

    return files

def common_result_payload(result, input_q12, input_name, true_label):
    correct = None

    if true_label is not None:
        correct = int(result["predicted_label"] == int(true_label))

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    append_benchmark([
        timestamp,
        result.get("backend", ""),
        input_name,
        true_label if true_label is not None else "",
        result["predicted_label"],
        correct if correct is not None else "",
        result.get("top1_class", ""),
        result.get("top1_logit", ""),
        result.get("top2_class", ""),
        result.get("top2_logit", ""),
        result.get("margin", ""),
        f"{result['latency_ms']:.6f}",
        f"{result['throughput_fps']:.6f}",
        json.dumps(result["logits"], ensure_ascii=False),
        result.get("note", "")
    ])

    return {
        "ok": True,
        "timestamp": timestamp,
        "input_file": input_name,
        "true_label": true_label,
        "correct": correct,
        "input_summary": summarize_input(input_q12),
        **result
    }


def ensure_draw_dataset_dirs():
    DRAW_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    DRAW_DATASET_RAW_DIR.mkdir(parents=True, exist_ok=True)

    for label in range(10):
        (DRAW_DATASET_DIR / str(label)).mkdir(parents=True, exist_ok=True)
        (DRAW_DATASET_RAW_DIR / str(label)).mkdir(parents=True, exist_ok=True)

def ensure_draw_metadata_header():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if not DRAW_DATASET_CSV_PATH.exists():
        with open(DRAW_DATASET_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "label", "mnist_28x28_path", "raw_path",
                "mnist_filename", "raw_filename"
            ])

def append_draw_metadata(row):
    ensure_draw_metadata_header()
    with open(DRAW_DATASET_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

@app.route("/")
def index():
    return render_template(
        "index.html",
        backends=BACKENDS,
        default_backend=DEFAULT_BACKEND,
        sample_files=list_sample_files()
    )

@app.route("/api/predict_upload", methods=["POST"])
def api_predict_upload():
    backend = request.form.get("backend", DEFAULT_BACKEND)
    true_label_raw = request.form.get("true_label", "").strip()

    if backend not in BACKENDS:
        return jsonify({"ok": False, "error": f"Backend không hợp lệ: {backend}"}), 400

    if "file" not in request.files:
        return jsonify({"ok": False, "error": "Chưa upload file input."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"ok": False, "error": "Tên file rỗng."}), 400

    if not allowed_file(file.filename):
        return jsonify({"ok": False, "error": "Định dạng file không được hỗ trợ."}), 400

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    saved_path = UPLOAD_DIR / safe_name
    file.save(saved_path)

    try:
        input_q12 = load_input_as_q12(saved_path)
        runner = get_runner(backend)
        result = runner.predict(input_q12)

        inferred_label = infer_label_from_filename(file.filename)
        true_label = int(true_label_raw) if true_label_raw != "" else inferred_label

        return jsonify(common_result_payload(result, input_q12, file.filename, true_label))

    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

@app.route("/api/predict_sample", methods=["POST"])
def api_predict_sample():
    data = request.get_json(force=True)
    backend = data.get("backend", DEFAULT_BACKEND)
    rel_path = data.get("sample_path", "")

    base_dir = Path(__file__).resolve().parent
    sample_path = (base_dir / rel_path).resolve()

    allowed_roots = [TEST_Q12_DIR.resolve(), TEST_UINT8_DIR.resolve()]

    if backend not in BACKENDS:
        return jsonify({"ok": False, "error": f"Backend không hợp lệ: {backend}"}), 400

    if not any(str(sample_path).startswith(str(root)) for root in allowed_roots):
        return jsonify({"ok": False, "error": "Sample path không hợp lệ."}), 400

    if not sample_path.exists():
        return jsonify({"ok": False, "error": f"Không tìm thấy sample: {sample_path.name}"}), 404

    try:
        input_q12 = load_input_as_q12(sample_path)
        runner = get_runner(backend)
        result = runner.predict(input_q12)

        true_label = infer_label_from_filename(sample_path.name)

        return jsonify(common_result_payload(result, input_q12, sample_path.name, true_label))

    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

@app.route("/api/predict_drawing", methods=["POST"])
def api_predict_drawing():
    data = request.get_json(force=True)
    backend = data.get("backend", DEFAULT_BACKEND)
    true_label_raw = str(data.get("true_label", "")).strip()
    image_data = data.get("image_data", "")

    if backend not in BACKENDS:
        return jsonify({"ok": False, "error": f"Backend không hợp lệ: {backend}"}), 400

    try:
        input_q12 = load_data_url_image_as_q12(image_data)
        runner = get_runner(backend)
        result = runner.predict(input_q12)

        true_label = int(true_label_raw) if true_label_raw != "" else None

        return jsonify(common_result_payload(result, input_q12, "canvas_drawing.png", true_label))

    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/save_drawing", methods=["POST"])
def api_save_drawing():
    data = request.get_json(force=True)

    label_raw = str(data.get("label", "")).strip()
    image_data = data.get("image_data", "")

    if label_raw == "":
        return jsonify({"ok": False, "error": "Cần nhập nhãn từ 0 đến 9 trước khi lưu ảnh vẽ."}), 400

    try:
        label = int(label_raw)
    except ValueError:
        return jsonify({"ok": False, "error": "Nhãn phải là số nguyên từ 0 đến 9."}), 400

    if label < 0 or label > 9:
        return jsonify({"ok": False, "error": "Nhãn phải nằm trong khoảng 0 đến 9."}), 400

    try:
        ensure_draw_dataset_dirs()

        raw_img, mnist_img = data_url_to_raw_and_mnist_images(image_data)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:8]

        mnist_filename = f"draw_label_{label}_{timestamp}_{uid}_28x28.png"
        raw_filename = f"draw_label_{label}_{timestamp}_{uid}_raw.png"

        mnist_path = DRAW_DATASET_DIR / str(label) / mnist_filename
        raw_path = DRAW_DATASET_RAW_DIR / str(label) / raw_filename

        mnist_img.save(mnist_path)
        raw_img.save(raw_path)

        meta_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        append_draw_metadata([
            meta_time,
            label,
            str(mnist_path.relative_to(Path(__file__).resolve().parent)),
            str(raw_path.relative_to(Path(__file__).resolve().parent)),
            mnist_filename,
            raw_filename
        ])

        return jsonify({
            "ok": True,
            "message": f"Đã lưu ảnh vẽ vào thư mục nhãn {label}.",
            "label": label,
            "mnist_28x28_path": str(mnist_path.relative_to(Path(__file__).resolve().parent)),
            "raw_path": str(raw_path.relative_to(Path(__file__).resolve().parent)),
            "metadata_csv": str(DRAW_DATASET_CSV_PATH.relative_to(Path(__file__).resolve().parent))
        })

    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

@app.route("/download/benchmark")
def download_benchmark():
    ensure_benchmark_header()
    return send_file(BENCHMARK_CSV_PATH, as_attachment=True)

@app.route("/api/clear_cache", methods=["POST"])
def clear_cache():
    _runner_cache.clear()
    return jsonify({"ok": True, "message": "Đã load lại model/weights."})

if __name__ == "__main__":
    ensure_benchmark_header()
    ensure_draw_dataset_dirs()
    ensure_draw_metadata_header()
    app.run(host="0.0.0.0", port=5000, debug=True)
