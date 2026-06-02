from pathlib import Path
import base64
import io
import re

import numpy as np
from PIL import Image, ImageOps

FRAC_BITS = 12
SCALE = 1 << FRAC_BITS

def q12_from_float_array(arr):
    arr = np.asarray(arr, dtype=np.float32)
    q = np.rint(arr * SCALE)
    q = np.clip(q, -32768, 32767).astype(np.int16)
    return q

def load_numeric_file(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    nums = re.findall(r"[-+]?\d+", text)

    if len(nums) != 784:
        raise ValueError(f"File {path.name} phải chứa đúng 784 số Q4.12, hiện có {len(nums)} số.")

    arr = np.array([int(x) for x in nums], dtype=np.int32)

    if arr.min() < -32768 or arr.max() > 32767:
        raise ValueError("Giá trị Q4.12 phải nằm trong khoảng int16 [-32768, 32767].")

    return arr.astype(np.int16)

def decode_data_url_image(data_url):
    if "," not in data_url:
        raise ValueError("Dữ liệu canvas không hợp lệ.")

    _, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(img_bytes)).convert("RGBA")

def preprocess_pil_image_to_mnist_pil(img):
    # Đưa ảnh về grayscale, nền đen, nét trắng, kích thước 28x28.
    # Với canvas của web: nền đen, nét trắng, nên gần như chỉ cần resize.
    img = img.convert("L")

    # Nếu ảnh upload là nền trắng/chữ đen thì đảo màu về chuẩn MNIST.
    arr = np.asarray(img, dtype=np.float32)
    if arr.mean() > 127:
        img = ImageOps.invert(img)

    img = ImageOps.contain(img, (28, 28))
    canvas = Image.new("L", (28, 28), color=0)

    x = (28 - img.width) // 2
    y = (28 - img.height) // 2
    canvas.paste(img, (x, y))

    return canvas

def preprocess_pil_image_to_q12(img):
    mnist_img = preprocess_pil_image_to_mnist_pil(img)
    arr_float = np.asarray(mnist_img, dtype=np.float32) / 255.0
    return q12_from_float_array(arr_float.reshape(-1))

def load_image_file_as_q12(path):
    img = Image.open(path)
    return preprocess_pil_image_to_q12(img)

def load_input_as_q12(path):
    path = Path(path)
    ext = path.suffix.lower()

    if ext in [".txt", ".csv", ".mem"]:
        return load_numeric_file(path)

    if ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        return load_image_file_as_q12(path)

    raise ValueError(f"Không hỗ trợ định dạng file: {ext}")

def load_data_url_image_as_q12(data_url):
    img = decode_data_url_image(data_url)
    return preprocess_pil_image_to_q12(img)

def data_url_to_raw_and_mnist_images(data_url):
    raw = decode_data_url_image(data_url)

    # Lưu raw dạng RGB để xem lại nét vẽ gốc 280x280.
    raw_rgb = raw.convert("RGB")

    # Lưu ảnh chuẩn MNIST 28x28 để dùng làm dataset test.
    mnist_img = preprocess_pil_image_to_mnist_pil(raw_rgb).convert("L")

    return raw_rgb, mnist_img

def infer_label_from_filename(filename):
    name = Path(filename).name
    match = re.search(r"label[_-]?(\d)", name)
    if match:
        return int(match.group(1))
    return None

def summarize_input(input_q12):
    arr = np.asarray(input_q12, dtype=np.int32)
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "num_values": int(arr.size),
    }
