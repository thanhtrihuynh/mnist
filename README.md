# CNN FPGA Web v4 Draw

Bản này đã chỉnh theo yêu cầu:

- Bỏ backend `mock`.
- Bỏ trạng thái hệ thống/debug khỏi giao diện.
- Thêm Top-1 class, Top-2 class, Margin.
- Gộp Upload, Sample, Vẽ số trong cùng một panel để giao diện gọn hơn.
- Panel nạp dữ liệu và panel kết quả nằm song song.
- Thêm chế độ tự vẽ số bằng canvas.
- Vẫn giữ benchmark CSV.

## Cách chạy

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python app.py
```

Mở:

```text
http://127.0.0.1:5000
```

## Đặt file đúng vị trí

Model Keras:

```text
models/best_overall_fpga_mnist_model.keras
```

Weights Q4.12:

```text
weights/q12/conv1_weight_q12.txt
weights/q12/conv1_bias_q12.txt
weights/q12/conv2_weight_q12.txt
weights/q12/conv2_bias_q12.txt
weights/q12/fc1_weight_q12.txt
weights/q12/fc1_bias_q12.txt
weights/q12/fc2_weight_q12.txt
weights/q12/fc2_bias_q12.txt
```

Sample Q4.12:

```text
test_images/q12/input_image_0_label_7_q12.txt
```

Bitstream FPGA:

```text
bitstreams/cnn_hls.bit
bitstreams/cnn_hls.hwh
bitstreams/cnn_rtl.bit
bitstreams/cnn_rtl.hwh
```

## Lưu ý về weight format

Trong `config.py`:

```python
Q12_WEIGHT_FORMAT = "hls"
```

Dùng `"hls"` nếu file weight đã transpose theo dạng HLS.

Nếu file weight còn dạng Keras gốc, đổi thành:

```python
Q12_WEIGHT_FORMAT = "keras"
```


## Lưu ảnh tự vẽ thành dataset

Trong tab `Vẽ số`:

1. Vẽ một chữ số.
2. Nhập nhãn thật của chữ số vào ô `True label`.
3. Bấm `Lưu ảnh vẽ vào dataset`.

Web sẽ tự phân loại theo thư mục nhãn:

```text
draw_dataset/
├── 0/
├── 1/
├── 2/
├── ...
└── 9/
```

Ảnh chuẩn 28x28 dùng để test model sẽ nằm trong:

```text
draw_dataset/<label>/
```

Ảnh raw gốc 280x280 dùng để xem lại nét vẽ sẽ nằm trong:

```text
draw_dataset_raw/<label>/
```

Thông tin metadata được ghi vào:

```text
results/draw_dataset_metadata.csv
```

File này lưu: thời gian, nhãn, đường dẫn ảnh 28x28, đường dẫn ảnh raw.
