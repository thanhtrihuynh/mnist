Copy 8 file weight Q4.12 vào đây.

Nếu xuất theo code Kaggle đã transpose HLS, giữ config.py:
Q12_WEIGHT_FORMAT = "hls"

Nếu file còn shape Keras gốc, đổi:
Q12_WEIGHT_FORMAT = "keras"
