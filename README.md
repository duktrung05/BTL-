# BTL — Phân loại và truy xuất abstract y khoa

Đã hoàn thành **Giai đoạn 1** (dữ liệu) và **Giai đoạn 2** (phân loại Naive Bayes). KNN, API và website chưa triển khai.

## Nguồn dữ liệu đã chọn

- Dataset: [TimSchopf/medical_abstracts](https://huggingface.co/datasets/TimSchopf/medical_abstracts)
- Quy mô: 14.438 abstract tiếng Anh
- Định dạng trên Hugging Face: Parquet
- Giấy phép: CC BY-SA 3.0
- Không cần tài khoản, token hoặc email

Nguồn này phù hợp trực tiếp với phân loại và truy xuất văn bản. Dataset có năm lớp:

1. `Neoplasms`
2. `Digestive system diseases`
3. `Nervous system diseases`
4. `Cardiovascular diseases`
5. `General pathological conditions`

Do dùng nhãn gốc của dataset để đảm bảo tính học thuật, phạm vi được điều chỉnh từ sáu chuyên khoa tự gán nhãn xuống năm nhóm bệnh có nhãn sẵn.

## Cấu trúc

```text
btl/
├── backend/
│   ├── data/
│   │   ├── raw/                       # dữ liệu tải từ Hugging Face
│   │   └── processed/                 # dữ liệu đã xử lý
│   ├── scripts/
│   │   ├── download_huggingface.py   # tải dataset bằng hai request
│   │   └── prepare_dataset.py         # làm sạch, cân bằng, chia tập
│   └── tests/
├── Makefile
└── README.md
```

## Chạy pipeline

Yêu cầu Python 3.11 trở lên. Pipeline chỉ dùng Python standard library.

```bash
cd /home/dev/btl
make test
make download
make prepare
```

`make download` tải hai file CSV train/test từ repository chính thức của tác giả, được liên kết ngay trên dataset card Hugging Face, sau đó hợp nhất thành:

```text
backend/data/raw/medical_abstracts.jsonl
```

Cách tải này không gọi API `/rows` theo từng 100 dòng, nhờ đó tránh lỗi `HTTP 429: Too Many Requests`. Nếu máy chủ yêu cầu chờ, script đọc `Retry-After` và tự thử lại. File đầu ra được ghi qua file `.part` để không để lại dataset hỏng nếu tải thất bại.

`make prepare` lấy tối đa 1.000 tài liệu mỗi lớp và chia lại có phân tầng theo tỷ lệ xấp xỉ 70/15/15. Thay giới hạn bằng:

```bash
make prepare MAX_PER_CATEGORY=500
```

Để kiểm tra nhanh với 100 dòng từ mỗi split nguồn:

```bash
make sample
```

## Đầu ra

Thư mục `backend/data/processed/` chứa:

- `all.jsonl`: toàn bộ dữ liệu kèm trường `split`;
- `train.jsonl`, `validation.jsonl`, `test.jsonl`;
- `summary.json`: thống kê dữ liệu và bản ghi bị loại;
- `manual_label_review.csv`: mẫu để kiểm tra nhãn thủ công.

## Kiểm tra dữ liệu

```bash
make audit
```

Lệnh kiểm tra ID trùng, nội dung trùng giữa các split, trường bắt buộc, phân bố lớp và độ dài abstract. Kết quả:

- `backend/data/processed/dataset_manifest.json`
- `backend/reports/data_audit.json`
- `backend/reports/class_distribution.csv`
- `backend/reports/class_distribution.svg`
- `backend/reports/abstract_length_stats.csv`

## Naive Bayes

Chạy toàn bộ 27 thí nghiệm và huấn luyện mô hình cuối:

```bash
make train
```

Ba họ mô hình được so sánh:

| Mô hình | Đặc trưng |
|---|---|
| NB-1 | Count unigram |
| NB-2 | TF-IDF unigram |
| NB-3 | TF-IDF unigram + bigram |

Lưới tham số gồm `min_df ∈ {2,3,5}` và `alpha ∈ {0.1,0.5,1.0}`. Cấu hình được chọn hoàn toàn bằng Macro F1 trên validation; test chỉ được dùng một lần sau khi đã chọn mô hình.

Kết quả hiện tại:

| Chỉ số test | Giá trị |
|---|---:|
| Accuracy | 0,7776 |
| Macro Precision | 0,7803 |
| Macro Recall | 0,7829 |
| Macro F1 | 0,7804 |

Mô hình tốt nhất là NB-2, TF-IDF unigram, `min_df=2`, `alpha=0.1`. Artifact nằm tại `backend/artifacts/naive_bayes_model.pkl`; toàn bộ kết quả và confusion matrix nằm trong `backend/reports/`.

Thử dự đoán:

```bash
make predict QUERY="coronary artery disease and myocardial infarction treatment"
```

Implementation hiện tại dùng Python standard library để chạy được trong môi trường không có NumPy/scikit-learn. Công thức là TF-IDF chuẩn hóa L2 và Multinomial Naive Bayes với additive smoothing; engine sử dụng được ghi trong metadata của model và báo cáo.

## Quy tắc xử lý

- Bỏ abstract dưới 80 ký tự và title dưới 10 ký tự.
- Loại trùng theo PMID/DOI nếu có và toàn bộ nội dung chuẩn hóa. Không dùng title suy diễn của dataset này làm khóa trùng vì nhiều abstract mở đầu bằng cùng một tiêu đề section như `BACKGROUND`.
- Loại toàn bộ nhóm abstract xuất hiện dưới nhiều nhãn và báo là `ambiguous_multilabel_content`; không giữ ngẫu nhiên một nhãn và không để cùng nội dung lọt vào nhiều tập.
- Cân bằng bằng cách giới hạn số tài liệu tối đa ở từng lớp với seed `42`.
- Chia độc lập trong từng lớp để cả ba tập có phân bố nhãn tương tự.
- Giữ `source_split` và `source_row` để truy vết về dữ liệu nguồn.
- Trường `title` được lấy từ câu đầu tiên vì dataset gốc lưu title và abstract chung trong `medical_abstract`.

Khi sử dụng dataset trong báo cáo hoặc sản phẩm phái sinh, cần ghi nguồn tác giả và tuân thủ điều kiện chia sẻ tương tự của giấy phép CC BY-SA 3.0.

