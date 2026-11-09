# Báo cáo hiện trạng dự án BTL

Ngày kiểm tra: 09/11/2026. Thư mục: `/home/dev/btl`.

## 1. Mục tiêu và phạm vi

Dự án xây dựng hệ thống phân loại và truy xuất bản tóm tắt nghiên cứu y khoa bằng tiếng Anh. Phần đã triển khai là chuẩn bị dữ liệu và phân loại văn bản bằng Multinomial Naive Bayes. Phần truy xuất KNN, API và website chưa triển khai.

Năm nhãn hiện tại là bệnh tim mạch, bệnh hệ tiêu hóa, tình trạng bệnh lý chung, khối u và bệnh hệ thần kinh. Đây là nhãn gốc của nguồn dữ liệu; không phải sáu chuyên khoa tự gán nhãn.

## 2. Dữ liệu

Theo README và metadata trong dự án, nguồn là `TimSchopf/medical_abstracts`, gồm 14.438 bản ghi ban đầu; giấy phép được ghi là CC BY-SA 3.0. Thông tin nguồn trong báo cáo này được đọc từ dự án, không kiểm chứng lại trực tuyến.

Pipeline tải CSV từ repository tác giả, hợp nhất thành JSONL, chuẩn hóa văn bản, lọc độ dài, khử trùng và loại toàn bộ nhóm nội dung có nhãn xung đột. Theo `summary.json`, có 6.140 bản ghi bị loại vì nội dung xuất hiện dưới nhiều nhãn; còn 8.298 bản ghi trước bước giới hạn mỗi lớp. Tập thực nghiệm cuối có 4.699 bản ghi.

| Nhóm | Train | Validation | Test | Tổng |
|---|---:|---:|---:|---:|
| Tim mạch | 700 | 150 | 150 | 1.000 |
| Tiêu hóa | 489 | 104 | 106 | 699 |
| Bệnh lý chung | 700 | 150 | 150 | 1.000 |
| Khối u | 700 | 150 | 150 | 1.000 |
| Thần kinh | 700 | 150 | 150 | 1.000 |
| **Tổng** | **3.289** | **704** | **706** | **4.699** |

Chia tập có phân tầng theo lớp, tỷ lệ xấp xỉ 70/15/15, seed 42. Dữ liệu chưa cân bằng hoàn toàn vì lớp tiêu hóa chỉ còn 699 bản ghi. `title` được suy ra từ câu đầu; nguồn không cung cấp PMID, DOI hoặc năm xuất bản theo data dictionary.

Kiểm tra hiện tại xác nhận SHA-256 của ba tập khớp manifest, không trùng ID và không có nội dung trùng giữa các tập theo hàm chuẩn hóa của công cụ audit. Kiểm tra này không đánh giá trùng lặp về ngữ nghĩa. Việc kiểm tra nhãn thủ công vẫn được ghi là `PENDING`.

## 3. Phương pháp và kết quả

Mã nguồn dùng Python standard library, tự triển khai vector hóa thưa và Multinomial Naive Bayes. TF-IDF sử dụng tần suất logarit, IDF có smoothing và chuẩn hóa L2.

Có 27 thí nghiệm: ba họ đặc trưng, ba giá trị `min_df` (2, 3, 5) và ba giá trị `alpha` (0.1, 0.5, 1.0).

| Họ mô hình | Đặc trưng | Macro F1 validation tốt nhất |
|---|---|---:|
| NB-1 | Count unigram | 0,790891 |
| NB-2 | TF-IDF unigram | 0,792570 |
| NB-3 | TF-IDF unigram + bigram | 0,782038 |

Cấu hình được chọn theo Macro F1 validation là **NB-2, `min_df=2`, `alpha=0.1`**. Sau khi chọn cấu hình, mô hình cuối được huấn luyện lại trên train + validation, tổng cộng 3.993 bản ghi, rồi đánh giá trên 706 bản ghi test.

| Chỉ số test | Giá trị |
|---|---:|
| Accuracy | 77,7620% |
| Macro Precision | 78,0306% |
| Macro Recall | 78,2918% |
| Macro F1 | 0,780375 |
| Weighted F1 | 0,773612 |

Mô hình dự đoán đúng 549/706 bản ghi. Bộ từ vựng của cấu hình khi thử trên train có 12.351 đặc trưng; bộ từ vựng trong artifact cuối sau khi học lại trên train + validation có 13.622 đặc trưng.

| Nhóm | F1 test |
|---|---:|
| Tim mạch | 0,842444 |
| Tiêu hóa | 0,888889 |
| Bệnh lý chung | 0,553957 |
| Khối u | 0,854369 |
| Thần kinh | 0,762215 |

Lớp bệnh lý chung là điểm yếu chính: chỉ nhận đúng 77/150 mẫu, trong đó 33 mẫu bị phân sang thần kinh. Phạm vi rộng và sự giao thoa nội dung là giả thuyết cần kiểm tra qua phân tích mẫu lỗi. NB-2 chỉ nhỉnh hơn NB-1 khoảng 0,00168 Macro F1 validation; chưa có đánh giá độ biến thiên qua nhiều seed để kết luận ưu thế ổn định.

## 4. Kiểm chứng ngày 09/11/2026

- `make test`: 9/9 kiểm thử đạt, bao gồm tải/chuyển đổi dữ liệu, xử lý trùng và nhãn xung đột, chia tập, vector hóa, Naive Bayes và metrics.
- Chạy lại suy luận artifact trên toàn bộ 706 mẫu test: các chỉ số Accuracy, Macro Precision, Macro Recall, Macro F1 và Weighted F1 khớp báo cáo đã lưu.
- Dự đoán thử câu `coronary artery disease and myocardial infarction treatment`: trả về `Cardiovascular diseases`.
- Xác nhận báo cáo chứa đủ 27 cấu hình thực nghiệm.

Lần kiểm chứng này dùng artifact hiện có, không huấn luyện lại 27 cấu hình. Kết quả huấn luyện đã lưu có thời điểm 05/11/2026. Việc chạy lại suy luận phục vụ kiểm tra tính nhất quán, không dùng để chọn lại cấu hình.

## 5. Tiến độ và công việc tiếp theo

| Hạng mục | Trạng thái |
|---|---|
| Tải, làm sạch, chia tập và kiểm tra dữ liệu tự động | Đã có |
| Kiểm tra nhãn thủ công | Chưa hoàn tất |
| Thực nghiệm và mô hình Naive Bayes | Đã có, chạy được bằng CLI |
| Báo cáo chỉ số, confusion matrix, từ khóa phân biệt | Đã có |
| Truy xuất KNN | Chưa triển khai |
| API | Chưa triển khai |
| Giao diện web | Chưa triển khai |

Thứ tự đề xuất: hoàn tất kiểm tra nhãn và phân tích lỗi; triển khai truy xuất KNN cùng cách đánh giá chất lượng top-k; xây dựng API phân loại/truy xuất; kết nối giao diện; kiểm tra luồng sử dụng hoàn chỉnh và bổ sung tài liệu trình diễn.

Hiện tại dự án đã có nền tảng dữ liệu và baseline phân loại được kiểm chứng. Để hoàn thành mục tiêu hệ thống phân loại và truy xuất có website, vẫn cần các phần truy xuất, API và giao diện.

## 6. Tệp đối chiếu

- `README.md`, `Makefile`: phạm vi và lệnh chạy.
- `backend/data/processed/summary.json`, `dataset_manifest.json`: thống kê và dấu vết dữ liệu.
- `backend/reports/data_audit.json`: kết quả audit và trạng thái kiểm tra nhãn.
- `backend/reports/nb_experiments.json`, `nb_test_metrics.json`: thí nghiệm và kết quả.
- `backend/reports/confusion_matrix.csv`, `classification_report.csv`: kết quả theo lớp.
- `backend/ml/train_nb.py`, `simple_ml.py`: quy trình huấn luyện và thuật toán.
- `backend/artifacts/naive_bayes_model.pkl`: mô hình đã lưu.
