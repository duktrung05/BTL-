# Data dictionary

| Trường | Kiểu | Ý nghĩa |
|---|---|---|
| `id` | string | `PMID_<pmid>` hoặc mã băm nội dung nếu không có PMID |
| `pmid` | null | Nguồn đã chọn không cung cấp PMID |
| `doi` | null | Nguồn đã chọn không cung cấp DOI |
| `title` | string | Tiêu đề đã chuẩn hóa khoảng trắng |
| `title_is_derived` | boolean | `true` khi title được suy ra từ câu đầu, không dùng để khử trùng |
| `abstract` | string | Abstract đã ghép các section và chuẩn hóa khoảng trắng |
| `category` | string | Một trong năm nhóm bệnh của dataset nguồn |
| `authors` | array[string] | Danh sách tác giả |
| `keywords` | array[string] | Từ khóa tác giả và MeSH terms |
| `year` | null | Dataset nguồn không cung cấp năm |
| `source` | string | `TimSchopf/medical_abstracts` |
| `source_split` | string | Split `train` hoặc `test` ban đầu trên Hugging Face |
| `source_row` | integer | Chỉ số dòng trong split nguồn |
| `license` | string | `CC BY-SA 3.0` |
| `url` | string | Liên kết dataset trên Hugging Face |
| `canonical_repository` | string | Repository CSV chính thức của tác giả |
| `split` | string | `train`, `validation` hoặc `test`; chỉ có sau bước prepare |

