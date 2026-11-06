# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU (Tesla T4, cuda)

**Python / PyTorch / Ultralytics:** Python 3.13.16, PyTorch 2.11.0+cu130, Ultralytics 8.4.145.

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` 

**Thay đổi so với notebook nguồn:** Không 
> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id` = 468; `class_name` = "cab"; `rank` = 1; `score` = 0.510915; `taxonomy_name` = "ImageNet-1K"

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
- Record này nói về cả bức ảnh. Tác vụ Phân loại ảnh (Image Classification) có đơn vị gán nhãn là toàn bức ảnh (image-level). Mô hình chỉ gán 1 nhãn tổng thể bao trùm cho cả khung hình chứ không định vị vị trí của từng vật thể riêng lẻ.
- Vì ảnh hiển thị một ngã tư đông đúc gồm 1 xe buýt lớn, nhiều xe hơi con, xe cảnh sát, người đi bộ và cột đèn. Trong output không có trường tọa độ hộp (bbox_xyxy) hay mặt nạ không gian (spatial localization).

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
- Danh sách tên lớp đi kèm Taxonomy cố định của bộ dữ liệu huấn luyện (ImageNet-1K) gắn liền với checkpoint model, mô hình không thể tự nghĩ ra. Lớp đầu ra tương ứng với 1.000 chỉ số logit cố định của mạng nơ-ron.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- Cần ghi cả 3 vì:
+ taxonomy_name xác định không gian nhãn (namespace), ngăn ngừa xung đột khái niệm giữa các bộ dữ liệu khác nhau.
+ class_id là mã số nguyên cố định, bất biến, dùng để huấn luyện mô hình và lưu trữ cơ sở dữ liệu không bị lỗi bảng mã/chính tả.
+ class_name là chuỗi ngôn ngữ tự nhiên để con người (annotator, reviewer) đọc hiểu được ngay.
Nếu chỉ lưu class_name, khi hệ thống đa ngôn ngữ dịch sang tiếng Việt ("xe taxi") hoặc viết hoa/viết thường ("Cab" vs "cab"), code nạp dữ liệu sẽ bị lỗi không khớp chuỗi; hoặc khi so sánh giữa COCO và ImageNet sẽ không biết đối chiếu vào đâu.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
- Guideline phải quy định rõ Quy tắc ưu tiên (Priority / Salience Rule):
+ Chọn đối tượng chiếm diện tích lớn nhất (dominant object)?
+ Hay chọn đối tượng nằm ở trung tâm điểm nhìn (focal object)?
+ Hay chọn mục tiêu của ứng dụng nghiệp vụ (ví dụ: phát hiện phương tiện công cộng)?
+ Hoặc nếu không thể chọn một, guideline phải yêu cầu chuyển sang bài toán phân loại đa nhãn (Multi-label classification).
Gán nhãn theo cảm tính sẽ gây ra hiện tượng mất nhất quán dữ liệu (inconsistency) — người A chọn xe buýt, người B chọn xe taxi, dẫn đến mô hình bị nhiễu nhãn (label noise) khi huấn luyện.

6. Vì sao không được lấy `score` của model làm ground truth?
- score chỉ là độ tự tin toán học (xác suất sau hàm Softmax) mà mô hình tự ước tính cho phán đoán của nó. Mô hình có thể bị tự tin thái quá (overconfident khi đoán sai) hoặc tự tin quá thấp. Ground truth bắt buộc phải là sự thật khách quan do con người kiểm chứng và khẳng định 100% dựa trên guideline. Mô hình là thứ đang cần được đánh giá hoặc huấn luyện lại. Nếu lấy chính kết quả của nó làm "chân lý", bạn đang tạo ra vòng lặp thiên kiến (confirmation bias), làm sai lệch hoàn toàn tập đánh giá kiểm thử.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name` = "bowl"; `score` = 0.701282; `bbox_xyxy` = [155.02, 168.67, 182.38, 184.18]; `bbox_width` = 27.36; `bbox_height` = 15.51

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
- Hộp bao của vật thể (cái bát) bắt đầu từ góc trên-bên-trái tại toạ độ pixel (155.02, 168.67) và kéo dài xuống góc dưới-bên-phải tại toạ độ pixel (182.38, 184.18). Chiều rộng của hộp bao chiếm 27.36 pixel theo phương ngang và chiều cao chiếm 15.51 pixel theo phương dọc.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- threshold=0.20: 17 vật thể → ['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl', 'spoon', 'potted plant', 'spoon', 'dining table', 'spoon', 'bottle']
- threshold=0.60: 6 vật thể → ['person', 'bowl', 'bowl', 'oven', 'oven', 'person']

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
- Reviewer phải tốn nhiều thời gian và công sức để sàng lọc, kiểm tra và xóa bỏ các hộp dự đoán sai (False Positives), loại bỏ các hộp bị gán nhãn ma (nhiễu nền) hoặc các hộp trùng lặp (duplicate boxes).

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- Bounding box phải ôm sát đường viền ngoài cùng của toàn bộ phần nhìn thấy của vật thể (tight bounding box), dung sai khoảng trống thừa không quá 2 pixel và tuyệt đối không cắt lẹm vào bất kỳ bộ phận nào của vật thể.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- Trên ảnh `kitchen`: Người phụ nữ đứng bên trái bị mép viền ảnh cắt ngang thân (truncation), và các chiếc bát đặt trên bàn bị tay hoặc bát khác che khuất một phần (occlusion).
- Xử lý: Đưa vào **Guideline**:
  + Quy định quy tắc bao đóng phần nhìn thấy (visible bounding box): Annotator chỉ khoanh vùng phần cơ thể / vật thể thực tế nhìn thấy được trong ảnh, không đoán mò phần bị mép cắt hay vật khác che mất.
  + Đưa ra ngưỡng diện tích tối thiểu: Nếu phần nhìn thấy < 15-20% diện tích vật thể hoặc không đủ đặc trưng nhận diện thì bỏ qua không gán nhãn.
  + Lý do: Đây là các tình huống cắt mép và che khuất rất phổ biến trong thực tế. Cần chuẩn hóa thành điều khoản rõ ràng trong guideline để đội ngũ annotator làm việc nhất quán với năng suất cao, không đưa lên escalation làm nghẽn tiến độ.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id` = "kitchen-006"; `class_name` = "bowl"; `score` = 0.528201; số điểm polygon = `polygon_point_count` = 84; 3–5 điểm đầu của `polygon_xy` = [ [29.0, 303.0], [28.0, 304.0], [26.0, 304.0], [25.0, 305.0], [23.0, 305.0] ]

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- Polygon cung cấp đường biên hình học chính xác ở cấp độ pixel (pixel-level mask), phản ánh đúng độ cong, góc cạnh và hình dạng thực tế của vật thể; loại bỏ hoàn toàn các vùng nền vô nghĩa bị kẹt bên trong hình chữ nhật.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- instance_id dùng để phân biệt từng cá thể đối tượng độc lập trong cùng một bức ảnh.
- Nó không phải class ID vì tất cả các cái bát đều có chung class_id = 45.
- Nó không phải tracking ID vì nó chỉ có giá trị cục bộ trong một ảnh tĩnh của bài lab, không duy trì danh tính liên tục của đối tượng xuyên suốt qua chuỗi khung hình video thời gian.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- Đường biên đa giác (polygon) phải bám khít mép thực tế của vật thể với dung sai không quá 2 pixel; phải khoét rỗng (donut hole) các khoảng nền nhìn xuyên qua được (như quai ấm trà, khe giữa cánh tay và hông) và không được lấn ranh giới sang vật thể kế bên.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- Với trường hợp tiếp xúc/chồng lấn rõ ràng: Dùng Guideline (quy định: vật phía trước được vẽ toàn vẹn, vật phía sau ngắt đường biên tại điểm tiếp giáp).
- Với trường hợp biên mờ/đồng màu (bóng đổ làm tối sầm không rõ mép đáy bát): Dùng Escalation để Lab Coach/Lead thống nhất chuẩn chung cho cả đội ngũ.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn phân loại duy nhất (`class_id` / `class_name`, vd: `468: cab`) cho toàn bộ file ảnh (cấp độ image-level); không chứa tọa độ hay vị trí không gian. | Trên ảnh `traffic`, khung cảnh chứa nhiều đối tượng cùng lúc (xe buýt lớn, xe con, người đi bộ). Model đoán `cab` chỉ với score `0.510915` (phân vân với `minibus` 0.16, `police_van` 0.08), không thể hiện hết toàn bộ bức ảnh. | Đọc kỹ quy tắc ưu tiên trong guideline (chọn đối tượng chiếm diện tích lớn nhất, nằm ở trung tâm hay bối cảnh chính) để gán đúng 1 nhãn duy nhất cho bức ảnh. | Kiểm tra nhãn được chọn có tuân thủ đúng quy tắc ưu tiên không; đánh giá các trường hợp nhập nhằng giữa các lớp xe tương đồng để tránh thiên vị nhãn. |
| Phát hiện vật thể | Tập hợp các Bounding Box pixel dạng `bbox_xyxy = [x_min, y_min, x_max, y_max]` kèm `class_id` cho từng vật thể nhìn thấy (cấp độ object-level). | Trên ảnh `kitchen`, người phụ nữ bị mép ảnh cắt ngang thân, nhiều chiếc bát bị tay hoặc bát khác che khuất một phần. Model ở ngưỡng 0.60 bị sót vật thể nhỏ (thìa, chai lọ); ở ngưỡng 0.20 lại sinh ra nhiều hộp nhiễu/sai. | Dùng công cụ kéo hộp bao ôm sát mép ngoài của toàn bộ phần nhìn thấy của từng vật thể (tight box) theo danh mục 80 lớp COCO, không bỏ sót đối tượng rõ nét. | Kiểm tra độ khít của hộp bao (IoU), đảm bảo không cắt lẹm vào vật thể và không thừa nền; rà soát loại bỏ các hộp gán nhãn ma (False Positives) hoặc bổ sung hộp bị sót. |
| Instance segmentation | Tập hợp chuỗi tọa độ đa giác `polygon_xy = [[x1, y1], [x2, y2], ...]` kèm `class_id` và mã `instance_id` riêng biệt cho từng cá thể (cấp độ pixel-level mask). | Trên ảnh `kitchen`, các chiếc bát sứ (`bowl`) xếp chồng/sát nhau và bàn tay người chạm đè lên bát; đường viền đa giác của model bị răng cưa và dễ bị lem/dính ranh giới giữa hai vật thể liền kề. | Dùng công cụ polygon/brush vẽ tỉ mỉ viền bám sát biên dạng thực tế từng cá thể; khoét rỗng các vùng nền nhìn xuyên qua được (như quai cốc, khoảng hở cánh tay) và tách biệt rõ ranh giới các vật. | Phóng to (zoom in) kiểm tra độ bám biên mask (dung sai <= 2-3 pixel), kiểm tra lỗ khoét rỗng (donut holes); đảm bảo mỗi cá thể có 1 `instance_id` riêng biệt, không bị gộp chung mask. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- Chỉ sử dụng tập hình ảnh công khai COCO đã được kiểm tra checksum trong bài lab; tuyệt đối không tải lên Colab hay GitHub bất kỳ hình ảnh cá nhân, khuôn mặt, biển số xe, dữ liệu khách hàng hay thông tin nội bộ của công ty.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
- Lập tức dừng mọi thao tác xử lý/gán nhãn trên dữ liệu đó, giữ nguyên hiện trường, tuyệt đối không tải về hay phát tán và báo cáo ngay cho Giảng viên / Lab Coach phụ trách để kiểm tra, cô lập và thu hồi.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
