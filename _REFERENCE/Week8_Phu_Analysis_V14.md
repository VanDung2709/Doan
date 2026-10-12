# TUẦN 8 – PHÂN TÍCH DỰ ĐOÁN ĐÚNG/SAI VÀ GIỚI HẠN MÔ HÌNH
**Người phụ trách: Khương Hoàng Phúc**  
**Nguồn kết quả: V14 + Grad-CAM FER fine-tune**

## 1. Mục tiêu
Phân tích các ví dụ dự đoán đúng và sai của mô hình FER-2013 MobileNetV2 đang được sử dụng trong hệ thống V14.  
Grad-CAM được dùng để minh họa vùng ảnh có ảnh hưởng tới quyết định của mô hình; heatmap chỉ hỗ trợ giải thích, không chứng minh quan hệ nhân quả.

## 2. Cấu hình và dữ liệu sử dụng
- Model: `mobilenetv2_fer2013_final.keras`
- Backbone: `mobilenetv2_1.00_224`
- Feature map dùng cho Grad-CAM: `(7, 7, 1280)`
- Bộ kiểm tra: `FER2013/test`
- Số ảnh quét tối đa mỗi lớp để chọn ví dụ: 200
- Số ví dụ đúng được lưu: 7
- Số ví dụ sai được lưu: 7
- Split-forward self-test diff: `0.00000000`
- Max diff model vs split forward: `0.00000000`

Hai giá trị sai khác bằng 0 cho thấy nhánh Grad-CAM đang sử dụng đúng luồng suy luận của model fine-tune, không phải heatmap của model ImageNet cũ.

## 3. Các ví dụ dự đoán đúng
| Nhãn thật | Dự đoán đúng | Độ tin cậy |
|---|---|---:|
| angry | angry | 0.834 |
| disgust | disgust | 0.340 |
| fear | fear | 0.400 |
| happy | happy | 0.541 |
| neutral | neutral | 0.563 |
| sad | sad | 0.690 |
| surprise | surprise | 0.948 |

### Nhận xét
- `surprise` là ví dụ đúng có độ tin cậy cao nhất, đạt `0.948`.
- `angry` cũng được nhận diện khá chắc chắn với `0.834`.
- `disgust` và `fear` tuy dự đoán đúng nhưng confidence chỉ lần lượt `0.340` và `0.400`, cho thấy ranh giới đặc trưng của hai lớp này chưa thật sự rõ.
- Các lớp `happy`, `neutral`, `sad` có mức tin cậy trung bình, phản ánh mô hình có khả năng nhận diện nhưng vẫn còn không gian cải thiện.

## 4. Các ví dụ dự đoán sai
| Nhãn thật | Dự đoán sai thành | Độ tin cậy |
|---|---|---:|
| angry | surprise | 0.753 |
| disgust | angry | 0.744 |
| fear | surprise | 0.789 |
| happy | surprise | 0.900 |
| neutral | sad | 0.702 |
| sad | angry | 0.807 |
| surprise | angry | 0.776 |

### Nhận xét
Một số lỗi có confidence cao, đặc biệt:
- `happy → surprise`: `0.900`
- `sad → angry`: `0.807`
- `fear → surprise`: `0.789`
- `surprise → angry`: `0.776`

Điều này cho thấy confidence cao không đồng nghĩa với dự đoán luôn đúng. Vì vậy hệ thống V14 không nên dùng một khung hình FER đơn lẻ để đưa ra kết luận cuối cùng.

Các cặp lớp dễ bị nhầm trong 14 ví dụ đã chọn:
- `angry ↔ surprise`
- `fear → surprise`
- `sad → angry`
- `neutral → sad`
- `disgust → angry`

Các nhầm lẫn này phù hợp với đặc điểm biểu cảm khuôn mặt có thể chia sẻ các vùng hoạt động tương tự như mắt, lông mày và miệng.

## 5. Vai trò của Grad-CAM
Grad-CAM được sinh từ feature map cuối của MobileNetV2 fine-tune trên FER-2013.  
Khi quan sát các ảnh overlay, Phúc cần kiểm tra:
1. Heatmap có tập trung chủ yếu vào vùng khuôn mặt hay không.
2. Mô hình đang chú ý nhiều tới mắt, lông mày, miệng hay vùng nền.
3. Ở ảnh dự đoán sai, vùng nóng có lệch khỏi vùng biểu cảm chính hay không.
4. Có trường hợp mô hình tập trung vào tóc, nền hoặc phần ảnh không liên quan hay không.

Không nên viết rằng Grad-CAM “chứng minh” nguyên nhân mô hình dự đoán đúng/sai. Grad-CAM chỉ cho thấy vùng ảnh có ảnh hưởng đến dự đoán của mạng.

## 6. Giới hạn mô hình FER
### 6.1. Độ chính xác tổng thể còn giới hạn
Kết quả đã có của FER-2013 trên tập test:
- Accuracy: `52.90%`
- Test Loss: `1.2650`

Do đó FER chỉ nên đóng vai trò tín hiệu hỗ trợ trong hệ thống.

### 6.2. Mất cân bằng dữ liệu
FER-2013 không cân bằng giữa 7 lớp. `disgust` là lớp ít mẫu, trong khi các lớp như `happy` có nhiều mẫu hơn. Mất cân bằng có thể làm mô hình học tốt hơn ở lớp phổ biến và khó phân biệt lớp hiếm.

### 6.3. Biểu cảm giữa các lớp có thể gần nhau
Một ảnh `fear`, `surprise` hoặc `angry` có thể có đặc trưng mắt/lông mày gần nhau. Điều này tạo nhầm lẫn giữa các lớp, đúng với các ví dụ sai đã ghi nhận.

### 6.4. FER-2013 khác môi trường lớp học thật
FER-2013 chủ yếu là ảnh khuôn mặt đã crop với điều kiện khác camera lớp học. Trong V14, học sinh có thể:
- ở xa camera;
- bị che khuất;
- quay đầu;
- ánh sáng thay đổi;
- khuôn mặt nhỏ;
- chỉ xuất hiện một phần.

Vì vậy hiệu năng trên FER-2013 không thể được xem là hiệu năng chính xác tuyệt đối ngoài lớp học.

### 6.5. Một frame không đủ để suy ra trạng thái tâm lý
Một biểu cảm `angry`, `sad` hay `fear` trong một khung hình không thể dùng để chẩn đoán lo âu.  
V14 chỉ dùng FER kết hợp với Engagement và điều kiện kéo dài theo thời gian để tạo **cảnh báo dấu hiệu lo âu**, không phải chẩn đoán tâm lý.

### 6.6. Confidence không phải xác suất đảm bảo đúng
Ví dụ `happy → surprise` có confidence `0.900` nhưng vẫn sai. Vì vậy không dùng confidence FER đơn lẻ làm quyết định cuối.

## 7. Cách V14 giảm ảnh hưởng của các giới hạn
V14 không phụ thuộc duy nhất vào FER mà kết hợp nhiều nguồn:
- YOLO + ByteTrack: theo dõi từng học sinh;
- EAR: nhắm mắt;
- Head Pose: quay đầu, cúi đầu, không nhìn phía trước;
- Body Pose: đứng/ngồi/quay người;
- vị trí ngồi: theo dõi rời chỗ;
- Engagement: tín hiệu hỗ trợ mức độ tập trung;
- FER: tín hiệu cảm xúc;
- Fusion theo thời gian: chỉ cảnh báo khi tín hiệu đủ kéo dài.

Do đó kết quả cuối là hệ thống hỗ trợ giáo viên quan sát, không phải hệ thống chẩn đoán.

## 8. Kết luận phần Phúc – Tuần 8
Đã chọn đủ 7 ví dụ dự đoán đúng và 7 ví dụ dự đoán sai trên 7 lớp FER-2013. Các ảnh đều có Grad-CAM tương ứng từ đúng model MobileNetV2 fine-tune đang được sử dụng trong V14.

Kết quả cho thấy mô hình có thể nhận diện đúng cả 7 lớp ở một số trường hợp, nhưng vẫn tồn tại các lỗi nhầm có confidence cao. Vì vậy nhóm giữ FER ở vai trò tín hiệu hỗ trợ và kết hợp với Engagement, hành vi, thời gian và tracking trong V14.

**Trạng thái phần Phúc – Tuần 8: HOÀN THÀNH về nội dung phân tích đúng/sai và giới hạn mô hình.**
