# Nhận diện khuôn mặt với YOLO và OpenCV

Đề tài XÂY DỰNG ỨNG DỤNG NHẬN DIỆN KHUÔN MẶT VỚI YOLOv11 VÀ OPENCV.

## Quy trình hoạt động

1. Chụp ảnh từ webcam bằng OpenCV.
2. Gửi ảnh vào mô hình YOLO để phát hiện khuôn mặt.
3. Nhận kết quả từ YOLO (bounding boxes).
4. OpenCV vẽ khung lên khuôn mặt và hiển thị ảnh đã xử lý.
   
## Hướng dẫn sử dụng
1. Tải các thư viện cần thiết bằng pip
2. Chạy file main.py(`python main.py`)
3. Lựa chọn chức năng trong GUI
   - Option 1: Nhận diện thông qua webcam
   - Option 2: Nhận diện thông qua video được nhập
   - Option 3: Nhận diện thông qua hình dảnh được nhập
4. Nhấn nút "q" trên bàn phím để thoát chức năng
5. Nhấn nút "q" trên bàn phím để thoát chương trình

## Tải về dự án

Do giới hạn dung lượng GitHub, bạn có thể tải toàn bộ dự án dưới dạng `.rar` tại đây:

👉 [Tải xuống ttnt.rar](https://drive.google.com/file/d/1h2F_Na10bF5FW7ZqNxlnOnRHpd8k_dA9/view?usp=sharing)

---

## 🔧 Yêu cầu

- Python 3.x
- OpenCV (`pip install opencv-python`)
- Ultralytics YOLOv11 (`pip install ultralytics`)
- Giao diện Đồ họa Thinker(`import tkinter`)



