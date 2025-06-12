import cv2
import tkinter as tk
from tkinter import filedialog
import threading
from ultralytics import YOLO

# Load mô hình YOLO đã huấn luyện
model = YOLO("best.pt")

# ====== Hàm dùng YOLO để phát hiện khuôn mặt ======
def detect_and_display(frame):
    results = model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# ========== HÀM MỞ CAMERA ==========
def open_camera():
    def run_camera():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = detect_and_display(frame)
            cv2.imshow("Camera - YOLO Detection", frame)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    threading.Thread(target=run_camera).start()

# ========== HÀM CHỌN VÀ PHÁT VIDEO ==========
def open_video():
    video_path = filedialog.askopenfilename(title="Chọn video", filetypes=[("Video files", "*.mp4;*.avi")])
    if not video_path:
        return

    def run_video():
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = detect_and_display(frame)
            cv2.imshow("Video - YOLO Detection", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    threading.Thread(target=run_video).start()

# ========== HÀM CHỌN VÀ HIỂN THỊ ẢNH ==========
def open_image():
    image_path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if not image_path:
        return
    image = cv2.imread(image_path)
    image = detect_and_display(image)
    cv2.imshow("Image - YOLO Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ===== GUI =====
window = tk.Tk()
window.title("Face Detection")
window.geometry("400x200")

btn_camera = tk.Button(window, text="Mở Camera (YOLO)", command=open_camera, width=30, height=2)
btn_video = tk.Button(window, text="Chọn Video (YOLO)", command=open_video, width=30, height=2)
btn_image = tk.Button(window, text="Chọn Ảnh (YOLO)", command=open_image, width=30, height=2)

btn_camera.pack(pady=10)
btn_video.pack(pady=10)
btn_image.pack(pady=10)

window.mainloop()