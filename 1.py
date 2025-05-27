import cv2
from ultralytics import YOLO
import os
# Load a pretrained YOLOv8 model
model = YOLO("runs/detect/train4/weights/best.pt")


def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results


def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results


# Function to create a writer for mp4 videos
def create_video_writer(video_cap, output_filename):
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    # Use 'mp4v' instead of 'MP4V' for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    return writer


# 视频文件夹路径
video_folder_path = r"shangchuan"

# 确保输出目录存在
output_dir = "runs/video3"
os.makedirs(output_dir, exist_ok=True)

# 遍历文件夹中的所有视频文件
for video_file in os.listdir(video_folder_path):
    if video_file.endswith(".MP4"):  # 只处理mp4文件
        video_path = os.path.join(video_folder_path, video_file)
        cap = cv2.VideoCapture(video_path)

        # 设置输出文件名
        output_filename = os.path.join(output_dir, f"processed_{video_file}")

        # 创建视频写入器
        writer = create_video_writer(cap, output_filename)

        while True:
            success, img = cap.read()
            if not success:
                break


            # Perform prediction and detection
            result_img, _ = predict_and_detect(model, img, classes=[], conf=0.5)



            # Write the frame to the output video
            writer.write(result_img)

            # Press 'q' to exit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources for the current video
        cap.release()
        writer.release()

# 关闭所有窗口
cv2.destroyAllWindows()