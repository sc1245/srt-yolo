from ultralytics import YOLO
import numpy as np


if __name__ == '__main__':
    model = YOLO('runs/detect/train3/weights/best.pt')
    model.predict(source='./测试/2024_11_04_15_09_IMG_3351.MOV', **{'save':True, 'show_conf':True}, stream=True)
    # #模型推理
    # import cv2
    # model = YOLO('yolov8s.pt')  # load an official model
    # results = model.predict('ultralytics/assets/bus.jpg',classes=0,save=True)