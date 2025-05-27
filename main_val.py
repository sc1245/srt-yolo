from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/train/weights/last.pt')
    model.val(**{'data': 'my_datasets.yaml', 'iou': 0.65})

# from ultralytics import YOLO
#
# # Load a model
# model = YOLO('runs/detect/val3/predictions.json')  # load an official model
# # Validate the model
# metrics = model.val(data='my_datasets.yaml', iou=0.7, conf=0.001, half=False, device=0, save_json=True)