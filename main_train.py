from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v6/yolov6n.yaml')
    # model.load('yolov6n.pt')
    model.train(**{'cfg': 'ultralytics/cfg/default.yaml', 'data': 'my_datasets.yaml'})
