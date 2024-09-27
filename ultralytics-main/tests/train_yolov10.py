# from ultralytics import YOLO
from ultralytics.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld

if __name__=="__main__":

    # model = YOLO("yolov10n.yaml")
    model = YOLO('yolov10n.pt')  # load a pretrained model (recommended for training)workers=0，batch=4，cache=True)#开始训练
    # model = YOLO("yolov10n.yaml").load("yolov10n.pt")
    results = model.train(data=r'D:\bilibili\model\ultralytics-main\ultralytics\cfg\datasets\VOC_my.yaml',
                          cfg="yolov10n.yaml",
                          epochs=100,
                          imgsz=640,
                          workers=0, batch=4, cache=True)

