
from ultralytics.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld

if __name__=="__main__":


    # 使用YOLOv8.yamy文件搭建的模型训练
    # model = YOLO(r"D:\bilibili\model\ultralytics-main\ultralytics\cfg\models\v8\yolov8_my.yaml")  # build a new model from YAML
    # results = model.train(data=r'D:\bilibili\model\ultralytics-main\ultralytics\cfg\datasets\VOC_my.yaml',
    #                       epochs=100, imgsz=640, batch=4)
    #
    # # 加载已训练好的模型权重搭建模型训练
    # model = YOLO(r'D:\bilibili\model\ultralytics-main\tests\yolov8n.pt')  # load a pretrained model (recommended for training)
    # results = model.train(data=r'D:\bilibili\model\ultralytics-main\ultralytics\cfg\datasets\VOC_my.yaml',
    #                       epochs=100, imgsz=640, batch=4)

    # 使用自己的YOLOv8.yamy文件搭建模型并加载预训练权重训练模型
    model = YOLO(r"D:\bilibili\model\ultralytics-main\ultralytics\cfg\models\v8\yolov8_my.yaml")\
        .load(r'D:\bilibili\model\ultralytics-main\tests\yolov8n.pt')  # build from YAML and transfer weights

    results = model.train(data=r'D:\bilibili\model\ultralytics-main\ultralytics\cfg\datasets\VOC_my.yaml',
                          epochs=100, imgsz=640, batch=8)



