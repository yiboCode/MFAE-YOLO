from ultralytics import YOLO
model = YOLO(
    "../model.pt")
results = model.val(
    split='test',
    data="../NWPU.yaml",
    # data="../RSOD.yaml",
    # data="../DIOR.yaml",
    imgsz=640,
    batch=64,
    device=0
    )
