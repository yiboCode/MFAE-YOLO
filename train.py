from ultralytics import YOLO

model = YOLO("../MFAE-YOLO.yaml")

model.train(
            data="../RSOD.yaml",
            # data="../NWPU.yaml",
            # data="../DIOR.yaml",
            optimizer="AdamW",
            epochs=400,
            iflrAuto=True,
            lr0 = 0.01,
            patience=100,
            imgsz=640,
            batch=64,
            device=0
            )

