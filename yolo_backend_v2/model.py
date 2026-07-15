

#polygon

from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO
from PIL import Image


class YOLOSegBackend(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = YOLO(
            r"C:\Users\u117134\Desktop\u-net\runs\segment\YOLO11_SEG\pcb_segmentation\weights\best.pt"
        )

    def predict(self, tasks, context=None, **kwargs):

        predictions = []

        for task in tasks:

            image_url = task["data"]["image"]
            image_path = self.get_local_path(image_url)

            results = self.model.predict(
                image_path,
                conf=0.25,
                verbose=False
            )

            result = []

            if results[0].masks is not None:

                width, height = Image.open(image_path).size

                masks = results[0].masks.xy
                classes = results[0].boxes.cls.cpu().numpy()

                for polygon, cls in zip(masks, classes):

                    points = []

                    for x, y in polygon:
                        points.append([
                            float(x) / float(width) * 100.0,
                            float(y) / float(height) * 100.0
                        ])

                    label = str(self.model.names[int(cls)])

                    result.append({
                        "from_name": "label",
                        "to_name": "image",
                        "type": "polygonlabels",
                        "value": {
                            "points": points,
                            "polygonlabels": [label]
                        }
                    })

            predictions.append({
                "result": result,
                "score": 1.0
            })

        return predictions