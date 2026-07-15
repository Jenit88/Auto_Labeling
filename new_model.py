from cmath import rect

import cv2
from matplotlib.pyplot import box
import numpy as np

from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO
from PIL import Image


class YOLOSegBackend(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        print("LOADING NEW_MODEL.PY")

        self.model = YOLO(
            r"C:\Users\u117134\Desktop\Auto_Labeling\best.pt"
        )

        print("CLASSES:", self.model.names)

    def predict(self, tasks, context=None, **kwargs):
            
        print("========== CUSTOM MODEL ==========")
        print("FILE:", __file__)
        print("==================================")

        print("MODEL EXECUTED")

        predictions = []

        for task in tasks:

            image_url = task["data"]["image"]
            image_path = self.get_local_path(image_url)

            results = self.model.predict(
                image_path,
                conf=0.25,
                verbose=False
            )

            width, height = Image.open(image_path).size

            output = []

            if results and results[0].boxes is not None:

                classes = results[0].boxes.cls.cpu().numpy()

                polygons = None
                if results[0].masks is not None:
                    polygons = results[0].masks.xy

                for idx, cls in enumerate(classes):

                    label = str(self.model.names[int(cls)])

                    print("Detected:", label)

                    if polygons is None:
                        continue

                    polygon = polygons[idx]
                    

                    # ==================================================
                    # RECTANGLES
                    # ==================================================
                    if label in ["Rectangle", "Rectangle_concave"]:

                        contour = polygon.astype(np.float32)

                        rect = cv2.minAreaRect(contour)
                        box = cv2.boxPoints(rect)
                        debug_img = cv2.imread(image_path)

                        cv2.polylines(
                            debug_img,
                            [np.int32(box)],
                            True,
                            (0, 255, 0),
                            10
                        )

                        cv2.imwrite(
                            fr"C:\Users\u117134\Desktop\u-net\debug\rect_{idx}.jpg",
                            debug_img
                        )                       
                        print("\n==============================")
                        print("Detected:", label)
                        print("OpenCV box points:")
                        print(box)

                        print("minAreaRect output:")
                        print(rect)
                        print("box points:")

                        # Top-left corner of rotated box
                        (cx, cy), (w_px, h_px), angle = rect

                        if w_px < h_px:
                            w_px, h_px = h_px, w_px
                            angle += 90
                        
                        rotation = (angle + 360) % 360

                        theta = np.radians(rotation)

                        dx = (w_px / 2.0) * np.cos(theta)
                        dy = (w_px / 2.0) * np.sin(theta)

                        hx = (h_px / 2.0) * np.sin(theta)
                        hy = (h_px / 2.0) * np.cos(theta)

                        x1 = cx - dx + hx
                        y1 = cy - dy - hy



                        # Fix opposite-facing rectangles

                        # Percent values for Label Studio
 
                        x = x1 / width * 100.0
                        y = y1 / height * 100.0
                        w = w_px / width * 100.0
                        h = h_px / height * 100.0

                        #if rotation == -90.0 or rotation == 270.0  :
                            
                            #y += w + 0.8
                       
                        #if 0.0 < rotation < 90.0 or -360.0 < rotation < -270.0:
                            
                           # y -= h*np.cos(np.radians(rotation))
                           # x += h*np.sin(np.radians(rotation)) - 0.9
                            
                        output.append({
                            "id": str(len(output)),
                            "from_name": "label",
                            "to_name": "image",
                            "type": "rectanglelabels",
                            "value": {
                                "x": float(x),
                                "y": float(y),
                                "width": float(w),
                                "height": float(h),
                                "rotation": float(rotation),
                                "rectanglelabels": [label]
                            }
                        })
                        print("box points:")
                        for p in box:
                            print(p)
                        print("computed origin:", x1, y1)

                        print(
                            f"ROTATED RECTANGLE: {label} "
                            f"X={x:.2f} "
                            f"Y={y:.2f} "
                            f"W={w:.2f} "
                            f"H={h:.2f} "
                            f"ANGLE={rotation:.2f}"
                        )
                        print("Center:", cx, cy)
                        print("Size:", w_px, h_px)
                        print("Angle:", angle)
                        print("Rotation:", rotation)
                        print("Origin:", x1, y1)
                    # ==================================================
                    # CIRCLES / ELLIPSES
                    # ==================================================
                    elif label in ["circle", "circle_full"]:

                        min_x = float(polygon[:, 0].min())
                        max_x = float(polygon[:, 0].max())

                        min_y = float(polygon[:, 1].min())
                        max_y = float(polygon[:, 1].max())

                        center_x = ((min_x + max_x) / 2.0) / width * 100.0
                        center_y = ((min_y + max_y) / 2.0) / height * 100.0

                        radius_x = ((max_x - min_x) / 2.0) / width * 100.0
                        radius_y = ((max_y - min_y) / 2.0) / height * 100.0

                        output.append({
                            "id": str(len(output)),
                            "from_name": "circles",
                            "to_name": "image",
                            "type": "ellipselabels",
                            "value": {
                                "x": float(center_x),
                                "y": float(center_y),
                                "radiusX": float(radius_x),
                                "radiusY": float(radius_y),
                                "rotation": 0,
                                "ellipselabels": [label]
                            }
                        })

                        print(
                            f"ELLIPSE: {label} "
                            f"X={center_x:.2f} "
                            f"Y={center_y:.2f} "
                            f"radiusX={radius_x:.2f} "
                            f"radiusY={radius_y:.2f}"
                        )

            predictions.append({
                "result": output,
                "score": 1.0
            })

        print("\n===== PREDICTIONS =====")
        print(predictions)
        print("=======================\n")

        return predictions