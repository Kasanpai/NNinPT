# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 17:27:20 2024

@author: AM4
"""

# Импортируем библиотеки
import random
import os
import torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import ultralytics
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ultralytics.checks()


def draw_bboxes(image, results):
    boxes = results[0].boxes.cpu()
    class_names = results[0].names

    for box in boxes:
        class_idx = box.cls
        confidence = box.conf

  
        if confidence > 0.5:
            x1, y1, x2, y2 = box.xyxy[0].numpy()

            cv2.rectangle(
                image,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                image,
                f"{class_names[class_idx.item()]} {float(confidence):.2f}",
                (int(x1), int(y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    return image



model = YOLO("yolov8s.pt")
# model = YOLO("yolov8_animals_final.pt")


# results = model.train(
#     data="animals.yaml",   # файл с описанием датасета
#     epochs=20,             # можно увеличить до 30-50
#     batch=8,               # если памяти мало, поставить 4
#     imgsz=640,
#     project="animals",
#     name="exp1",
#     val=True,
#     verbose=True
# )


results = model("test/obj_train_data/test/cat/d2c4df3d06895ef22850a997ef27b75e.jpg")

result = results[0]

# Выводим результат
test_folder = "test/obj_train_data/test"

image_paths = []
for root, dirs, files in os.walk(test_folder):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(root, file))


NUM_IMAGES = 5

# Берем случайные
sample_images = random.sample(image_paths, min(NUM_IMAGES, len(image_paths)))

print("Случайные изображения:")
for path in sample_images:
    print(path)

result = results[0]

for img_path in sample_images:
    print("\nОбработка:", img_path)

    results = model(img_path)
    result = results[0]

    # OpenCV вывод
    cv2.imshow("YOLOv8", result.plot())

    # matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(result.plot()[:, :, ::-1])
    plt.title(f"Результат: {os.path.basename(img_path)}")
    plt.axis("off")
    plt.show()

    # Доп. информация
    print("Рамки:", result.boxes)
    print("Классы:", result.boxes.cls)
    print("Вероятности:", result.boxes.conf)

    # Своя отрисовка
    img = result.orig_img
    annotated_img = draw_bboxes(img.copy(), results)

    cv2.imshow("Custom draw", annotated_img)

    plt.figure(figsize=(8, 6))
    plt.imshow(annotated_img[:, :, ::-1])
    plt.title("Custom draw")
    plt.axis("off")
    plt.show()

    # Пауза между изображениями
    if cv2.waitKey(0) & 0xFF == 27:  # ESC — выход
        break

cv2.destroyAllWindows()