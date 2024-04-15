from ultralytics.data.annotator import auto_annotate

auto_annotate(data='./images', det_model='./runs/detect/train/weights/best.pt', sam_model='mobile_sam.pt', output_dir = './autoannotated')


import os
import json

# Указываем путь к директории с YOLO-файлами
yolo_dir = './autoannotatedYOLO/labels'

# Список YOLO-файлов в директории
yolo_files = [f for f in os.listdir(yolo_dir) if f.endswith(".txt")]

# Функция для чтения аннотаций из YOLO-файла
def read_yolo_annotations(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

# Преобразование YOLO-аннотаций
json_annotations_list = []
for yolo_file in yolo_files:
    yolo_annotations = read_yolo_annotations(os.path.join(yolo_dir, yolo_file))
    json_annotations = []
    for line in yolo_annotations:
        class_id, x_center, y_center, width, height = map(float, line.split())
        x_min = (x_center - width / 2)
        y_min = (y_center - height / 2)
        x_max = (x_center + width / 2)
        y_max = (y_center + height / 2)
        
        annotation_dict = {
            "file_name": yolo_file.replace(".txt", ".jpg"),  # Пример: заменяем расширение для связи с изображением
            "class_id": int(class_id),
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max
        }
        json_annotations.append(annotation_dict)
    json_annotations_list.extend(json_annotations)

# Сохранение JSON аннотаций в файл
with open('annotations.json', 'w') as file:
    json.dump(json_annotations_list, file, indent=2)
