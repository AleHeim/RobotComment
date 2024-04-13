# import os
# HOME = os.getcwd()
import torch

# from IPython import display
# display.clear_output()

# import ultralytics
# ultralytics.checks()

from ultralytics import YOLO

model = YOLO('yolov8n.yaml') # Build a model from scratch. In yolo8n.yaml - "n" means nano. (n,s,m,l,x) в порядке возрастания

if __name__ == '__main__':
    results = model.train(data="config.yaml", epochs = 100) #train the model data="файл конфигурации" в файле описаны пути к файлам и классы объектов которые нужно обнаружить
    # epochs - глубина тренировки