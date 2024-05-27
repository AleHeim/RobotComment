from ultralytics import YOLO
import cv2
import time as t

model = YOLO('./runs/detect/train3/weights/best.pt')
input('safe start')
results = model.predict('./images/2r-00000.png')
print(results[0].boxes.xywh[0])
for r in results:
     print(f'BOXES: {r.boxes}ENDBOXES')
boxes = results[0].boxes.xywh.cpu()
# track_ids = results[0].boxes.id.int().cpu().tolist()
results[0].save_txt('saved_results')
# print('ids:',track_ids)

robot_ids = {}

def track_video(model, device_id):
    """Track objects from a video"""
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        print(f"Failed to open device {device_id}")
        return
    while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.track(frame, persist=True)
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            for id in track_ids:
                robot_ids[id] = t.time()
            # cv2.imshow('frame', frame_)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()