from ultralytics import YOLO
import supervision as sv
import cv2
import os




model = YOLO('./runs/detect/train/weights/best.pt')

VIDEO_DIR_PATH = './videos'
video_paths = sv.list_files_with_extensions(
    directory=VIDEO_DIR_PATH,
    extensions=["mov", "mp4", "mkv"])
print('videos: ', video_paths)

for video_path in video_paths:
    print('\n video_path: ', video_path)
    video_file_path = os.path.join(".", video_path)
    print('\n video_file_path: ', video_file_path)
    cap = cv2.VideoCapture(video_file_path)
    ret = True

    while ret:
        ret, frame = cap.read()
        results = model.track(frame, persist=True)
        frame_ = results[0].plot()
        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break