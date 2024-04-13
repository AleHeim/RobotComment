from ultralytics import YOLO
import supervision as sv
import cv2
import os




model = YOLO('runs/detect/train/weights/best.pt')

VIDEO_DIR_PATH = './videos/'
video_paths = sv.list_files_with_extensions(
    directory=VIDEO_DIR_PATH,
    extensions=["mov", "mp4", "mkv"])
print('videos: ', video_paths)

for video_path in video_paths:
    print('\n video_path: ', video_path)
    video_file_path = os.path.join(VIDEO_DIR_PATH, video_path)
    cap = cv2.VideoCapture(video_file_path)
    print('1')
    ret = True

    while(True):
        ret, frame = cap.read()
        print('2')
        results = model.track(frame, persist=True)
        print('3')
        frame_ = results[0].plot()
        print('4')
        cv2.imshow('frame', frame_)
        print('5')
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break