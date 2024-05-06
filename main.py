from ultralytics import YOLO
import supervision as sv
import cv2
import os
import time as t

# Constants
VIDEO_EXTENSIONS = ["mov", "mp4", "mkv"]
HOME = "./"  # Adjust this to the appropriate home directory
main_model = YOLO('./runs/detect/train3/weights/best.pt')


# Config
check_bad_detections = True


def create_directories():
    """Create necessary directories if they don't exist."""
    for directory in ['./videos', './images']:
        if not os.path.exists(directory):
            os.makedirs(directory)

def split_video():
    """Split videos into frames."""
    create_directories()
    VIDEO_DIR_PATH = os.path.join(HOME, 'videos')
    IMAGE_DIR_PATH = os.path.join(HOME, 'images')
    FRAME_STRIDE = 50  # Adjust as needed

    video_paths = sv.list_files_with_extensions(VIDEO_DIR_PATH, VIDEO_EXTENSIONS)
    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        image_name_pattern = video_name + "-{:05d}.jpg"
        with sv.ImageSink(IMAGE_DIR_PATH, image_name_pattern=image_name_pattern) as sink:
            for image in sv.get_video_frames_generator(str(video_path), stride=FRAME_STRIDE):
                sink.save_image(image=image)

def track_video_dir(model, source):
    """Track objects in a video."""

    """Find all videos in directory"""
    video_paths = sv.list_files_with_extensions(
        directory=source,
        extensions=VIDEO_EXTENSIONS)
    print('videos: ', video_paths)

    for video_path in video_paths:
        video_file_path = os.path.join(".", video_path)
        track_video(model, video_file_path)

def track_video(model, device_id):
    """Track objects from a video"""
    frame_number = 0
    average_fps = 0.0

    frame_count = 0 # sv.list_files_with_extensions(directory='./bad_detections',extensions=['.jpeg'])
    if not os.path.exists(f'./bad_detections/{device_id}'):
        os.makedirs(f'./bad_detections/{device_id}')

    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        print(f"Failed to open device {device_id}")
        return
    while cap.isOpened():
            # robot_ids = { 1: [], 2: []}
            ret, frame = cap.read()
            if not ret:
                break
            start = t.time()
            results = model.predict(frame)#, persist=True, tracker = 'bytetrack.yaml')
            frame_ = results[0].plot()
            cv2.imshow('frame', frame_)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            track_ids = results[0].boxes
            try:
                track_ids = results[0].boxes.xywh.cpu()
            except AttributeError:
                track_ids = []
            print(f'Len: {len(track_ids)}\n{track_ids}')
            if (len(track_ids) != 2 and check_bad_detections):
                cv2.imwrite(f'./bad_detections/{device_id}/frame_{frame_count:05d}.jpg', frame)
                print(f'Bad detection saved in ./bad_detections/{device_id} as frame_{frame_count:05d}.jpg')
                frame_count+=1
                if(len(track_ids) == 1):
                    print(f'winner: {track_ids[0]}')
            
            end = t.time()
            frame_number = frame_number + 1
            fps = 1.0/(end-start)
            average_fps += fps
            print("FPS: %.1f" % fps)
            print("Average FPS: %.1f" % (average_fps / frame_number) )

    cap.release()
    cv2.destroyAllWindows()

def train_model():
    """Train the YOLO model."""
    model = YOLO('yolov8s.yaml')  # Adjust the configuration file as needed
    results = model.train(data="config.yaml", epochs=110)  # Adjust epochs as needed

def list_devices():
    """List available video devices."""
    devices = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            devices.append(i)
            cap.release()
    print("Available video devices:", devices)

if __name__ == '__main__':
    print('1: Tracking\n2: Training\n3: Split video by frames\nChoose mode and write a number:')
    mode = input()

    if mode == '1':
        print('1: Tracking from device\n2: Tracking from video directory\nChoose input option:')
        input_option = input()

        if input_option == '1':
            list_devices()
            device_id = input('Choose your device: ')
            track_video(main_model, device_id)

        elif input_option == '2':
            video_dir = input('Enter video directory path: ')
            track_video_dir(main_model, video_dir)

    elif mode == '2':
        train_model()

    elif mode == '3':
        split_video()
