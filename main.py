from ultralytics import YOLO
import supervision as sv
import cv2
import os
import time as t
import argparse


# Constants
VIDEO_EXTENSIONS = ["mov", "mp4", "mkv"]
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]
HOME = "./"  # Adjust this to the appropriate home directory
main_model = YOLO('./runs/detect/train3/weights/best.pt')


'''arguments parser'''
parser = argparse.ArgumentParser()
parser.add_argument("mode", type=str, choices=['track', 'train', 'split', 'label'], help="Track: Track object.\nTrain: Train model.\nSplit: Split video by frames.\nLabel: auto-labeling tool.\n")
parser.add_argument("-cbd", "--check_bad_detections", type=int, nargs="?", default=False, help="usage: -cbd <expected amount of objects>")
parser.add_argument("-dir", "--directory", action="store_true")
parser.add_argument("-i", "--input", type=str,default=f"{HOME}videos", help="input device, directory or http/https address")

# Добавить аргументы

args = parser.parse_args()

'''Для аргумента --cbd проверяет, введено ли число'''
if args.check_bad_detections == None:
    print(f"error: correct usage: -cbd <expected amount of objects>\nfor example: -cbd 2")
    quit()

print(args)


def create_directories():
    """Create necessary directories if they don't exist."""
    for directory in [f'{HOME}videos', f'{HOME}images']:
        if not os.path.exists(directory):
            os.makedirs(directory)
    return 0

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
    return 0

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
    return 0

def track_video(model, device_id):
    """Track objects from a video"""
    timeout = None
    frame_number = 0
    average_fps = 0.0

    frame_count = 0 # sv.list_files_with_extensions(directory='./datasets/bad_detections',extensions=['.jpeg'])
    if not os.path.exists(f'./datasets/bad_detections/{device_id}'):
        os.makedirs(f'./datasets/bad_detections/{device_id}')

    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        print(f"Failed to open device {device_id}")
        return
    while cap.isOpened():
            # robot_ids = { 1: [], 2: []}
            ret, frame = cap.read()
            if not ret:
                break
            start = t.time() #FPS COUNTER
            results = model.predict(frame, verbose=False)#, persist=True, tracker = 'bytetrack.yaml', verbose=False)
            # frame_ = results[0].plot()
            # cv2.imshow('frame', frame_)     ### ADDITIONAL LINES FOR model.track()
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            track_ids = results[0].boxes
            try:
                track_ids = results[0].boxes.cls
            except AttributeError:
                track_ids = []
            if (len(track_ids) != args.check_bad_detections and args.check_bad_detections is not False):
                cv2.imwrite(f'./datasets/bad_detections/{device_id}/frame_{frame_count:05d}.jpg', frame)
                print(f'Bad detection saved in ./datasets/bad_detections/{device_id} as frame_{frame_count:05d}.jpg')
                frame_count+=1
            if(len(track_ids) == 1 and args.check_bad_detections is False):
                if(timeout is not None):
                    if(t.time()-timeout>=5):
                        if(input("winner? write y/n")=="y"):
                            print(f'winner: {track_ids[0]}')
                            break
                        else:
                            timeout = None
                else:
                    timeout=t.time()
                    # print(f'winner: {track_ids[0]}')
            else:
                timeout = None
            
            
            end = t.time() #FPS COUNTER
            frame_number = frame_number + 1
            fps = 1.0/(end-start)
            average_fps += fps
            print("FPS: %.1f" % fps)
            print("Average FPS: %.1f" % (average_fps / frame_number) )

    cap.release()
    cv2.destroyAllWindows()
    return 0

def train_model():
    """Train the YOLO model."""
    model = YOLO('yolov8s.yaml')  # Adjust the configuration file as needed
    results = model.train(data="config.yaml", epochs=110)  # Adjust epochs as needed
    return 0

def list_devices():
    """List available video devices."""
    devices = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            devices.append(i)
            cap.release()
    print("Available video devices:", devices)
    return 0

def autolabel(model, images_dir, save_dir):
    image_paths = sv.list_files_with_extensions(
    directory=images_dir,
    extensions=IMAGE_EXTENSIONS)
    print('images: ', image_paths)
    for image in image_paths:
        results = main_model.predict(image)
        image_name = os.path.basename(image)

        results[0].save_txt(f'{save_dir}/{image_name[:image_name.rindex('.')]}.txt')
    return 0


if __name__ == '__main__':
    match args.mode:
        case "track":
            if(args.directory):
                track_video_dir(main_model, args.input)
            else:
                track_video(main_model, args.input)
        case "train":
            train_model()
        case "split":
            split_video()
        case "label":
            images_dir = './datasets/one/images'
            save_dir = './datasets/one/labels'
            autolabel(main_model, images_dir, save_dir)



'''
    print('1: Tracking\n2: Training\n3: Split video by frames\nChoose mode and write a number:')
    mode = input()

    if mode == '1':
        print('1: Tracking from device\n2: Tracking from directory\n3: Auto label\nChoose input option:')
        input_option = input()

        if input_option == '1':
            list_devices()
            device_id = input('Choose your device: ')
            track_video(main_model, device_id)

        elif input_option == '2':
            video_dir = input('Enter video directory path: ')
            track_video_dir(main_model, video_dir)

        elif input_option == '3':
            images_dir = './datasets/one/images'
            save_dir = './datasets/one/labels'
            autolabel(main_model, images_dir, save_dir)

    elif mode == '2':
        train_model()

    elif mode == '3':
        split_video()
'''