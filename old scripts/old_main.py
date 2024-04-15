from ultralytics import YOLO
import supervision as sv
import cv2
import os

model = YOLO('./runs/detect/train3/weights/best.pt')

def splitvid():
    try:
        # MAKEDIR
        os.mkdir('./videos')
        os.mkdir('./images')
    except: 
        print("\n \n \n")


    VIDEO_DIR_PATH = f"{HOME}/videos"
    IMAGE_DIR_PATH = f"{HOME}/images"
    FRAME_STRIDE = 1100 #10


    video_paths = sv.list_files_with_extensions(
        directory=VIDEO_DIR_PATH,
        extensions=["mov", "mp4", "mkv"])

    print("Путь к видео необходимого расширения",video_paths,"\n")
    # TEST_VIDEO_PATHS = video_paths[1:]
    # TRAIN_VIDEO_PATHS = video_paths[:1]
    TRAIN_VIDEO_PATHS = video_paths
    # print(TEST_VIDEO_PATHS,"\n\n", TRAIN_VIDEO_PATHS)

    #try:
    for video_path in TRAIN_VIDEO_PATHS:
        video_name = video_path.stem
        image_name_pattern = video_name + "-{:05d}.png"
        with sv.ImageSink(target_dir_path=IMAGE_DIR_PATH, image_name_pattern=image_name_pattern) as sink:
            for image in sv.get_video_frames_generator(source_path=str(video_path), stride=FRAME_STRIDE):
                sink.save_image(image=image)

def tracking(model, input, location):
    if(input=='dir'):
        VIDEO_DIR_PATH = location
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
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
    elif(input=='device'):
        cap = cv2.VideoCapture(location)
        ret = True
        while ret:
            ret, frame=cap.read()
            results=model.track(frame, persist=True)
            frame_ = results[0].plot()
            cv2.imshow('frame', frame_)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

def training():
    model = YOLO('yolov8s.yaml') # Build a model from scratch. In yolo8n.yaml - "n" means nano. (n,s,m,l,x) в порядке возрастания
    results = model.train(data="config.yaml", epochs = 110)

def list_devices():
    devices = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            devices.append(i)
            cap.release()
    print(devices)


if __name__ == '__main__':
    print('1: tracking\n2: training\n3: Split video by frames\nChoose mode and write a number: ')
    cinput = input()
    if(cinput=='1'):
        print('1: Tracking from device\n2: Tracking from video\nChoose input options: ')
        cinput = input()
        if(cinput=='1'):
            list_devices()
            print('Choose your device')
            # add device selector
            location = int(input())
            tracking(model, 'device', location)
        elif(cinput=='2'):
            print('Write video dir: \n Example - ./videos')
            location=input()
            tracking(model, 'dir', location)
    elif(cinput=='2'):
        training()
    elif(cinput=='3'):
        splitvid()





