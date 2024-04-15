import supervision as sv
from tqdm.notebook import tqdm
import os
HOME = os.getcwd()
print("Корень ",HOME)


try:
    # MAKEDIR
    os.mkdir(HOME + '/videos')
    os.mkdir(HOME + '/images')
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
#except:
#    pass

#for i in range(5):
#    pass
