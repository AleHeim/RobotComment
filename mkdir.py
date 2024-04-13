import os
HOME = os.getcwd()
print(HOME)

try:
    # MAKEDIR
    os.mkdir(HOME + '/videos')
    os.mkdir(HOME + '/images')
except: 
    print("\n \n \n")

VIDEO_DIR_PATH = f"{HOME}/videos"
IMAGE_DIR_PATH = f"{HOME}/images"
FRAME_STRIDE = 10

print(VIDEO_DIR_PATH)