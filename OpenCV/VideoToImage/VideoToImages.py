import cv2
import os
print(cv2.__version__)

Folder = "videos/"
FileName = "027_Centre_2TL-Zone 3_2017-09-03_02-55-06_4"
# start from this number and add to the name of image.
count = 1000

if (os.path.isdir('images/{}'.format(FileName)) == False):
    os.mkdir('images/{}'.format(FileName))

vidcap = cv2.VideoCapture(Folder + FileName + ".mp4")

frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Frame count:', frame_count)
# frame_start = frame_count // 2
frame_start = 0
frame_increment = 100
frame_index = frame_start
vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

success, image = vidcap.read()
while success:
    if (frame_index == frame_start + frame_increment):
        # save frame as JPEG file
        cv2.imwrite("images/{0}/img%d.jpg".format(FileName) % count, image)
        count += 1
        frame_start += frame_increment
    success, image = vidcap.read()
    frame_index += 1
    print('Read a new frame: ', success)
