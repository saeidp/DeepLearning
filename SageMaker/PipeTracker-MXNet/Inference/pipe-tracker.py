import mxnet as mx
from mxnet import image
from gluoncv.data.transforms.presets.segmentation import test_transform
import gluoncv
from matplotlib import pyplot as plt
import cv2
import numpy as np
import time
import csv
import copy


class PipeTracker:
    def __init__(self):
        self.ctx = mx.gpu(0)
        self.model = gluoncv.model_zoo.DeepLabV3(nclass=2, backbone='resnet50',ctx=self.ctx)
        #self.model = gluoncv.model_zoo.FCN(nclass=2, backbone='resnet50',ctx=self.ctx)
        self.model.load_parameters('model/model_algo-1',ctx=self.ctx)
       
        self.kalman = cv2.KalmanFilter(4, 2) 
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0]], np.float32)

        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32)

        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32) * 0.03
        self.etha = 0.325
        self.previous = None

    def collapse_contours(self, contours):
        contours = np.vstack(contours)
        contours = contours.squeeze()
        return contours
    
    def get_rough_contour(self, image):
        """Return rough contour of character.
        image: Grayscale image.
        """
        img = image.copy()
        # Linux
        # contours, h = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Windows
        im2, contours, h = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = self.collapse_contours(contours)
        return contour
    
    def draw_contour(self, image, contour, colors=(255, 255, 255)):
        for p in contour:
            cv2.circle(image, tuple(p), 1, colors, 2)
    
    def process_frame(self, frame, frame_count, frameWidth, frameHeight):
        # cropped based on 240
        frm = cv2.resize(frame,(224,224))
        
        img = mx.nd.array(frm, ctx=self.ctx)
        img = test_transform(img, ctx=self.ctx)
        # img = img.astype('float32')

        output = self.model.predict(img)

        #argx = mx.nd.argmax(output, 1)
        argx = mx.nd.topk(output,1)
        #argx = np.argmax(output, axis=1)

        # Wait to read actually waits for calculation
        # argx.wait_to_read()
        # or
        # np.squeeze(argx).wait_to_read()

        # tic = time.perf_counter()
        # tic = time.time()
        predict = np.squeeze(argx).asnumpy()
        # toc = time.time()
        # print("Proccesed in {}".format(toc-tic))   

        # predict = predict.astype(np.uint8)
        # current = cv2.fastNlMeansDenoising(predict,None,10,7,21)
        if(frame_count > 1):
            predict = self.etha * predict + (1-self.etha) * self.previous
            # current = current.astype(np.uint8)
        self.previous = copy.copy(predict)
        ret, mask = cv2.threshold(predict.astype(float), .5, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(mask) > 0:
            mask = cv2.resize(mask, (frameWidth, frameHeight))
            # contourMask  = self.get_rough_contour(mask)
            im2, contours, h = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if (len(contours) > 0):
                contourMask = max(contours, key=cv2.contourArea)
                # Centroid
                M = cv2.moments(contourMask)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    self.kalman.correct(np.array([[np.float32(cx)], [np.float32(cy)]] , np.float32))
                    cent = self.kalman.predict()
                    if frame_count >= 25:
                        cx=np.int(cent[0][0])
                        cy=np.int(cent[1][0])

                    # Centroid
                    cv2.circle(frame, (cx,cy), 0,(0,255,255), 5)
            
                    # rectangle
                    rect = cv2.minAreaRect(contourMask)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(frame,[box],0,(255,255,255),1)
            
                    # Line
                    # rows,cols = mask.shape[:2]
                    # [vx,vy,x,y] = cv2.fitLine(contourMask, cv2.DIST_L2,0,0.01,0.01)
                    # lefty = int((-x*vy/vx) + y)
                    # righty = int(((cols-x)*vy/vx)+y)
                    # img = cv2.line(frame,(cols-1,righty),(0,lefty),(255,255,255),1)
                    return frame, cx, cy

        return frame, 0, 0


pipeTracker = PipeTracker()

frame_index = 0
# fileName = "videos/bak/027_Centre_2TL-Zone 3_2017-09-03_02-55-06_4.mp4"
fileName = "videos/BlueShaded/006_Centre_2TL-Zone 2_2017-09-01_12-27-35_5.mp4"
# fileName = "videos/CPProbeCurvedPipe/003_Centre_PFB-Zone15_2017-08-29_20-25-26_0.mp4"

# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(fileName)
if(cap.isOpened() == False):
    print("Error opening video stream or file")
    exit(0)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Frame count:', frame_count)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
print("Frame Width x Height:  ({},{})".format(frame_width, frame_height))

# out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 24, (frame_width, frame_height))

out = cv2.VideoWriter('outpy.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 24, (frame_width, frame_height))

csvFile = open('coordinates.csv', 'w', newline='')
csvWriter = csv.writer(csvFile)
csvWriter.writerow(["Frame #","X", "Y"])
tic = time.time()
# Read until video is completed
while(cap.isOpened()):
    success, frm = cap.read()
    if(success == True):
        frame_index +=1
        if (frame_index % 1 != 0):
            continue
        processed_frame, x, y = pipeTracker.process_frame(frm, frame_index, frame_width, frame_height)
        csvWriter.writerow([frame_index,x,y])
        out.write(processed_frame)
        cv2.imshow("result", processed_frame)
        if(frame_index % 100 == 0):
            print("Number of frame written: {} ".format(frame_index))

        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
print("Number of frame written: {} ".format(frame_index))
toc = time.time()
print("Proccesed in {} seconds".format(toc-tic))  

# When everything done, release the csv file, video capture and writter objects
csvFile.close()
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()
