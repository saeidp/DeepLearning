import mxnet as mx
from mxnet import image
from gluoncv.data.transforms.presets.segmentation import test_transform
import gluoncv
import cv2
import numpy as np
import time
import copy
import enum
import socket
from threading import Thread

class Server:
    def __init__(self):
        self.host = '127.0.0.1'
        self.port = 8000

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.host, self.port))
        self.server.listen(5)
    
    def listen_for_clients(self):
        Thread(target=self.handle_client).start()

    def handle_client(self):
        print('Listening...')
        while True:
            client_socket, address = self.server.accept()
            print(
                'Accepted Connection from: ' + str(address[0]) + ':' + str(address[1])
            )

            size = 1024
            while True:
                try:
                    data = client_socket.recv(size)
                    if 'q^' in data.decode():    
                        print('Received request for exit from: ' + str(
                            address[0]) + ':' + str(address[1]))
                        break

                    else:
                        # send getting after receiving from client
                        client_socket.sendall('Welcome to server'.encode())
                        print(data.decode())

                except socket.error:
                    client_socket.close()
                    return False

            client_socket.sendall(
                'Received request for exit. Deleted from server threads'.encode()
            )

            # send quit message to client too
            client_socket.sendall(
                'q^'.encode()
            )
            client_socket.close()
   

class Signal(enum.Enum):
    NoPipe = "NOPIPE"
    Left = "LEFT"
    Right = "RIGHT" 
    Center = "CENTER"

class PipeTracker:
    def __init__(self):
        self.etha = 0.325
        self.previous = None
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

    def find_signal(self, cx, leftcx, rightcx ):
        if(cx < leftcx):
            return Signal.Left
        elif(cx > rightcx):
            return Signal.Right
        elif(cx>= leftcx and cx <= rightcx):
            return Signal.Center
        else:
            return Signal.NoPipe


    def process_frame(self, frame, frame_count, frameWidth, frameHeight):
        fcx = frame_width // 2 
        fcy = frame_height // 2
        percent = 0.2
        rightcx = int(fcx + percent * frame_width)
        leftcx = int(fcx - percent * frame_width)
        signal = Signal.NoPipe
        # cropped based on 240
        frm = cv2.resize(frame,(224,224))
        
        img = mx.nd.array(frm, ctx=self.ctx)
        img = test_transform(img, ctx=self.ctx)

        output = self.model.predict(img)

        argx = mx.nd.topk(output,1)
        predict = np.squeeze(argx).asnumpy()
        if(frame_count > 1):
            predict = self.etha * predict + (1-self.etha) * self.previous
        self.previous = copy.copy(predict)
        ret, mask = cv2.threshold(predict.astype(float), .5, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(mask) > 0:
            mask = cv2.resize(mask, (frameWidth, frameHeight))
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
                    
                    signal = self.find_signal(cx, leftcx, rightcx)

                    # Centroid
                    cv2.circle(frame, (cx,cy), 0,(0,255,255), 5)
            
                    # rectangle
                    rect = cv2.minAreaRect(contourMask)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(frame,[box],0,(255,255,255),1)
            
                    return frame, cx, cy, signal.value
        return frame, 0, 0, Signal.NoPipe.value


pipeTracker = PipeTracker()

frame_index = 0
fileName = "videos/PartiallyVisiblePipe/036_Centre_PFA-Zone9_2017-08-28_01-58-17_3.mp4"

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

# Socket server
server = Server()
server.listen_for_clients()

# Socket client
socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket_client.connect((server.host,  server.port))

tic = time.time()
# Read until video is completed
while(cap.isOpened()):
    success, frm = cap.read()
    if(success == True):
        frame_index +=1
        processed_frame, x, y, signal = pipeTracker.process_frame(frm, frame_index, frame_width, frame_height)
        cv2.imshow("result", processed_frame)
        socket_client.send(signal.encode())
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
print("Number of frame processed: {} ".format(frame_index))
toc = time.time()
print("Proccesed in {} seconds".format(toc-tic))  

socket_client.close()
cap.release()
cv2.destroyAllWindows()
