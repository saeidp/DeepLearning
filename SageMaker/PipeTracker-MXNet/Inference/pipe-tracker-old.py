import mxnet as mx
from mxnet import image
from gluoncv.data.transforms.presets.segmentation import test_transform
import gluoncv
from matplotlib import pyplot as plt
import cv2
import numpy as np


def collapse_contours(contours):
    contours = np.vstack(contours)
    contours = contours.squeeze()
    return contours


def get_rough_contour(image):
    """Return rough contour of character.
    image: Grayscale image.
    """
    img = image.copy()
    # Linux
    # contours, h = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    im2, contours, h = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = collapse_contours(contours)
    return contour

def draw_contour(image, contour, colors=(255, 255, 255)):
    for p in contour:
        cv2.circle(image, tuple(p), 1, colors, 2)


# use cpu
ctx = mx.cpu()
fileName = "003_Centre_PFB-Zone15_2017-08-29_20-25-26_0"
# fileName = "083_Centre_PFA-Zone10_2017-08-28_21-53-06_0"
frame_index = 0
model = gluoncv.model_zoo.DeepLabV3(nclass=2, backbone='resnet50',ctx=ctx)
#model = gluoncv.model_zoo.FCN(nclass=2, backbone='resnet50',ctx=ctx)
# model = gluoncv.model_zoo.DeepLabV3(nclass=2, backbone='resnet101',ctx=ctx)
model.load_parameters('model/model_algo-1',ctx=ctx)

cap = cv2.VideoCapture(fileName + ".mp4")
while(cap.isOpened()):
    success, frm = cap.read()
    frame_index +=1
    if (frame_index % 40 != 0):
        continue
    # cropped based on 240
    frame = cv2.resize(frm,(224,224))
    img = mx.nd.array(frame)
    img = test_transform(img, ctx=ctx)
    img = img.astype('float32')
    output = model.predict(img)
    
    # mx.nd argmax
    # argx = mx.nd.argmax(output, 1)

    # numpy argmax
    # argx = np.argmax(output, axis=1)

    # topk 
    argx = mx.nd.topk(output,1)

    predict = mx.nd.squeeze(argx).asnumpy()
    predict = predict.astype(np.uint8)
    
    dst = cv2.fastNlMeansDenoising(predict,None,10,7,21)
    ret, mask = cv2.threshold(dst, .5, 255, cv2.THRESH_BINARY)
    if (cv2.countNonZero(mask) == 0):
        continue
 
    # no need to put mask on frame
    #white_background = np.full(frame.shape, 255, dtype=np.uint8)
    #bk = cv2.bitwise_or(white_background, white_background, mask=mask)
    #final = cv2.bitwise_and(frame, bk)
    
    final = frame
    contourMask  = get_rough_contour(mask)
    
    # rectangle
    rect = cv2.minAreaRect(contourMask)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(final,[box],0,(255,255,255),1)
    
    
    # Line
    rows,cols = mask.shape[:2]
    [vx,vy,x,y] = cv2.fitLine(contourMask, cv2.DIST_L2,0,0.01,0.01)
    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    img = cv2.line(final,(cols-1,righty),(0,lefty),(255,255,255),1)

    # Centroid
    M = cv2.moments(contourMask)
    if (M['m00'] != 0):
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.circle(final, (cx,cy), 0,(0,255,255), 5)
    
    cv2.imshow("result", final)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


