import cv2
cap = cv2.VideoCapture('udp://127.0.0.1:5000')
if not cap.isOpened():
    print('VideoCapture not opened')
    exit(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print('frame empty')
        break

    cv2.imshow('image', frame)

    if cv2.waitKey(1)&0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()