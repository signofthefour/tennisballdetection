import numpy as np 
import cv2
import time

# Open def
cap = cv2.VideoCapture(2)

def detectOrangeObject(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    #hsv color of table tennis ball (36, 174, 253)
    lower_orange = np.array([0, 100, 100])
    upper_orange = np.array([50, 255, 255])

    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    res = cv2.bitwise_and(image, image, mask=mask)
    img = cv2.medianBlur(res,5)
    img = cv2.GaussianBlur(res, (13, 13), 2, 2)
    # img = cv2.GaussianBlur(frame, (9, 9), 2, 2)
    cimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cimg = cv2.cvtColor(img, )

    circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=10,maxRadius=200)
    # print(circles)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(cimg,(i[0],i[1]),i[2],(255,0,0),2)
            # draw the center of the circle
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow('detected circles',cimg)
    return img


while(True):
    start_time = time.clock()
    ret, frame = cap.read()
    # r = frame.copy()
    # r[:, :, 0] = 0
    # r[:, :, 1] = 0
    # cv2.imshow('red channel', r)
    cv2.imshow("cam", frame)
    res = detectOrangeObject(frame)
    close_time = time.clock()
    print(close_time - start_time)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.realease()
cv2.destroyAllWindows