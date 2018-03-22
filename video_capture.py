import numpy as np
import cv2
import matplotlib  
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt 

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('demo.avi',fourcc, 20.0, (1280,960))
# out.write(frame) # write to video file

def detect_die(img, original, name):
    img = cv2.medianBlur(img,7)

    canny = cv2.Canny(img,100,200,7)
    canny_thick = cv2.dilate(canny, np.ones((2, 2)))

    cv2.imshow("canny_thick", canny_thick)

    ret, contours, hierarchy = cv2.findContours(canny_thick,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour, True)
        x,y,w,h = cv2.boundingRect(contour)
        perimeter = cv2.arcLength(contour,True)

        box_ratio = w / h
        
        if area > 2000 and area < 20000 and box_ratio > .85 and box_ratio < 1.15: 
            single_die = gray[y+1:y+h, x+1:x+w]

            otus_ret, single_thresh = cv2.threshold(single_die,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            single_thresh = cv2.bitwise_not(single_thresh)

            cv2.imshow("single_die_thresh", single_thresh)
            cv2.imshow("single_die", single_die)

            params = cv2.SimpleBlobDetector_Params()
            params.minThreshold = 0
            params.maxThreshold = 256
            params.filterByArea = True
            params.minArea = 200
            params.maxArea = 2000
            params.filterByCircularity = False
            params.minCircularity = .8
            params.filterByConvexity = False
            params.minConvexity = 0.5
            params.filterByInertia =False
            params.minInertiaRatio = 0.5
            detector = cv2.SimpleBlobDetector_create(params)

            keypoints = detector.detect(single_thresh)

            single_die_keypoints = cv2.drawKeypoints(single_thresh, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            cv2.imshow("keypoints", single_die_keypoints)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)                
            cv2.imshow("imageaa", img)
            cv2.putText(original, 'die=' + str(len(keypoints)) ,(x,y), font, .8,(0,0,0),2,cv2.LINE_AA)
            print(str(len(keypoints)))

    print("===")

while(cap.isOpened()):
    # ret, frame = cap.read()
    frame = cv2.imread("dice/im1.jpg")
    ret = True

    if ret==True:
        rgb = frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        frame_b, frame_g, frame_r = cv2.split(rgb)
        frame_h, frame_s, frame_v = cv2.split(hsv)
        lab_l, lab_a, lab_b = cv2.split(lab)

        ret2,th2 = cv2.threshold(lab_b,100,255,cv2.THRESH_BINARY)

        detect_die(th2, rgb, "normal")

        cv2.imshow('gray', rgb)
        out.write(rgb)
        print(rgb.shape)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()


