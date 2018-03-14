import numpy as np
import cv2
import matplotlib  
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt 

cap = cv2.VideoCapture(1)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1280,960))
# out.write(frame) # write to video file

def detect_die(img, original, name):
    canny = cv2.Canny(img,100,200,7)
    canny = cv2.dilate(canny, np.ones((2, 2)))

    medianFilt = cv2.medianBlur(rgb,3)

    im2, contours, hierarchy = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour, True)
        x,y,w,h = cv2.boundingRect(contour)

        rectangle_ratio = (w) / (h)
        
        if area > 3000 and rectangle_ratio > .85 and rectangle_ratio < 1.15: 
            perimeter = cv2.arcLength(contour,True)
            print("RATIO")
            print(rectangle_ratio)

            dice_crop = gray[y+1:y+h, x+1:x+w]
            # ret3,th3 = cv2.threshold(dice_crop,50,255,cv2.THRESH_BINARY)
            ret3,th3 = cv2.threshold(dice_crop,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            th3 = cv2.bitwise_not(th3)
            # dice_crop = cv2.Canny(th3,50,100,7)
            cv2.imshow("crop_thresh"+name, th3)


            params = cv2.SimpleBlobDetector_Params()
            # Change thresholds
            params.minThreshold = 0
            params.maxThreshold = 256
            # Filter by Area.
            params.filterByArea = True
            params.minArea = 0
            params.maxArea = 1000
            # Filter by Circularity
            params.filterByCircularity = False
            params.minCircularity = .8
            # Filter by Convexity
            params.filterByConvexity = False
            params.minConvexity = 0.5
            # Filter by Inertia
            params.filterByInertia =False
            params.minInertiaRatio = 0.5
            
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(th3)

            for keypoint in keypoints:
                print(keypoint.size)

            im_with_keypoints = cv2.drawKeypoints(dice_crop, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            # Show keypoints
            cv2.imshow("crop_"+name, im_with_keypoints)

            cv2.rectangle(original,(x,y),(x+w,y+h),(0,255,0),2)                

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(original, 'die=' + str(len(keypoints)) ,(x,y), font, .8,(255,255,255),2,cv2.LINE_AA)

    print("===")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        rgb = frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret2,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        detect_die(th2, gray, "normal")

        # segmentation
        # lower_blue = np.array([110,50,50])
        # upper_blue = np.array([130,255,255])
        # mask = cv2.inRange(frame, lower_blue, upper_blue)
        # res = cv2.bitwise_and(frame,frame, mask= mask)

        # ret,th1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        # th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
        # th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        cv2.imshow('gray', gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()


