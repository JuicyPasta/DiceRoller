import numpy as np
import cv2
import matplotlib  
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt 



def applyOtus(img):
    ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    # Otsu's thresholding
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # plot all the images and their histograms
    images = [img, 0, th1,
            img, 0, th2,
            blur, 0, th3]
    titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
            'Original Noisy Image','Histogram',"Otsu's Thresholding",
            'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

    for i in range(3):
        plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
        plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    plt.show()



cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1280,960))
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # write to video
        # out.write(edges)
        
        # convert color space
        rgb = frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret2,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        canny = cv2.Canny(th2,100,200,7)
        canny = cv2.dilate(canny, np.ones((2, 2)))
        # applyOtus(hjsv[:,:,0])

        medianFilt = cv2.medianBlur(rgb,3)

        im2, contours, hierarchy = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 3000: 
                print(area)
                x,y,w,h = cv2.boundingRect(contour)
                perimeter = cv2.arcLength(contour,True)

                dice_crop = gray[y+1:y+h, x+1:x+w]
                ret3,th3 = cv2.threshold(dice_crop,50,255,cv2.THRESH_BINARY)
                # dice_crop = cv2.Canny(th3,50,100,7)

                params = cv2.SimpleBlobDetector_Params()
                # Change thresholds
                params.minThreshold = 0
                params.maxThreshold = 256
                # Filter by Area.
                params.filterByArea = False
                params.minArea = 30
                # Filter by Circularity
                params.filterByCircularity = False
                params.minCircularity = 0.1
                # Filter by Convexity
                params.filterByConvexity = False
                params.minConvexity = 0.5
                # Filter by Inertia
                params.filterByInertia =False
                params.minInertiaRatio = 0.5
                
                detector = cv2.SimpleBlobDetector_create(params)
                keypoints = detector.detect(th3)
                im_with_keypoints = cv2.drawKeypoints(dice_crop, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                
                # Show keypoints
                cv2.imshow("Keypoints", im_with_keypoints)
                cv2.imshow('crop', th3)

                cv2.rectangle(gray,(x,y),(x+w,y+h),(0,255,0),2)                
                # cv2.drawContours(gray, contour, -1, (0,255,0), 3)

        print("===")

        # segmentation
        # lower_blue = np.array([110,50,50])
        # upper_blue = np.array([130,255,255])
        # mask = cv2.inRange(frame, lower_blue, upper_blue)
        # res = cv2.bitwise_and(frame,frame, mask= mask)

        # ret,th1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        # th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
        # th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

        cv2.imshow('otus', canny)
        cv2.imshow('cont', gray)
 
        # cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()


