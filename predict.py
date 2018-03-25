import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np

#Loading our classifier
classifier = joblib.load("digits_classifier.pkl")

#Testing it
img = cv2.imread("digit1.jpg")

output = ""
if img is not None:
    #Convert image to grayscale and applying filters
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

    #Creating a threshold
    ret, img_th = cv2.threshold(img_gray, 90, 255, cv2.THRESH_BINARY_INV)

    #Find contours in the image
    image, ctrs, hier = cv2.findContours(img_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Get rectangles contains each contour
    #rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    for c in ctrs:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a green rectangle to visualize the bounding rect
        #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # get the min area rect
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="int")

        # convert all coordinates floating point values to int
        box = np.int0(box)
        # draw a red 'nghien' rectangle
        cv2.drawContours(img, [box], 0, (0, 0, 255))

        # finally, get the min enclosing circle
        #(x, y), radius = cv2.minEnclosingCircle(c)
        # convert all values to int
        #center = (int(x), int(y))
        #radius = int(radius)
        # and draw the circle in blue
        #img = cv2.circle(img, center, radius, (255, 0, 0), 2)

        roi = img_th[y:y+h, x:x+w]
        # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        #a pixel element is ‘1’ if atleast one pixel under the kernel is ‘1’.
        #So it increases the white region in the image or size of foreground object increases
        roi = cv2.dilate(roi, (3, 3))

        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        nbr = classifier.predict(np.array([roi_hog_fd], 'float64'))
        output = output +" "+str(nbr[0])
        cv2.putText(img, str(int(nbr[0])), (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 3)
    cv2.imshow("Predicted digits within the image", img)
    print("Output: ", output)             #Contours goes from right to left

#print("Number of contours is: ", len(ctrs))
#cv2.drawContours(img, ctrs, -1, (255, 255, 0), 1)

#cv2.imshow("contours", img)
ESC = 27
while True:
    keycode = cv2.waitKey()
    if keycode == ESC:
        break
cv2.destroyAllWindows()