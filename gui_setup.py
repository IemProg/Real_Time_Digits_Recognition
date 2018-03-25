import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np

#Loading our classifier
classifier = joblib.load("digits_classifier.pkl")

#Testing it
cap = cv2.VideoCapture(0)
while(cap.isOpened()):  # check !
    # capture frame-by-frame
    ret, frame = cap.read()

    if ret: # check ! (some webcam's need a "warmup")
        # our operation on frame come here
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

        #Creating a threshold
        ret, img_th = cv2.threshold(img_gray, 90, 255, cv2.THRESH_BINARY_INV)

        #Find contours in the image
        ctrs, hier, _ = cv2.findContours(img_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #Get rectangles contains each contour
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]

        # For each rectangular region, calculate HOG features and predict
        # the digit using Linear SVM.
        for rect in rects:
            #Draw the rectangles
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
            # Make the rectangular region around the digit
            leng = int(rect[3] * 1.6)
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
            roi = img_th[pt1:pt1+leng, pt2:pt2+leng]

            # Resize the image
            def split_up_resize(arr, res):
                """
                function which resizes large array (direct resize yields error (addedtypo))
                """

                # compute destination resolution for subarrays
                res_1 = (res[0], res[1]//2)
                res_2 = (res[0], res[1] - res[1]//2)

                # get sub-arrays
                arr_1 = arr[0:len(arr)//2]
                arr_2 = arr[len(arr)//2:]

                # resize sub arrays
                arr_1 = cv2.resize(arr_1, res_1, interpolation = cv2.INTER_LINEAR)
                arr_2 = cv2.resize(arr_2, res_2, interpolation = cv2.INTER_LINEAR)

                # init resized array
                arr = np.zeros((res[1], res[0]))

                # merge resized sub arrays
                arr[0 : len(arr)//2] = arr_1
                arr[len(arr)//2:] = arr_2

                return arr
            split_up_resize(roi, [28, 28])
            #roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            """ a pixel element is ‘1’ if atleast one pixel under the kernel is ‘1’.
            So it increases the white region in the image or size of foreground object increases"""
            roi = cv2.dilate(roi, (3, 3))

            # Calculate the HOG features
            roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
            nbr = classifier.predict(np.array([roi_hog_fd], 'float64'))
            cv2.putText(frame, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#cv2.imshow("Predicted digits within the image", frame)
#cv2.waitKey()