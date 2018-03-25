# Real_Time_Digits_Recognition
Real_Time_Digits_Recognition used to recognize zip codes (postal codes) using OpenCv and Sklearn, trained using SVM 
(Support Vector Machines) on MNIST dataset.

# How It Works:
<i>
<b>I -Training the classifier(Preprocessing):</b>

  1 -Calculating the HOG features for each sample in the database.
  
  2 -Training a multi-class linear SVM with HOG feature of each sample along with its label.
  
  3 -Saving the classfier in a file (digit_classifer.pkl), we don't want to do training each time.
  
<b>II -Prediction:</b>

  1- "Predict.py": uploading image used to test the classifier "digit1.jpg".
  
  2- "real_time_setup.py": using opencv2 to open camera, and predict digits from frames captured by camera.
<i/>
<img src="https://github.com/IemProg/Real_Time_Digits_Recognition/blob/master/Predicted_Digits.png">

# Contribution:
Everyong is welcome to contribute.
