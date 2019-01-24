# GreenBit

Here is the source code for our Facial Recongnition program. (Note: you only need to download the lated version to run)

We are now at Version 0.9, which includes:
  - the facial landmarks
  - blink counter
  - eye aspect ratio
  - mouth aspect ratio(Version 0.6)
  - eye tracker (Version 0.7)
  - Determine which way a person is looking (Version 0.7)
  - Write the table to a file (Version 0.7)
  - Eye timer to see how long eyes are closed (Version 0.8)
  - Display if eyes are opened or closed (Version 0.8)
  - Head movement - Nod Direction (Version 0.9)
  
## To turn on the virtual environment (using anaconda):
activate opencv-env
## Command to run the code with webcam:  
python Version_0.9.py --shape-predictor shape_predictor_68_face_landmarks.dat
## Command to run the code without webcam:  
python Version_0.9.py -p shape_predictor_68_face_landmarks.dat -v videofile.mp4
## Required data file can be found here:
https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat
When you update the code, rename it to the next version and update this document.
