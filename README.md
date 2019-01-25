# GreenBit
Here is the source code for our Facial Recongnition program.

We are now at Version 0.9 for the facial scanning, which includes:
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

# Instructions to run this code  
### 1.1 -> Follow the instruction
   https://www.learnopencv.com/install-opencv-3-and-dlib-on-windows-python-only/
### 1.2 -> when you get to setp 3.2, after the first 2 commands, install cmake (pip install cmake), then use this command to install dlib: 
  conda install -c conda-forge dlib
### 2.0 -> turn on the virtual environment (using anaconda):
  activate opencv-env
### 3.0 -> Command to run the Web Application:  
  python main.py
## Required data file can be found here:
https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat


