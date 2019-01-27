# import the necessary packages
import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


class VideoCamera(object):        
    
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
    
        success, image = self.video.read()
        
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	    # detect faces in the grayscale frame
        rects = detector(gray, 0)
        for rect in rects:
    
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
        
            #################################################################
            ############### Head nodding Code ###############################
            #################################################################
            nosePointX = shape[30]
            nodDirection = " "
            noseState = (int) (800 / 2)
            xArc = (int) (800 / 10)
            noseR = (int) (xArc * 4)
            noseL = (int) (xArc * 6)

            if(nosePointX[0] <= noseR and nosePointX[0] < noseState):
                noseState = nosePointX[0]
                nodDirection = "Nodding Left to Right"
                
            if(nosePointX[0] > noseL and nosePointX[0] > noseState):
                noseState = nosePointX[0]
                nodDirection = "Nodding Right to Left"

            # draw the direction of head nod
            cv2.putText(image, "Head nod: {}".format(nodDirection), (10, 30),
            cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 204, 0), 2)
            #draw the nose state - for testing
            cv2.putText(image, "Nose state: {}".format(noseState), (10, 60),
            cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 204, 0), 2)

            #################################################################
            ############### Checking if eyes are closed/open ################
            #################################################################
            # grab the indexes of the facial landmarks for the left and
            # right eye, mouth respectively
            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            ##### Get the left ear ####
            A = dist.euclidean(leftEye[1], leftEye[5])
            B = dist.euclidean(leftEye[2], leftEye[4])
            # compute the euclidean distance between the horizontal
            # eye landmark (x, y)-coordinates
            C = dist.euclidean(leftEye[0], leftEye[3])
            # compute the eye aspect ratio
            leftEAR = (A + B) / (2.0 * C)
                
            ##### Get the Right ear ####
            A = dist.euclidean(rightEye[1], rightEye[5])
            B = dist.euclidean(rightEye[2], rightEye[4])
            # compute the euclidean distance between the horizontal
            # eye landmark (x, y)-coordinates
            C = dist.euclidean(rightEye[0], rightEye[3])
            # compute the eye aspect ratio
            rightEAR = (A + B) / (2.0 * C)

	        # check to see if the eye aspect ratio is below the blink
		    # threshold, and if so, increment the blink frame counter
            if (leftEAR + rightEAR) / 2.0 < 0.2:
                # draw the computed eye aspect ratio for the frame
                cv2.putText(image, "Eyes are closed", (10, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 204, 0), 2)
            else:
                # draw the computed eye aspect ratio for the frame
                cv2.putText(image, "Eyes are Open", (10, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 204, 0), 2)
               
            #################################################################
            ########### End of checking if eyes are closed/open #############
            #################################################################
           
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        ret, jpeg = cv2.imencode('.jpeg', image)
        return jpeg.tobytes()
        
   