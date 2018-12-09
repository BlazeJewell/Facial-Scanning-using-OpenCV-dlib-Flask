# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from prettytable import PrettyTable
import numpy as np
import argparse
import imutils
import time
from timeit import default_timer as timer
import dlib
import cv2

table = PrettyTable()
table.field_names = ['Time', 'Eye Aspect Ratio', 'Right Eye Aspect Ratio', 'Left Eye Aspect Ratio', 'Blinks', 'Right Eye Blinks', 'Left Eye Blinks', 'Mouth Aspect Ratio', 'Mouth Opening Count', "Looking Direction", "Eyes Closed Timer", "Open or closed"]


past_values_x = []
def min_intensity_x(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	min_sum_y = 255 * len(img)
	min_index_x = -1
	
	for x in range(len(img[0])):
		
		temp_sum_y = 0
		
		for y in range(len(img)):
			temp_sum_y += img[y][x]
		
		if temp_sum_y < min_sum_y:
			min_sum_y = temp_sum_y
			min_index_x = x
	
	past_values_x.append(min_index_x)
	
	if len(past_values_x) > 3:
		past_values_x.pop(0)
	
	return int(sum(past_values_x) / len(past_values_x))

past_values_y = []
def min_intensity_y(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	min_sum_x = 255 * len(img[0])
	min_index_y = -1
	
	for y in range(len(img)):
		
		temp_sum_x = 0
		
		for x in range(len(img[0])):
			temp_sum_x += img[y][x]
		
		if temp_sum_x < min_sum_x:
			min_sum_x = temp_sum_x
			min_index_y = y
	
	past_values_y.append(min_index_y)
	
	if len(past_values_y) > 3:
		past_values_y.pop(0)
	
	return int(sum(past_values_y) / len(past_values_y))

def extract_eye(image, left, bottom_left, bottom_right, right, upper_right, upper_left):
	lower_bound = max([left[1], right[1], bottom_left[1], bottom_right[1], upper_left[1], upper_right[1]])
	upper_bound = min([left[1], right[1], upper_left[1], upper_right[1], bottom_left[1], bottom_right[1]])

	eye = image[upper_bound-3:lower_bound+3, left[0]-3:right[0]+3]
	
	pupil_x = min_intensity_x(eye)
	pupil_y = min_intensity_y(eye)
	
	cv2.line(eye,(pupil_x,0),(pupil_x,len(eye)),(255,0,0), 1)
	cv2.line(eye,(0,pupil_y),(len(eye[0]),pupil_y),(0,255,0), 1)
	
	cv2.line(image,(int((bottom_left[0] + bottom_right[0]) / 2), lower_bound), (int((upper_left[0] + upper_right[0]) / 2), upper_bound),(0,0,255), 1)
	cv2.line(image,(left[0], left[1]), (right[0], right[1]),(0,0,255), 1)
	
	image[upper_bound-3:lower_bound+3, left[0]-3:right[0]+3] = eye
	return eye

def getDirection(image, left, bottom_left, bottom_right, right, upper_right, upper_left):
	lower_bound = max([left[1], right[1], bottom_left[1], bottom_right[1], upper_left[1], upper_right[1]])
	upper_bound = min([left[1], right[1], upper_left[1], upper_right[1], bottom_left[1], bottom_right[1]])

	eye = image[upper_bound-3:lower_bound+3, left[0]-3:right[0]+3]
	
	pupil_x = min_intensity_x(eye)
	pupil_y = min_intensity_y(eye)
	
	#Here is wen we check if user is looking left or right
	if pupil_x < 25:
		lookingDirection = "right"

	elif pupil_x > 35:
		lookingDirection = "left"

	else:
		lookingDirection = "center"

	return lookingDirection

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
 
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
 
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
 
	# return the eye aspect ratio
	return ear
# 60-68
# 48-68
def mouth_aspect_ratio(mouth):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(mouth[1], mouth[7])
	B = dist.euclidean(mouth[2], mouth[6])

	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
	C = dist.euclidean(mouth[3], mouth[5])
	D = dist.euclidean(mouth[0], mouth[4])
 
	# compute the eye aspect ratio
	mar = (A + B + C) / (3.0 * D)
 
	# return the mouth aspect ratio
	return mar

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 2
MOUTH_AR_THRESH = 0.09
# initialize the frame counters and the total number of blinks
COUNTER = 0
LEFTCOUNTER = 0
LEFTTOTAL = 0
RIGHTCOUNTER = 0
RIGHTTOTAL = 0
TOTAL = 0
MOUTHCOUNTER = 0
MOUTHTOTAL = 0

# initialize the frame counter for the eyes closed
START = 0
END = 0
SECONDS = END-START
EYES= ""

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, mouth respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

	
# start the video stream thread
print("[INFO] starting video stream thread...")
vs = FileVideoStream(args["video"]).start()
#fileStream = True
vs = VideoStream(src=0).start()
#vs = VideoStream(usePiCamera=True).start()
fileStream = False

time.sleep(1.0)

# loop over frames from the video stream
while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if fileStream and not vs.more():
		break
 
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=800)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
	# detect faces in the grayscale frame
	rects = detector(gray, 0)


	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
 
		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		mouthFind = shape[mStart+12:mEnd]
		mouthMAR = mouth_aspect_ratio(mouthFind)
 
 		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(frame, (x, y), 1, (0, 200, 0), -1)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		mouthHull = cv2.convexHull(mouthFind)
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

			# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if rightEAR < EYE_AR_THRESH and leftEAR < EYE_AR_THRESH:
			COUNTER += 1        		
		# otherwise, the eye aspect ratio is not below the blink
		# threshold
		else:
			# if the eyes were closed for a sufficient number of
			# then increment the total number of blinks
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1
		
			# reset the eye frame counter
			COUNTER = 0
        # Start timer when eyes are closed and display message
		if rightEAR > EYE_AR_THRESH and leftEAR > EYE_AR_THRESH:		
			EYES = "Open"
			END = timer()
		else:
			START = timer()
			SECONDS = END-START
			EYES = "Closed"
		
		if rightEAR < EYE_AR_THRESH:
			RIGHTCOUNTER += 1
		else:
			if RIGHTCOUNTER >= EYE_AR_CONSEC_FRAMES:
				RIGHTTOTAL += 1
			RIGHTCOUNTER = 0

		if leftEAR < EYE_AR_THRESH:
			LEFTCOUNTER += 1
		else: 
			if LEFTCOUNTER >= EYE_AR_CONSEC_FRAMES:
				LEFTTOTAL += 1
			LEFTCOUNTER = 0

		if mouthMAR < MOUTH_AR_THRESH:
			MOUTHCOUNTER += 1
		else:
			if MOUTHCOUNTER >= EYE_AR_CONSEC_FRAMES:
				MOUTHTOTAL += 1
			MOUTHCOUNTER = 0

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		count = 1
		#Will break program if the eye cannot be found or it disapears.
		right_eye = imutils.resize(extract_eye(frame, shape[36], shape[41], shape[40], shape[39], shape[38], shape[37]), width=100, height=50) 	
		
		lookingDirection = getDirection(frame, shape[36], shape[41], shape[40], shape[39], shape[38], shape[37])
		
		frame[0:len(right_eye),0:len(right_eye[0])] = right_eye
			

		# Draw the date and time on the screen
		time.ctime()
		cv2.putText(frame, time.strftime("%a, %d %b %Y %H:%M:%S"), (10, 60),
		cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 204, 0), 2)

		# draw the computed eye aspect ratio for the frame
		cv2.putText(frame, "left eye ratio: {:.6f}".format(leftEAR), (10, 90),
		cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 204, 0), 2)

		# draw the computed eye aspect ratio for the frame
		cv2.putText(frame, "right eye ratio: {:.6f}".format(rightEAR), (10, 120),
		cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 204, 0), 2)

		# draw the computed eye aspect ratio for the frame
		cv2.putText(frame, "eye aspect ratio: {:.6f}".format(ear), (10, 150),
		cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 204, 0), 2)

		# draw the total number of blinks on the frame
		cv2.putText(frame, "Left Blinks: {}".format(LEFTTOTAL), (10, 180),
		cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 204, 0), 2)

		# draw the total number of blinks on the frame
		cv2.putText(frame, "Right Blinks: {}".format(RIGHTTOTAL), (10, 210),
		cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 204, 0), 2)

		#draw the total number of blinks on the frame
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 240),
		cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 204, 0), 2)

		#draw the total number of blinks on the frame
		cv2.putText(frame, "Mouth Opened: {}".format(MOUTHTOTAL), (10, 270),
		cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 204, 0), 2)

		# draw the computed mouth aspect ratio for the frame
		cv2.putText(frame, "mouth aspect ratio: {:.6f}".format(mouthMAR), (10, 300),
		cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 204, 0), 2)

		# draw the looking direction for the frame
		cv2.putText(frame, "Looking direction: {}".format(lookingDirection), (10, 330),
		cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 204, 0), 2)
	
	 	# display counter for how long the eyes are closed
		cv2.putText(frame, "Closed Eyes: {:.6f} SECONDS".format(SECONDS), (10, 360),
        cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 204, 0), 2)

	    # display counter for how long the eyes are closed
		cv2.putText(frame, "Eyes are: {}".format(EYES), (10, 390),
        cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 204, 0), 2)

        # Add table
		table.add_row([time.strftime("%a, %d %b %Y %H:%M:%S"), "{:.6f}".format(ear), "{:.6f}".format(rightEAR), "{:.6f}".format(leftEAR), "{}".format(TOTAL), "{}".format(RIGHTTOTAL), "{}".format(LEFTTOTAL),   "{:.6f}".format(mouthMAR),  "{}".format(MOUTHTOTAL), lookingDirection, "{:.6f}".format(SECONDS), EYES])

	# show the frame
	cv2.imshow("Facial Detection with OpenCV, Python, and dlib", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

print(table)
with open('table.txt', 'w') as w:
    w.write(str(table))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
