from scipy.spatial import distance
from imutils import face_utils
import cv2
import dlib
import imutils
import pygame
import time

def e_a_r(eye):
	x = distance.euclidean(eye[1], eye[5])
	y = distance.euclidean(eye[2], eye[4])
	z = distance.euclidean(eye[0], eye[3])
	ear = (x + y) / (2.0 * z)
	return ear
	
count_threshold = 0.25
count_frame_check = 10
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("/home/ashish/packages/Drowsiness_Detection/shape_predictor_68_face_landmarks.dat")

(lS, lE) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rS, rE) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap=cv2.VideoCapture(0)
Count=0
while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=449)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	subjects = detect(gray, 0)
	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)
		lEye = shape[lS:lE]
		rEye = shape[rS:rE]
		lefEAR = e_a_r(lEye)
		righEAR = e_a_r(rEye)
		ear = (lefEAR + righEAR) / 2.0
		leftEyeHull = cv2.convexHull(lEye)
		rightEyeHull = cv2.convexHull(rEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (255, 128, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (255, 128, 0), 1)
		if ear < count_threshold:
			Count =Count + 1
			print (Count)
			if Count >= count_frame_check:
				cv2.putText(frame, "                ALERT                      ", (11,323),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				pygame.init()

				pygame.mixer.music.load("alert1.mp3")

				pygame.mixer.music.play()
				

		else:
			Count = 0
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("s"):
		break
cv2.destroyAllWindows()
cap.stop()
