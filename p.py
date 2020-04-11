import numpy as np
import cv2, os
# import imutils
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# cap = cv2.VideoCapture('./static/vid1.mov')
# cap = cv2.VideoCapture('rtsp://admin:admin@123@compaid.dyndns.org:1045/unicast/s1/live')
# i=0
# while True:
#     ret, frames = cap.read()
#     cv2.imwrite('demo'+str(i)+'.jpg',frames)
#     image = cv2.imread('demo'+str(i)+'.jpg')

#     # Resize the image - change width to 500
#     image = imutils.resize(image, width=500)

#     # RGB to Gray scale conversion
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Noise removal with iterative bilateral filter(removes noise while preserving edges)
#     gray = cv2.bilateralFilter(gray, 11, 17, 17)

#     # Find Edges of the grayscale image
#     # edged = cv2.Canny(gray, 170, 200)
#     edged = imutils.auto_canny(gray)

#     # Find contours based on Edges
#     cnts, new  = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#     # Create copy of original image to draw all contours
#     img1 = image.copy()
#     cv2.drawContours(img1, cnts, -1, (0,255,0), 3)
#     # cv2.imshow("4- All Contours", img1)
#     # cv2.waitKey(0)

#     #sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
#     cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
#     NumberPlateCnt = None #we currently have no Number plate contour

#     # Top 30 Contours
#     img2 = image.copy()
#     cv2.drawContours(img2, cnts, -1, (0,255,0), 3)
#     # cv2.imshow("5- Top 30 Contours", img2)
#     # cv2.waitKey(0)

#     # loop over our contours to find the best possible approximate contour of number plate

#     for c in cnts:
#         peri = cv2.arcLength(c, True)
#         approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#         # print ("approx = ",approx)
#         if len(approx) == 4:  # Select the contour with 4 corners
#             NumberPlateCnt = approx #This is our approx Number Plate Contour

#             # Crop those contours and store it in Cropped Images folder
#             x, y, w, h = cv2.boundingRect(c) #This will find out co-ord for plate
#             new_img = gray[y:y + h, x:x + w] #Create new image
#             cv2.imwrite( str(i) + '.png', new_img) #Store new image

#             break


#     # Drawing the selected contour on the original image
#     # print(NumberPlateCnt)
#     if NumberPlateCnt is not None:
#         cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)

#         Cropped_img_loc = str(i)+'.png'
#         cv2.imshow("Cropped Image ", cv2.imread(Cropped_img_loc))
        
#         # Use tesseract to covert image into string
#         text = pytesseract.image_to_string(Cropped_img_loc, lang='eng')
#         if text:          
#             print("Number is :", text)
#         else:
#             os.remove(str(i)+'.png')
#     os.remove('demo'+str(i)+'.jpg')
#     i += 1

# cap.release()
# cv2.destroyAllWindows()

# import the necessary packages
from threading import Thread
import sys
import cv2

from queue import Queue

class FileVideoStream:
	def __init__(self, path, queueSize=128):
		# initialize the file video stream along with the boolean
		# used to indicate if the thread should be stopped or not
		self.stream = cv2.VideoCapture(path)
		self.stopped = False
		# initialize the queue used to store frames read from
		# the video file
		self.Q = Queue(maxsize=queueSize)

	def update(self):
		# keep looping infinitely
		while True:
			# if the thread indicator variable is set, stop the
			# thread
			if self.stopped:
				return
			# otherwise, ensure the queue has room in it
			if not self.Q.full():
				# read the next frame from the file
				(grabbed, frame) = self.stream.read()
				# if the `grabbed` boolean is `False`, then we have
				# reached the end of the video file
				if not grabbed:
					self.stop()
					return
				# add the frame to the queue
				self.Q.put(frame)

	def start(self):
		# start a thread to read frames from the file video stream
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self

	def read(self):
		# return next frame in the queue
		return self.Q.get()

	def more(self):
		# return True if there are still frames in the queue
		return self.Q.qsize() > 0

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True


# import the necessary packages

from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2


# start the file video stream thread and allow the buffer to
# start to fill
print("[INFO] starting video file thread...")
fvs = FileVideoStream('./static/2020.mp4').start()
time.sleep(1.0)
# start the FPS timer
fps = FPS().start()

# loop over frames from the video file stream
i = 0
while fvs.more():
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale (while still retaining 3
	# channels)
	# frame = fvs.read()
	# frame = imutils.resize(frame, width=450)
	# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# frame = np.dstack([frame, frame, frame])
	# display the size of the queue on the frame
	# cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
	# 	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
	# show the frame and update the FPS counter
	# cv2.imshow("Frame", frame)
	frames = fvs.read()
	cv2.imwrite('demo'+str(i)+'.jpg',frames)
	image = cv2.imread('demo'+str(i)+'.jpg')

	# Resize the image - change width to 500
	image = imutils.resize(image, width=500)

	# RGB to Gray scale conversion
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Noise removal with iterative bilateral filter(removes noise while preserving edges)
	gray = cv2.bilateralFilter(gray, 11, 17, 17)

	# Find Edges of the grayscale image
	# edged = cv2.Canny(gray, 170, 200)
	edged = imutils.auto_canny(gray)

	# Find contours based on Edges
	cnts, new  = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	# Create copy of original image to draw all contours
	img1 = image.copy()
	cv2.drawContours(img1, cnts, -1, (0,255,0), 3)
	# cv2.imshow("4- All Contours", img1)
	# cv2.waitKey(0)

	#sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
	cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
	NumberPlateCnt = None #we currently have no Number plate contour

	# Top 30 Contours
	img2 = image.copy()
	cv2.drawContours(img2, cnts, -1, (0,255,0), 3)
	# cv2.imshow("5- Top 30 Contours", img2)
	# cv2.waitKey(0)

	# loop over our contours to find the best possible approximate contour of number plate

	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		# print ("approx = ",approx)
		if len(approx) == 4:  # Select the contour with 4 corners
			NumberPlateCnt = approx #This is our approx Number Plate Contour

			# Crop those contours and store it in Cropped Images folder
			x, y, w, h = cv2.boundingRect(c) #This will find out co-ord for plate
			new_img = gray[y:y + h, x:x + w] #Create new image
			cv2.imwrite( str(i) + '.png', new_img) #Store new image

			break

	# Drawing the selected contour on the original image
	# print(NumberPlateCnt)
	if NumberPlateCnt is not None:
		cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)

		Cropped_img_loc = str(i)+'.png'
		# cv2.imshow("Cropped Image ", cv2.imread(Cropped_img_loc))

		# Use tesseract to covert image into string
		text = pytesseract.image_to_string(Cropped_img_loc, lang='eng')
		if text:
			print("Number is :", text)
		else:
			os.remove(str(i)+'.png')
	os.remove('demo'+str(i)+'.jpg')
	i += 1

	cv2.waitKey(1)
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()