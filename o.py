import numpy as np
import cv2
import imutils
import sys
import pytesseract
import pandas as pd
import time, os

#Add this line to assert the path. Else TesseractNotFoundError will be raised.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

cap = cv2.VideoCapture('./static/video5.mp4')

i=0
while True:
    ret, frames = cap.read()
    cv2.imwrite('demo'+str(i)+'.jpg',frames)
    #Read the original image.
    img = cv2.imread('demo'+str(i)+'.jpg')
    # img = cv2.imread('./static/images/number_plate_4.jpg')

    #Using imutils to resize the image.
    img = imutils.resize(img, width=500)

    #Convert from colored to Grayscale.
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Applying Bilateral Filter on the grayscale image.
    #It will remove noise while preserving the edges. So, the number plate remains distinct.

    gray_img = cv2.bilateralFilter(gray_img, 11, 17, 17)

    #Finding edges of the grayscale image.
    c_edge = cv2.Canny(gray_img, 170, 200)

    #Finding contours based on edges detected.
    cnt, new = cv2.findContours(c_edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
   
    #Storing the top 30 edges based on priority
    cnt = sorted(cnt, key = cv2.contourArea, reverse = True)[:30]
    NumberPlateCount = 0
    im2 = img.copy()
    cv2.drawContours(im2, cnt, -1, (0,255,0), 3)

    for c in cnt:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # print ("approx = ",approx)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCount = approx #This is our approx Number Plate Contour

            # Crop those contours and store it in Cropped Images folder
            x, y, w, h = cv2.boundingRect(c) #This will find out co-ord for plate
            new_img = gray_img[y:y + h, x:x + w] #Create new image
            cv2.imwrite(str(i) + '.png', new_img) #Store new image

            break

    # Drawing the selected contour on the original image
    #print(NumberPlateCnt)
    cv2.drawContours(img, [NumberPlateCount], -1, (0,255,0), 3)
    cv2.imshow("Final Image With Number Plate Detected", img)
    cv2.waitKey(0)

    Cropped_img_loc = str(i)+'.png'
    cv2.imshow("Cropped Image ", cv2.imread(Cropped_img_loc))

    # Use tesseract to covert image into string
    text = pytesseract.image_to_string(Cropped_img_loc, lang='eng')
    print("Number is :", text)
    # count = 0
    # for c in cnt:
    #     perimeter = cv2.arcLength(c, True)      #Getting perimeter of each contour
    #     approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    #     if len(approx) == 4:            #Selecting the contour with 4 corners/sides.
    #         NumberPlateCount = approx
    #         break

    '''A picture can be stored as a numpy array. Thus to mask the unwanted portions of the
    picture, we simply convert it to a zeros array.'''
    #Masking all other parts, other than the number plate.
    # masked = np.zeros(gray_img.shape,np.uint8)
    # new_image = cv2.drawContours(masked,[NumberPlateCount],0,255,-1)
    # new_image = cv2.bitwise_and(img,img,mask=masked)
    
    # # secondCrop = new_image[y:y+h,x:x+w]
    # cv2.imshow("4 - Final_Image",new_image)     #The final image showing only the number plate.
    # cv2.waitKey(0)

    #Configuration for tesseract
    # configr = ('-l eng --oem 1 --psm 3')

    # #Running Tesseract-OCR on final image.
    # text_no = pytesseract.image_to_string(new_image, config=configr)
    # print(text_no)
    # #The extracted data is stored in a data file.
    # data = {'Date': [time.asctime(time.localtime(time.time()))],'Vehicle_number': [text_no]}
    # print(data)
    # df = pd.DataFrame(data, columns = ['Date', 'Vehicle_number'])
    # df.to_csv('Dataset_VehicleNo.csv')

    # #Printing the recognized text as output.
    # print(text_no)
    os.remove('demo'+str(i)+'.jpg')
    # os.remove(str(i)+'.png')
    cv2.waitKey(0)
    i+=1
    
    if cv2.waitKey(33) == 13:
        break
cap.release()
cv2.destroyAllWindows()

# import cv2, time, imutils, pytesseract, numpy as np

# cap = cv2.VideoCapture('./static/video3.mp4')
# # cap = cv2.VideoCapture(0)
# pedestrian_cascade = cv2.CascadeClassifier('./static/cars.xml')
# i=0
# while True:
#     pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#     ret, frames = cap.read()
#     pedestrians = pedestrian_cascade.detectMultiScale( frames, 1.1, 1)
#     for (x,y,w,h) in pedestrians:
#         cv2.waitKey(0)
#         # cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)
#         # font = cv2.FONT_HERSHEY_DUPLEX
#         # cv2.putText(frames, 'Cars', (x + 6, y - 6), font, 0.5, (0, 0, 255), 1)
#         # cv2.imshow('Object detection', frames)
#         # cv2.imwrite('kang'+str(i)+'.jpg',frames)
#         # image = cv2.imread('C:/Users/Vaibhav/Desktop/Assignment/demo/static/images/car.png')
#         # image = cv2.imread(frames)
#         # image = imutils.resize(image, width=500)
#         gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
#         # cv2.imshow("1 - Grayscale Conversion", gray)

#         # Noise removal with iterative bilateral filter(removes noise while preserving edges)
#         gray = cv2.bilateralFilter(gray, 11, 17, 17)
#         # cv2.imshow("2 - Bilateral Filter", gray)

#         # Find Edges of the grayscale image
#         edged = cv2.Canny(gray, 170, 200)
#         # cv2.imshow("4 - Canny Edges", edged)

#         cnt, new = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#         #Storing the top 30 edges based on priority
#         cnt = sorted(cnt, key = cv2.contourArea, reverse = True)[:30]
#         NumberPlateCount = None


#         # im2 = image.copy()
#         cv2.drawContours(edged, cnt, -1, (0,255,0), 3)
#         cv2.imshow("Top 30 Contours", edged)          #Show the top 30 contours.
#         cv2.waitKey(0)


#         count = 0
#         for c in cnt:
#             perimeter = cv2.arcLength(c, True)      #Getting perimeter of each contour
#             approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
#             if len(approx) == 4:            #Selecting the contour with 4 corners/sides.
#                 NumberPlateCount = approx
#                 break
        
#         masked = np.zeros(gray.shape,np.uint8)
#         new_image = cv2.drawContours(masked,[NumberPlateCount],0,255,-1)
#         new_image = cv2.bitwise_and(frames,frames,mask=masked)
#         cv2.imshow("4 - Final_Image",new_image)     #The final image showing only the number plate.
#         cv2.waitKey(0)

#         configr = ('-l eng --oem 1 --psm 3')

#         #Running Tesseract-OCR on final image.
        
#         text_no = pytesseract.image_to_string(new_image, config=configr)

#         #The extracted data is stored in a data file.
#         data = {'Date': [time.asctime(time.localtime(time.time()))],'Vehicle_number': [text_no]}
#         print(data)
#     i+=1
    
#     if cv2.waitKey(33) == 13:
#         break
# cap.release()
# cv2.destroyAllWindows()