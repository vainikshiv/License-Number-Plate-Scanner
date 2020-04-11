from django.shortcuts import render, redirect
import concurrent.futures
import cv2, os, imutils, pytesseract, numpy as np
from django.http import JsonResponse
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from django.views.decorators.csrf import csrf_exempt
from .models import data
import json, re
# Create your views here.

# Home page function
def home(request):
    return render(request,'home.html')

@csrf_exempt
def update(request):
    if request.method == "POST":
        cap = cv2.VideoCapture('./static/2020.mp4')
        # cap = cv2.VideoCapture('rtsp://admin:admin@123@compaid.dyndns.org:1045/unicast/s1/live')
        i = 0
        while True:
            ret, frames = cap.read()
            cv2.imwrite('demo' + str(i) + '.jpg', frames)
            image = cv2.imread('demo' + str(i) + '.jpg')

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
            cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # Create copy of original image to draw all contours
            img1 = image.copy()
            cv2.drawContours(img1, cnts, -1, (0, 255, 0), 3)

            # sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
            NumberPlateCnt = None  # we currently have no Number plate contour

            # Top 30 Contours
            img2 = image.copy()
            cv2.drawContours(img2, cnts, -1, (0, 255, 0), 3)

            # loop over our contours to find the best possible approximate contour of number plate

            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                # print ("approx = ",approx)
                if len(approx) == 4:  # Select the contour with 4 corners
                    NumberPlateCnt = approx  # This is our approx Number Plate Contour

                    # Crop those contours and store it in Cropped Images folder
                    x, y, w, h = cv2.boundingRect(c)  # This will find out co-ord for plate
                    new_img = gray[y:y + h, x:x + w]  # Create new image
                    cv2.imwrite('./media/'+str(i) + '.png', new_img)  # Store new image

                    break

            # Drawing the selected contour on the original image
            # print(NumberPlateCnt)
            if NumberPlateCnt is not None:
                cv2.drawContours(image, [NumberPlateCnt], -1, (0, 255, 0), 3)

                Cropped_img_loc = './media/'+str(i) + '.png'

                # Use tesseract to covert image into string
                text = pytesseract.image_to_string(Cropped_img_loc, lang='eng')
                if text:
                    result = re.findall('(\d{2,4})$',text)
                    print(result)
                    if result:
                        d = data.objects.create(license_number=text, image='./media/'+str(i) + '.png')
                        d.save()
                        new = {'number':text,"image":'./media/'+str(i) + '.png'}
                        return JsonResponse({"status": 200, "detect":json.dumps(new)})
                    else:
                        os.remove('./media/' + str(i) + '.png')
                else:
                    os.remove('./media/'+str(i) + '.png')
            os.remove('demo' + str(i) + '.jpg')
            i += 1
        cap.release()
        cv2.destroyAllWindows()
        return JsonResponse({"status":200,"data":None})

