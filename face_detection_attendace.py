import pandas as pd
import cv2
import urllib.request
import numpy as np
import os
from datetime import datetime, timezone, timedelta
import face_recognition
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import requests
import time

# Configuration
path = os.path.join(os.getcwd(), 'image_folder')
attendance_dir = os.path.join(os.getcwd(), 'attendance')
url = 'http://192.168.2.200/cam-hi.jpg'
flash_url = 'http://192.168.2.200/flash'  # New endpoint for flash control

# Google Sheets configuration
GOOGLE_SHEET_ID = "1duSqoo9TsavNX3v5pSGycLNwvzQqZa9UyHnW6n4IXwE"
CREDENTIALS_FILE = "credentials.json"
SHEET_NAME = "Sheet1"

# Timezone configuration (+0545, Nepal Time)
TZ = timezone(timedelta(hours=5, minutes=45))

# Ensure attendance directory exists
os.makedirs(attendance_dir, exist_ok=True)

# Initialize attendance CSV with timestamp
timestamp = datetime.now(TZ).strftime('%Y%m%d_%H%M%S')
attendance_file = os.path.join(attendance_dir, f'Attendance_{timestamp}.csv')
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=['Name', 'Time'])
    df.to_csv(attendance_file, index=False)

# Google Sheets authentication
def authenticate_google_sheets():
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet(SHEET_NAME)
        return sheet
    except Exception as e:
        print(f"Google Sheets authentication failed: {e}")
        return None

# Function to append attendance to Google Sheet
def append_to_google_sheet(sheet, name, time_str):
    try:
        if sheet:
            sheet.append_row([name, time_str])
            print(f"Appended {name}, {time_str} to Google Sheet")
        else:
            print("Google Sheet not authenticated, skipping append")
    except Exception as e:
        print(f"Error appending to Google Sheet: {e}")

# Function to trigger flash blink
def trigger_flash():
    try:
        response = requests.get(flash_url, timeout=2)
        if response.status_code == 200:
            print("Flash triggered")
        else:
            print(f"Failed to trigger flash: {response.status_code}")
    except Exception as e:
        print(f"Error triggering flash: {e}")

# Load reference images
images = []
classNames = []
myList = os.listdir(path)
print("Reference images:", myList)
for cl in myList:
    curImg = cv2.imread(os.path.join(path, cl))
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
print("Class names:", classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if len(encodings) > 0:
            encodeList.append(encodings[0])
        else:
            print(f"No face detected in {img}")
    return encodeList

def markAttendance(name, sheet, last_flash_time):
    with open(attendance_file, 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        if name not in nameList:
            now = datetime.now(TZ)
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            append_to_google_sheet(sheet, name, dtString)
            # Trigger flash only if enough time has passed (debouncing)
            current_time = time.time()
            if current_time - last_flash_time >= 1:  # 1-second debounce
                trigger_flash()
                return current_time
    return last_flash_time

# Encode known faces
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Authenticate Google Sheets
sheet = authenticate_google_sheets()

# Initialize last flash time for debouncing
last_flash_time = 0

while True:
    try:
        # Fetch and decode image from ESP32-CAM
        img_resp = urllib.request.urlopen(url, timeout=5)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgnp, -1)
        if img is None:
            print("Failed to decode image")
            continue

        # Flip the image horizontally (mirror effect)
        img = cv2.flip(img, 1)

        # Process image for face recognition
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                last_flash_time = markAttendance(name, sheet, last_flash_time)

        cv2.imshow('Webcam', img)
        key = cv2.waitKey(5)
        if key == ord('q'):
            break
    except Exception as e:
        print(f"Error fetching image: {e}")
        continue

cv2.destroyAllWindows()