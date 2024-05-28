import cv2
import time
import os
import HandTrackingModule as htm

# Set camera dimensions
wCam, hCam = 640, 480

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Load overlay images
folderPath = "FingerImages"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))

# Initialize hand detector
pTime = 0
detector = htm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]

while True:
    # Capture frame from camera
    success, img = cap.read()
    if not success:
        break

    # Find hand and landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # Count fingers
        totalFingers = fingers.count(1)
        print(totalFingers)

        # Overlay corresponding image
        if totalFingers > 0 and totalFingers <= len(overlayList):
            h, w, c = overlayList[totalFingers - 1].shape
            overlay = cv2.resize(overlayList[totalFingers - 1], (wCam, hCam))
            img[0:hCam, 0:wCam] = overlay

        # Draw rectangle and text
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Display the image
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
