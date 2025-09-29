import cv2
import os

if not os.path.exists('snapshots'):
    os.makedirs('snapshots')
#load the cascade 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

#start webcam
cap = cv2.VideoCapture(0)

snapshot_count = 0 #to save multiple snapshots

while True:
    #capture frame by frame
    ret, frame = cap.read()
    #convert to grayscale(better for detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    #draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        #region of interest (only inside face)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        #detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
        
        #detect smiles
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    
    #show face count
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            
    #show the results
    cv2.imshow("Face + eyes + smile detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): #quit
        break
    elif key == ord('s'): #save snapshot
        snapshot_name = f"snapshots/face_snapshot_{snapshot_count}.jpg"
        cv2.imwrite(snapshot_name, frame)
        print(f"Snapshot saved as {snapshot_name}")
        snapshot_count += 1

#release everything
cap.release()
cv2.destroyAllWindows()