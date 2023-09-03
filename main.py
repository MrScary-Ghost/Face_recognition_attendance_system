import csv
import face_recognition
import numpy as np
import cv2
from datetime import datetime

video_capture = cv2.VideoCapture(0)
sougata_face = face_recognition.load_image_file("faces/sougata.jpeg")
sougata_face_encoding = face_recognition.face_encodings(sougata_face)[0]
known_face_encodings = [sougata_face_encoding]
known_face_names = ["sougata"]
students_attendance = known_face_names.copy()
face_locations = []
face_encodings = []

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []  # Initialize face_names as an empty list for each frame
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")  # Use underscores instead of colons in the filename

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        if name in students_attendance:
            with open(f"{current_time}.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([name, current_time])
            students_attendance.remove(name)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
