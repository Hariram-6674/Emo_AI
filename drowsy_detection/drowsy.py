from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from collections import Counter, deque
from deepface import DeepFace
from facenet_pytorch import MTCNN
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import torch
import winsound

class IntegratedMonitor:
    def __init__(self):
        # Constants for drowsiness detection
        self.EYE_AR_THRESH = 0.3
        self.EYE_AR_CONSEC_FRAMES = 30
        self.YAWN_THRESH = 20
        
        # State variables
        self.alarm_status = False
        self.alarm_status2 = False
        self.saying = False
        self.COUNTER = 0
        
        # Initialize emotion tracking
        self.emotion_history = deque(maxlen=10)
        self.mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load drowsiness detection models
        self.detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def beep(self, duration=2000):
        frequency = 2500
        winsound.Beep(frequency, duration)

    def alarm(self, msg):
        while self.alarm_status:
            print('Drowsiness detected:', msg)
            self.beep()

        if self.alarm_status2:
            print('Yawn detected:', msg)
            self.saying = True
            self.beep()
            self.saying = False

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def final_ear(self, shape):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0
        return (ear, leftEye, rightEye)

    def lip_distance(self, shape):
        top_lip = shape[50:53]
        top_lip = np.concatenate((top_lip, shape[61:64]))

        low_lip = shape[56:59]
        low_lip = np.concatenate((low_lip, shape[65:68]))

        top_mean = np.mean(top_lip, axis=0)
        low_mean = np.mean(low_lip, axis=0)

        distance = abs(top_mean[1] - low_mean[1])
        return distance

    def analyze_emotion(self, face_roi):
        try:
            if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                
                if result and 'dominant_emotion' in result[0]:
                    current_emotion = result[0]['dominant_emotion']
                    self.emotion_history.append(current_emotion)
                    return Counter(self.emotion_history).most_common(1)[0][0], result[0]['emotion']
            
            return None, None
        except Exception as e:
            print(f"Emotion analysis error: {e}")
            return None, None

    def run(self, webcam_index=0):
        print("-> Starting Video Stream")
        vs = VideoStream(src=webcam_index).start()
        time.sleep(1.0)

        while True:
            frame = vs.read()
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Drowsiness detection
            rects = self.detector.detectMultiScale(gray, scaleFactor=1.1, 
                minNeighbors=5, minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE)

            # MTCNN face detection for emotion analysis
            boxes, _ = self.mtcnn.detect(rgb_frame)

            # Process detected faces
            if len(rects) > 0:
                for (x, y, w, h) in rects:
                    rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                    shape = self.predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    # Eye analysis
                    eye = self.final_ear(shape)
                    ear = eye[0]
                    leftEye = eye[1]
                    rightEye = eye[2]

                    # Lip analysis
                    distance = self.lip_distance(shape)

                    # Draw eye contours
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                    # Draw lip contour
                    lip = shape[48:60]
                    cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

                    # Drowsiness detection
                    if ear < self.EYE_AR_THRESH:
                        self.COUNTER += 1
                        if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                            if not self.alarm_status:
                                self.alarm_status = True
                                t = Thread(target=self.alarm, args=('wake up sir',))
                                t.daemon = True
                                t.start()
                            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        self.COUNTER = 0
                        self.alarm_status = False

                    # Yawn detection
                    if distance > self.YAWN_THRESH:
                        cv2.putText(frame, "Yawn Alert", (10, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        if not self.alarm_status2 and not self.saying:
                            self.alarm_status2 = True
                            t = Thread(target=self.alarm, args=('take some fresh air sir',))
                            t.daemon = True
                            t.start()
                    else:
                        self.alarm_status2 = False

                    # Display metrics
                    cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, f"YAWN: {distance:.2f}", (300, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Emotion detection for each face detected by MTCNN
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    face_roi = rgb_frame[y1:y2, x1:x2]
                    
                    emotion, scores = self.analyze_emotion(face_roi)
                    if emotion:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, emotion, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        # Display emotion scores
                        y_pos = 90
                        for emo, score in scores.items():
                            cv2.putText(frame, f"{emo}: {score:.2f}", (10, y_pos),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                            y_pos += 20

            cv2.imshow("Integrated Monitoring", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        vs.stop()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--webcam", type=int, default=0,
                    help="index of webcam on system")
    args = vars(ap.parse_args())
    
    monitor = IntegratedMonitor()
    monitor.run(args["webcam"])