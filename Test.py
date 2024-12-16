import sys
import cv2
import numpy as np
import time
import threading
from tensorflow.keras.models import load_model
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QPushButton, QWidget
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
import geocoder
import pygame
import requests

class DrowsinessDetector(QWidget):
    def __init__(self, model_path, pushover_credentials):
        super().__init__()
        self.setWindowTitle("Drowsiness Detection")
        self.setGeometry(100, 100, 800, 600)

        # Load pre-trained Haar cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Load the trained model
        self.model = load_model(model_path)

        # Initialize camera
        self.cap = cv2.VideoCapture(0)

        # Pushover credentials
        self.pushover_user_key = pushover_credentials["user_key"]
        self.pushover_api_token = pushover_credentials["api_token"]

        # Create UI elements
        self.image_label = QLabel(self)
        self.result_label = QLabel("Status: Detecting...", self)
        self.start_button = QPushButton("Start Detection", self)
        self.stop_button = QPushButton("Stop Detection", self)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        self.setLayout(layout)

        # Connect buttons
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button.clicked.connect(self.stop_detection)

        # Timer for video feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Alarm variables
        self.sleepy_start_time = None
        self.alarm_thread = None
        self.alarm_running = False
        self.location = None

    def preprocess_eye(self, eye_region):
        """Preprocess eye image for the model."""
        resized = cv2.resize(eye_region, (75, 75))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 75, 75, 1))
        return reshaped

    def start_detection(self):
        """Start the video feed and detection."""
        self.timer.start(30)

    def stop_detection(self):
        """Stop the video feed."""
        self.timer.stop()
        self.stop_alarm()

    def update_frame(self):
        """Update the video feed and perform detection."""
        ret, frame = self.cap.read()
        if not ret:
            return

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face(s)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        status = "Awake"

        for (x, y, w, h) in faces:
            face_region = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(face_region)

            for (ex, ey, ew, eh) in eyes:
                eye_region = face_region[ey:ey+eh, ex:ex+ew]
                preprocessed_eye = self.preprocess_eye(eye_region)

                # Predict drowsiness
                prediction = self.model.predict(preprocessed_eye)[0][0]
                if prediction > 0.5:
                    status = "Sleepy"

                # Draw rectangle around eyes
                cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)

            # Draw bounding box around face and add status label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Handle alarm logic
        if status == "Sleepy":
            if self.sleepy_start_time is None:
                self.sleepy_start_time = time.time()
            elif time.time() - self.sleepy_start_time > 2:  # 3 seconds threshold
                self.start_alarm()
        else:
            self.sleepy_start_time = None
            self.stop_alarm()

        # Update UI
        self.result_label.setText(f"Status: {status}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(image))

    def start_alarm(self):
        """Start the alarm sound."""
        if not self.alarm_running:
            self.alarm_running = True
            self.alarm_thread = threading.Thread(target=self.play_alarm, daemon=True)
            self.alarm_thread.start()

            # Send Pushover notification with location if alarm is triggered
            self.send_pushover_alert()

    def stop_alarm(self):
        """Stop the alarm sound."""
        self.alarm_running = False

    def play_alarm(self):
        pygame.mixer.init()
        pygame.mixer.music.load('alarm.mp3')  # Load the alarm sound
        pygame.mixer.music.play(-1)  # Play the alarm sound in a loop
        while self.alarm_running:
            time.sleep(1)  # Keeps the thread running while the alarm plays
        pygame.mixer.music.stop() 

    def send_pushover_alert(self):
        """Send a Pushover alert."""
        if self.location is None:
            self.get_location()

        message = f"Warning: The person is feeling sleepy. Their location is {self.location}."

        data = {
            'token': self.pushover_api_token,
            'user': self.pushover_user_key,
            'message': message
        }

        try:
            response = requests.post('https://api.pushover.net:443/1/messages.json', data=data)
            if response.status_code == 200:
                print("Pushover notification sent successfully.")
            else:
                print(f"Failed to send Pushover notification: {response.status_code}")
        except Exception as e:
            print(f"Error sending Pushover notification: {e}")

    def get_location(self):
        """Get the location of the user."""
        g = geocoder.ip('me')  # This automatically fetches location based on IP
        if g.latlng:
            self.location = f"Latitude: {g.latlng[0]}, Longitude: {g.latlng[1]}"
        else:
            self.location = "Unknown"

    def closeEvent(self, event):
        """Handle app closing."""
        self.cap.release()
        self.stop_alarm()
        cv2.destroyAllWindows()
        event.accept()


# Main application
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Specify the path to your trained model
    model_path = "path_to_your_model"  

    # Specify the Pushover credentials
    pushover_credentials = {
        "user_key": "Replace with your Pushover User Key",  
        "api_token": "Replace with your Pushover API Token"  
    }

    window = DrowsinessDetector(model_path, pushover_credentials)
    window.show()

    sys.exit(app.exec_())
