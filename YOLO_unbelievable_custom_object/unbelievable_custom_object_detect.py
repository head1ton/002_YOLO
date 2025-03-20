import base64
import os
import threading
import time

import cv2
import cvzone
import pygame
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from ultralytics import YOLO

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

class PackageDetectionProcessor:
    def __init__(self, video_file, yolo_model_path="best.pt"):
        self.gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

        try:
            self.yolo_model = YOLO(yolo_model_path)
            self.names = self.yolo_model.names
        except Exception as e:
            raise RuntimeError(f"Error loading YOLO model: {e}")

        self.cap = cv2.VideoCapture(video_file)
        if not self.cap.isOpened():
            raise FileNotFoundError("Error: Could not open video file.")

        self.cx1 = 416
        self.offset = 6
        self.process_track_ids = set()
        self.cropped_images_folder = "cropped_packages"
        os.makedirs(self.cropped_images_folder, exist_ok=True)

        self.current_date = time.strftime("%Y-%m-%d")
        self.output_filename = f"package_data_{self.current_date}.txt"

        pygame.mixer.init()
        self.sound_playing = False
        self.alert_sound = "alert.mp3"

        if not os.path.exists(self.output_filename):
            with open(self.output_filename, "w", encoding="utf-8") as file:
                file.write("Timestamp | Track ID | Package Type | Box Front Open | Damage Condition\n")
                file.write("-" * 85 + "\n")


    def analyze_image_with_gemini(self, image_path):
        try:
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")

            message = HumanMessage(
                content=[
                    {"type": "text", "text": """
                    Analyze the given image of a package and extract the following details:
                    
                    - **Box Front Open (Yes/No)**
                    - **Damage Condition (Yes/No)**
                    
                    Return results in table format only:
                    | Package Type (box) | Box Front Flap Open (Yes/No) | Damage Condition (Yes/No) |
                    |--------------------|----------------------------|--------------------------|
                    """},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            )

            response = self.gemini_model.invoke([message])
            return response.content.strip()

        except Exception as e:
            print(f"Error invoking Gemini model: {e}")
            return "Error processing image."

    def play_alert(self):
        if not self.sound_playing:
            pygame.mixer.music.load(self.alert_sound)
            pygame.mixer.music.play(-1) # loop indefinitely
            self.sound_playing = True

    def stop_alert(self):
        if self.sound_playing:
            pygame.mixer.music.stop()
            self.sound_playing = False

    def process_crop_image(self, image, track_id):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        image_filename = os.path.join(self.cropped_images_folder, f"package_{track_id}_{timestamp}.jpg")
        cv2.imwrite(image_filename, image)

        response_content = self.analyze_image_with_gemini(image_filename)
        extracted_data = response_content.split("\n")[2:]

        alert_triggered = False

        if extracted_data:
            with open(self.output_filename, "a", encoding="utf-8") as file:
                for row in extracted_data:
                    if "----------" in row or not row.strip():
                        continue

                    values = [col.strip() for col in row.split("|")[1:-1]]
                    if len(values) == 3:
                        package_type, box_open, damage_status = values
                        file.write(f"{timestamp} | Track ID: {track_id} | {package_type} | {box_open} | {damage_status}\n")

                        if box_open.lower == "yes":
                            alert_triggered = True

        if alert_triggered:
            self.play_alert()
            print("Package alert triggered!")
        else:
            self.stop_alert()

    def crop_and_process(self, frame, box, track_id):
        if track_id in self.process_track_ids:
            return

        x1, y1, x2, y2 = map(int, box)
        cropped_image = frame[y1:y2, x1:x2]
        self.process_track_ids.add(track_id)
        threading.Thread(target=self.process_crop_image, args=(cropped_image, track_id), daemon=True).start()

    def process_video_frame(self, frame):
        frame = cv2.resize(frame, (1020, 600))
        results = self.yolo_model.track(frame, persist=True)

        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [-1] * len(boxes)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2

                if self.cx1 - self.offset < cx < self.cx1 + self.offset:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cvzone.putTextRect(frame, f"ID: {track_id}", (x2, y2), 1, 1)
                    self.crop_and_process(frame, box, track_id)

        return frame

    @staticmethod
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            print(f"Mouse Position: ({x}, {y})")

    def start_processing(self):
        cv2.namedWindow("Package Detection")
        cv2.setMouseCallback("Package Detection", self.mouse_callback)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            processed_frame = self.process_video_frame(frame)
            cv2.line(frame, (416, 2), (416, 599), (0, 255, 0), 2)
            cv2.imshow("Package Detection", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_file = "co.mp4"
    processor = PackageDetectionProcessor(video_file)
    processor.start_processing()