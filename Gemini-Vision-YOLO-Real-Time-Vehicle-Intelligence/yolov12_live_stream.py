import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import base64
import os
import time
import threading
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from vidgear.gears import CamGear

GOOGLE_API_KEY = "AIzaSyAJIHBNfUdgy7LnwOmtVlbcgxe5bAlLCxo"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

youtube_stream_link = "https://www.youtube.com/watch?v=zPMGa6ckrqM"

stream = CamGear(source=youtube_stream_link, stream_mode=True, logging=False).start()

car_up = {}
car_down = {}

processed_directions = set()

class VehicleDetectionProcessor:
    def __init__(self, yolo_model_path="yolo12n.pt"):
        self.gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

        try:
            self.yolo_model = YOLO(yolo_model_path)
            self.names = self.yolo_model.names
        except Exception as e:
            raise RuntimeError(f"Error loading YOLO model: {e}")

        self.area = np.array([(267, 353), (386, 456), (470, 428), (351, 328)], np.int32)
        self.area1 = np.array([(364, 326), (473, 422), (562, 399), (443, 321)], np.int32)

        self.processed_track_ids = set()
        self.track_id_color_map = {}

        self.current_date = time.strftime("%Y-%m-%d")
        self.output_filename = f"vehicle_data_{self.current_date}.txt"
        self.cropped_images_folder = "cropped_vehicles"

        os.makedirs(self.cropped_images_folder, exist_ok=True)

        if not os.path.exists(self.output_filename):
            with open(self.output_filename, "w", encoding="utf-8") as file:
                file.write("Timestamp | Track ID | Vehicle Type | Vehicle Color | Direction\n")
                file.write("-" * 75 + "\n")

    @staticmethod
    def draw_info(frame, box, track_id):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cvzone.putTextRect(frame, f"ID: {track_id}", (x2, y2), 1, 1)

    def analyze_image_with_gemini(self, image_path):
        try:
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")

            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Analyze the image and extract ONLY the following details:\n\n"
                                            "|Vehicle Type(Name of Vehicle) | Vehicle Color |\n"
                                            "|--------------|--------------|"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                     "description": "Detected vehicle"}
                ]
            )

            response = self.gemini_model.invoke([message])
            return response.content.strip()
        except Exception as e:
            print(f"Error invoking Gemini model: {e}")
            return "Error processing image"

    def save_vehicle_info(self, track_id, direction, response_content, timestamp):
        lines = response_content.split("\n")
        valid_rows = []

        for row in lines:
            if "Vehicle Type" in row or "---" in row or not row.strip():
                continue
            values = [col.strip() for col in row.split("|")[1:-1]]
            if len(values) == 2:
                vehicle_type, vehicle_color = values
                valid_rows.append((vehicle_type, vehicle_color))

        if valid_rows:
            with open(self.output_filename, "a", encoding="utf-8") as file:
                for vehicle_type, vehicle_color in valid_rows:
                    file.write(f"{timestamp} | Track ID: {track_id} | {vehicle_type} | {vehicle_color} | {direction}\n")
                    self.track_id_color_map[track_id] = vehicle_color
            print(f"Data saved for track ID {track_id} - Direction: {direction}.")

    def process_crop_image(self, image, track_id, direction):
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        image_filename = os.path.join(self.cropped_images_folder, f"vehicle_{track_id}_{timestamp}.jpg")
        cv2.imwrite(image_filename, image)

        response_content = self.analyze_image_with_gemini(image_filename)
        self.save_vehicle_info(track_id, direction, response_content, timestamp)

    def crop_and_process(self, frame, box, track_id, direction):
        if (track_id, direction) in processed_directions:
            return

        processed_directions.add((track_id, direction))

        x1, y1, x2, y2 = map(int, box)
        cropped_image = frame[y1:y2, x1:x2]

        threading.Thread(target=self.process_crop_image,
                         args=(cropped_image, track_id, direction),
                         daemon=True).start()

    def process_video_frame(self, frame):
        frame = cv2.resize(frame, (1020, 500))
        results = self.yolo_model.track(frame, persist=True)

        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [-1] * len(boxes)

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                if self.names[class_id] != "car":
                    continue

                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if cv2.pointPolygonTest(self.area, (cx, cy), False) >= 0:
                    car_up[track_id] = (cx, cy)

                if track_id in car_up:
                    if cv2.pointPolygonTest(self.area1, (cx, cy), False) >= 0:
                        direction = "Up"
                        self.draw_info(frame, box, track_id)
                        self.crop_and_process(frame, box, track_id, direction)

                if cv2.pointPolygonTest(self.area1, (cx, cy), False) >= 0:
                    car_down[track_id] = (cx, cy)

                if track_id in car_down:
                    if cv2.pointPolygonTest(self.area, (cx, cy), False) >= 0:
                        direction = "Down"
                        self.draw_info(frame, box, track_id)
                        self.crop_and_process(frame, box, track_id, direction)

        return frame

    def mouse_callback(self, event, x, y, flags, params):
        if event == cv2.EVENT_MOUSEMOVE:
            print(f"Mouse Position: ({x}, {y})")

    def start_processing(self):
        cv2.namedWindow("Vehicle Detection")
        cv2.setMouseCallback("Vehicle Detection", self.mouse_callback)
        frame_count = 0

        while True:
            frame = stream.read()
            if frame is None:
                print("Stream ended or failed.")
                break

            frame_count += 1
            if frame_count % 3 != 0:
                continue

            processed_frame = self.process_video_frame(frame)
            cv2.polylines(processed_frame, [self.area], True, (0, 0, 255), 2)
            cv2.polylines(processed_frame, [self.area1], True, (0, 255, 255), 2)

            cv2.imshow("Vehicle Detection", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Exiting...")
                break

        stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    processor = VehicleDetectionProcessor()
    processor.start_processing()