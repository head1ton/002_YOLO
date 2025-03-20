import base64
import threading
import time
import os

import cv2
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPEN_ROUTER'),
)

area = [(48, 102), (35, 434), (831, 443), (776, 137)]


def encode_image(image):
    _, img_bytes = cv2.imencode(".jpg", image)
    return base64.b64encode(img_bytes).decode('utf-8')

def save_response_to_file(response_text, area_name):
    file_name = f"Airport ground crew operations_{area_name}.txt"
    with open(file_name, "a") as file:
        file.write(f"Response for {area_name}:\n")
        file.write(response_text + "\n")
        file.write("=" * 80 + "\n")
    print(f"Response saved to {file_name}")

def process_image(image, area_name):
    if image is None or image.size == 0:
        print(f"Skipping empty image for {area_name}")
        return "No image available for analysis."

    img_base64 = encode_image(image)

    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "<YOUR_SITE_URL>", # Optional for OpenRouter rankings
                "X-Title": "<YOUR_SITE_NAME>",  #  Optional for OpenRouter rankings
            },
            model="google/gemini-2.0-pro-exp-02-05:free",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"""
                        Identify all activities in the image.
                        Provide a structured table format with the following columns:

                        | Airport ground crew operations(details) | Area |
                        |-------------------------|------|
                        |                         | {area_name} |

                        - Ensure the "Area" column only contains "{area_name}" ("area").
                        """},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                    ]
                }
            ]
        )

        response_text = completion.choices[0].message.content
        save_response_to_file(response_text, area_name)
        return response_text

    except Exception as e:
        print(f"Error processing image for {area_name}: {e}")
        return "Error processing image."

def crop_area(frame, polygon):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, np.int32)], (255, 255, 255))
    cropped = cv2.bitwise_and(frame, frame, mask=mask)
    x, y, w, h = cv2.boundingRect(np.array(polygon, np.int32))
    # cv2.imwrite(f"cropped_image_{y+h}_{x+w}.jpg", cropped[y:y+h, x:x+w])
    return cropped[y:y+h, x:x+w]

def send_to_gemini(image, area_name):
    print(f"Sending image from {area_name} to Gemini...")
    response_text = process_image(image, area_name)
    print(f"{area_name} Response: {response_text}")

def process_and_print(frame):
    cropped_area = crop_area(frame, area)
    threading.Thread(target=send_to_gemini, args=(cropped_area, "area"), daemon=True).start()

def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse Coordinates: X={x}, Y={y}")

def start_video():
    cap = cv2.VideoCapture("airport.mp4")
    if not cap.isOpened():
        print("Error opening video file.")
        return

    cv2.namedWindow("Machinery Detection")
    cv2.setMouseCallback("Machinery Detection", mouse_event)

    last_processed_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            # print("Error: Failed to capture frame.")
            break

        frame = cv2.resize(frame, (1020, 500))

        cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 255), 2)

        cv2.putText(frame, "area", (area[0][0], area[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Machinery Detection", frame)

        if time.time() - last_processed_time > 5:
            last_processed_time = time.time()
            threading.Thread(target=process_and_print, args=(frame, ), daemon=True).start()

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_video()