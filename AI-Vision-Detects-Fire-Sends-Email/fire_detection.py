import base64
import os
import smtplib
import threading
import time
from email.message import EmailMessage

import cv2
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()

EMAIL_SENDER=os.getenv('EMAIL_SENDER')
EMAIL_RECEIVERS=os.getenv('EMAIL_RECEIVERS')
EMAIL_PASSWORD=os.getenv('EMAIL_PASSWORD')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

images_folder = "emergency"
IMAGE_PATH = os.path.join(images_folder, "latest_fire_image.jpg")
SEND_INTERVAL = 5
last_sent_time = 0

os.makedirs(images_folder, exist_ok=True)

def send_email_alert(subject, body):
    if not os.path.exists(IMAGE_PATH):
        print("Latest fire image not found.")
        return

    try:
        subject = subject.replace("\n", " ").replace("\r", "").strip()

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVERS
        msg.set_content(body)

        with open(IMAGE_PATH, "rb") as img_file:
            msg.add_attachment(img_file.read(), maintype="image", subtype="jpeg", filename="fire_alert.jpg")

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)

        print("AI Email Alert Sent Successfully.")

    except Exception as e:
        print(f"Error sending email: {e}")

def analyze_with_gemini():
    if not os.path.exists(IMAGE_PATH):
        print("Latest fire image not found.")
        return

    try:
        with open(IMAGE_PATH, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")

        message = HumanMessage(
            content=[
                {"type": "text", "text": """
                Analyze the image and determine if fire or smoke is present.
                If fire or smoke is detected, generate a complete email automatically.
                - Write a **clear and urgent subject** (avoid long text).
                - Write a **professional but urgent email body**.
                - **Include emergency contact numbers** for Fire Department and Ambulance (local numbers for MUMBAI MAHARASHTRA).
                - If no fire or smoke is detected, simply respond with "No fire detected."
                """},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )

        response = gemini_model.invoke([message])
        result = response.content.strip()
        print("AI Detection Result: ", result)

        if "No fire detected" in result:
            print("No fire detected. Skipping email alert.")
            return

        lines = result.split("\n")
        subject = lines[0] if len(lines) > 0 else "Fire Alert!"
        body = "\n".join(lines[1:]) if len(lines) > 1 else "Possible fire detected. Take immediate action."

        send_email_alert(subject, body)

        if os.path.exists(IMAGE_PATH):
            os.remove(IMAGE_PATH)

    except Exception as e:
        print(f"AI Analysis Skipped: Image could not be processed.: {e}")

def process_frame(frame):
    global last_sent_time
    current_time = time.time()

    if current_time - last_sent_time >= SEND_INTERVAL:
        last_sent_time = current_time
        cv2.imwrite(IMAGE_PATH, frame)

        ai_thread = threading.Thread(target=analyze_with_gemini)
        ai_thread.daemon = True
        ai_thread.start()

def start_monitoring(video_file):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1020, 500))
        process_frame(frame)

        cv2.imshow("Fire/Smoke Monitoring", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Monitoring Completed.")

if __name__ == "__main__":
    video_file = "fire.mp4"
    start_monitoring(video_file)