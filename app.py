from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# مدل فقط یک بار لود میشه (خیلی مهم برای سرعت)
model = YOLO("yolov8n.pt")


@app.get("/")
def home():
    return {"status": "YOLO API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # 1. خواندن فایل از کاربر (bytes)
    image_bytes = await file.read()

    # 2. تبدیل bytes به فایل قابل فهم در RAM
    image = Image.open(io.BytesIO(image_bytes))

    # 3. اجرای YOLO
    results = model(image)

    detections = []

    # 4. استخراج نتایج
    for box in results[0].boxes:
        detections.append({
            "class": model.names[int(box.cls[0])],
            "confidence": float(box.conf[0])
        })

    # 5. خروجی API
    return {"detections": detections}