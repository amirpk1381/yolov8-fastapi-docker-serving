# Production-Ready YOLOv8 Object Detection API with FastAPI & Docker

A high-performance, containerized Microservice developed to serve the *YOLOv8 object detection model in production environments. This project wraps the object detection pipeline into an asynchronous FastAPI application and packages the entire environment inside Docker for platform-independent deployment.

## 🚀 Key Architectural Features
 *Model Serviced: YOLOv8 Nano (`yolov8n.pt`) optimized for low-latency real-time inference.
 *API Framework: FastAPI leveraged for asynchronous request handling and automatic OpenAPI/Swagger documentation.
 *Containerization: Fully dockerized production environment ensuring zero dependency conflicts across edge devices or cloud servers.
 *Input/Output Pipelines: Built to ingest raw binary image payloads and return structured bounding box coordinates alongside class labels in JSON format.

---

## 📂 Repository Structure & Components

 *`app.py`: Core API application handling endpoints, image transformations (via OpenCV/PIL), and model predictions.
 *`Dockerfile`: Multistage/optimized build recipe that sets up the Python environment, installs core dependencies, cleans up caches, and fires up the Uvicorn server.
 *`requirements.txt`: Explicit pinning of required production libraries (`ultralytics`, `fastapi`, `uvicorn`, `opencv-python-headless`).
 *`yolov8n.pt`:* Pre-trained model weights compiled and ready for local inference.

---

## 🛠️ How to Run Locally (Quick Start)

### 1. Using Docker (Recommended)
Build the container image and spin up the API service instantly without needing any local library installations:

```bash
# Build the production docker image
docker build -t yolov8-api .

# Run the container mapping port 8000
docker run -d -p 8000:8000 yolov8-api

