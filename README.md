# BioDigital Avatar for Remote Health Monitoring & Mental Wellness

## Project Overview
BioDigital Avatar is an AI-powered remote health monitoring system that uses real-time biometric sensors and machine learning to infer human emotional states, visualized through a responsive 3D avatar. The solution is designed for telehealth, mental wellness, elder care, and empathetic remote interactions.

### Key Features
- Integrates wearable sensors for heart rate, temperature, motion, and ECG.  
- Uses AI models to predict physiological and emotional states.  
- Visualizes results through animated 3D avatars in a web/mobile dashboard.  
- Real-time, privacy-conscious, and intuitive for caregivers and users.  

---

## Table of Contents
1. Introduction  
2. System Architecture  
3. Installation & Setup  
4. Usage Instructions  
5. Software Components  
6. API Reference  
7. Data & Model Details  
8. Contribution Guide  
9. License  

---

## Introduction
Traditional remote communication platforms lack emotional nuance and physiological context. BioDigital Avatar bridges this gap by connecting wearable sensor data to an AI pipeline that animates a 3D avatar reflecting real-time user emotional and physical states.

---

## System Architecture

### Hardware Stack
- ESP32 Microcontroller  
- MAX30102 (Heart rate, SpO2 sensor)  
- DS18B20 (Temperature sensor)  
- MPU6050 (Accelerometer, Gyroscope)  
- AD8232 (Analog ECG sensor)  
- TP4056 (Battery Charging module)  

### Software Stack
- Python backend (Flask REST API)  
- Machine learning models: Random Forest baseline, Hybrid Deep Learning CBCNN-LSTM-Attention models  
- Frontend: Vite & React for UI, Three.js for 3D rendering, Blender & Ready Player Me for avatar assets, Mixamo for animations  
- MongoDB database for storage and statistics  

---

## Installation & Setup

### Prerequisites
- Python 3.8+  
- Node.js (16+)  
- ESP32 module (with compatible firmware flashed)  
- MongoDB instance (local or Atlas)  
- Git  

### Backend Setup
git clone https://github.com/Sharvary-HH/BioDigitalAvatr.git
cd BioDigitalAvatr/backend

Create a virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

(Optional) Download model files if not present
Copy configuration .env file as per template
python app.py # Starts Flask REST API


### Frontend Setup
cd ../frontend
npm install
npm run dev # Starts the React dashboard (default: http://localhost:3000)


### Hardware Integration
- Connect ESP32 and sensors as per wiring instructions in the docs or schematic.  
- Flash data collection firmware and ensure WiFi/Bluetooth connectivity.  

---

## Usage Instructions
1. Start sensor data streaming on the ESP32 device.  
2. Run REST backend to receive and preprocess streaming data.  
3. The backend will run ML inference to detect emotional and physical states.  
4. Open the dashboard in a browser to monitor real-time health and see the animated avatar respond to detected states.  
5. Access data statistics and export logs as required.  

### Example API Request
Send latest 100 sensor readings for emotion prediction:
curl -X POST http://localhost:5000/predict-emotion
-d @sensor_data.json


### Dashboard
- Visualizes sensor metrics (BPM, SpO2, temperature, motion, ECG).  
- Maps data to avatar emotions (happy, sad, angry, surprised, etc.).  
- Animations triggered live based on ML output.  

---

## Software Components

| Component       | Technology                 | Purpose                                    |
|-----------------|----------------------------|--------------------------------------------|
| Backend REST API| Flask (Python)             | Data ingestion, preprocessing, ML inference |
| ML Model        | Random Forest / Deep Hybrid| Emotion/activity prediction                 |
| Frontend        | Vite, React & Three.js     | UI dashboard, 3D avatar visualization       |
| Avatar Assets   | Blender, Ready Player Me, Mixamo | Rigging and animation                  |
| Database        | MongoDB                    | Data persistence, stats, logs               |

---

## API Reference
- **/predict-emotion (POST)**: Upload sensor readings, get predicted emotion/activity.  
- **/get-history (GET)**: Fetch previous prediction data.  
- **/dashboard-data (GET)**: Get real-time stats for dashboard visualization.  

Refer to backend docs/API documentation for full endpoint details.

---

## Data & Model Details
- **Supported features**: BPM, SpO2, Body Temp., ECG, Accelerometer, Gyroscope.  
- **Model weights**:  
  - Random Forest: `backend/bio_avatar_models/baseline_rf_emotion.pkl`  
  - Deep learning models stored in subfolders.  
- **Real-time pipeline**: Sliding window (100 readings), PCA feature extraction, ML inference.  

### Example Outputs
- Emotions with confidence scores: `"Fear: 0.61, Angry: 0.62, Neutral: 0.74"`  
- Activities with scores: Sitting/Standing/Walking/Running  

---

## Contribution Guide
- Fork the repository and create feature branches.  
- Write clear commit messages and document major code changes.  
- Ensure software and data privacy standards are upheld.  
- Run all test cases and validate with sample sensor streams.  
- For UI/UX or animation, export compatible `.glb/.fbx` assets.  
- Open a pull request with a clear explanation of feature or fix.  

---

For more details, consult the documentation and hardware schematics. Contributions and collaborations are welcome!  
