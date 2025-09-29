BioDigital Avatar for Remote Health Monitoring & Mental Wellness
Project Overview
BioDigital Avatar is an AI-powered remote health monitoring system that uses real-time biometric sensors and machine learning to infer human emotional states, visualized through a responsive 3D avatar. The solution is designed for telehealth, mental wellness, elder care, and empathetic remote interactions.

Key Features:

Integrates wearable sensors for heart rate, temperature, motion, and ECG.

Uses AI models to predict physiological and emotional states.

Visualizes results through animated 3D avatars in a web/mobile dashboard.

Real-time, privacy-conscious, and intuitive for caregivers and users.

Table of Contents
Introduction

System Architecture

Installation & Setup

Usage Instructions

Software Components

API Reference

Data & Model Details

Contribution Guide

License

Introduction
Traditional remote communication platforms lack emotional nuance and physiological context. BioDigital Avatar bridges this gap by connecting wearable sensor data to an AI pipeline that animates a 3D avatar reflecting real-time user emotional and physical states.

System Architecture
Hardware Stack

ESP32 Microcontroller

MAX30102 (Heart rate, SpO2 sensor)

DS18B20 (Temperature sensor)

MPU6050 (Accelerometer, Gyroscope)

AD8232 (Analog ECG sensor)

TP4056 (Battery Charging module)

Software Stack

Python backend (Flask REST API)

Machine learning models: Random Forest baseline, Hybrid Deep Learning CBCNN-LSTM-Attention models

Frontend: Vite & React for UI, Three.js for 3D rendering, Blender & Ready Player Me for avatar assets, Mixamo for animations

MongoDB database for storage and statistics

Installation & Setup
Prerequisites
Python 3.8+

Node.js (16+)

ESP32 module (with compatible firmware flashed)

MongoDB instance (local or Atlas)

Git

Backend Setup
bash
git clone https://github.com/Sharvary-HH/BioDigitalAvatr.git
cd BioDigitalAvatr/backend
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# (Optional) Download model files if not present
# Copy configuration .env file as per template
python app.py  # Starts Flask REST API
Frontend Setup
bash
cd ../frontend
npm install
npm run dev  # Starts the React dashboard (default: http://localhost:3000)
Hardware Integration
Connect ESP32 and sensors as per wiring instructions in the docs or schematic

Flash data collection firmware and ensure WiFi/Bluetooth connectivity

Usage Instructions
Start sensor data streaming on the ESP32 device

Run REST backend to receive and preprocess streaming data

The backend will run ML inference to detect emotional and physical states

Open the dashboard in a browser to monitor real-time health and see the animated avatar respond to detected states

Access data statistics and export logs as required

Example API Request
Send latest 100 sensor readings for emotion prediction:

bash
curl -X POST http://localhost:5000/predict-emotion \
     -d @sensor_data.json
Dashboard
Visualizes sensor metrics (BPM, SpO2, temperature, motion, ECG) and maps them to avatar emotions (happy, sad, angry, surprised, etc.)

Animations triggered live based on ML output

Software Components
Component	Technology	Purpose
Backend REST API	Flask (Python)	Data ingestion, preprocessing, ML inference
ML Model	Random Forest / Hybrid Deep Learning	Emotion/activity prediction
Frontend	Vite, React & Three.js	UI dashboard, 3D avatar visualization
Avatar Assets	Blender, Ready Player Me, Mixamo	Rigging and animation
Database	MongoDB	Data persistence, stats, logs
API Reference
/predict-emotion (POST): Upload sensor readings, get predicted emotion/activity

/get-history (GET): Fetch previous prediction data

/dashboard-data (GET): Get real-time stats for dashboard visualization

Refer to backend docs/API documentation for full endpoint details.

Data & Model Details
Supported features: BPM, SpO2, Body Temp., ECG, Accelerometer, Gyroscope

Model weights: backend/bio_avatar_models/baseline_rf_emotion.pkl (Random Forest), deep models in corresponding subfolders

Real-time prediction pipeline: Sliding window of latest 100 sensor readings, PCA feature extraction, ML inference

Example outputs:

Emotions with confidence scores (e.g. "Fear: 0.61, Angry: 0.62, Neutral: 0.74")

Activities (Sitting/Standing/Walking/Running) with scores

Contribution Guide
Fork the repository and create feature branches

Write clear commit messages; document major code changes

Ensure software and data privacy standards are upheld

Run all test cases and validate with sample sensor streams

For UI/UX or animation contributions, export compatible .glb/.fbx assets

Open a pull request with clear explanation of feature or fix

License
This project is licensed under the MIT License. For details refer to LICENSE.

Credits
Created by:

Parth Shukla

Sharvary H H

Dileep Raj G

Hareeshkumar M

Malavika S Babu
Guided by Prof. Neeta B Malvi, RV College of Engineering

For more details, consult the project documentation and hardware schematics. Feel free to open issues or reach out for collaboration.

Note: This README uniquely emphasizes the software implementation and how end-users and developers interact with the system, providing clear steps for setup, extension, and active use.