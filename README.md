# Machine-Learning - Eye Tracking System
# 👁️ AI Eye Tracking System (Focus Detection)

An AI-based real-time eye tracking system that detects user attention levels (Focused, Distracted, Sleeping) using computer vision and machine learning techniques.

---

## 📌 Project Description

The **AI Eye Tracking System** is a real-time monitoring application that uses webcam input to analyze eye movements and determine user focus levels. It leverages facial landmark detection to track eye behavior and classify attention into three states:

- ✅ Focused  
- ⚠️ Distracted  
- 😴 Sleeping  

The system continuously tracks focus over time and generates a performance report, helping improve productivity and awareness.

---

## ✨ Features

- 🎥 Real-time webcam-based eye tracking  
- 👁️ Eye aspect ratio calculation for focus detection  
- 🧠 Intelligent classification:
  - Focused
  - Distracted
  - Sleeping  
- 🔊 Alert system using sound notifications  
- 📊 Focus percentage tracking over time  
- 📁 Automatic CSV report generation  
- 📈 Data visualization using graphs  

---

## 🛠️ Technologies Used

- **Python**
- **OpenCV** – for video capture & image processing  
- **MediaPipe** – for facial landmark detection  
- **NumPy** – for mathematical calculations  
- **Matplotlib** – for visualization  
- **CSV** – for storing focus data  

---

## ⚙️ How It Works

- The system captures video from the webcam  
- Detects face and extracts eye landmarks  
- Calculates **Eye Aspect Ratio (EAR)**  
- Classifies attention state:
  - Low EAR → Sleeping  
  - Medium EAR → Distracted  
  - High EAR → Focused  
- Tracks focus percentage over time  
- Saves results into a CSV file  

---

## 📂 Project Structure
Eye-Tracking-System/
│
├── main.py # Main real-time tracking system
├── analysis.py # Data visualization (graph)
├── focus_report.csv # Generated focus data
├── requirements.txt # Dependencies
└── face_landmarker.task # MediaPipe model file


---

## 📊 Output

- Real-time display with:
  - Focus status  
  - Focus percentage  
  - Time elapsed  

- CSV file generated:

- Key Logic

From your implementation :

Uses MediaPipe Face Landmarker
Tracks eye landmarks (468-point mesh)

Computes eye ratio:

Eye Ratio = Vertical Distance / Horizontal Distance
Determines focus based on threshold values
🚀 Future Improvements
Deep learning-based gaze tracking
Mobile app integration
Attention analytics dashboard
Multi-user monitoring system
Cloud storage for reports



 Acknowledgement

This project demonstrates the application of Computer Vision + Machine Learning in real-time human attention tracking and productivity enhancement.
