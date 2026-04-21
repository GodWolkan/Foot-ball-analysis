# Foot-ball-analysis
Football video analysis system using YOLO and computer vision to track players, ball, and referees. Extracts speed, distance, team possession, and camera motion. Includes Streamlit UI for video upload, processing, and visualization.

# ⚽ Football Video Analysis System

A computer vision-based system that analyzes football match videos using YOLO and tracking algorithms. It detects players, ball, and referees, and extracts advanced insights like speed, distance, team possession, and camera movement.

---

## 🚀 Features

- 🎯 Object Detection (Players, Ball, Referees)
- 🧠 Multi-object Tracking (ByteTrack)
- 🏃 Speed & Distance Estimation
- 🎨 Team Assignment via Color Clustering
- ⚽ Ball Possession Detection
- 🎥 Camera Movement Estimation
- 🖥️ Streamlit Web App for easy interaction

---

## 🏗️ Tech Stack

- Python, OpenCV
- YOLO (Ultralytics) :contentReference[oaicite:0]{index=0}  
- Streamlit UI :contentReference[oaicite:1]{index=1}  
- NumPy, Pandas, Scikit-learn  

---

## ⚙️ How It Works

1. Upload a football video via Streamlit UI  
2. Detect objects using YOLO  
3. Track players and ball across frames  
4. Assign teams using color clustering :contentReference[oaicite:2]{index=2}  
5. Estimate:
   - Player speed & distance :contentReference[oaicite:3]{index=3}  
   - Ball possession :contentReference[oaicite:4]{index=4}  
   - Camera movement :contentReference[oaicite:5]{index=5}  
6. Generate annotated output video  

---

## 📂 Project Structure
├── app.py # Streamlit app
├── trackers/ # Tracking logic
├── utils/ # Helper functions
├── input_videos/ # Input videos
├── output_videos/ # Processed outputs
├── models/ # YOLO model


---

## ▶️ Run Locally

bash
pip install -r requirements.txt
streamlit run app.py
📌 Use Cases
Sports analytics
Player performance tracking
Match insights & visualization
AI + Computer Vision projects
🎯 Summary

This project combines deep learning and computer vision to transform raw football footage into meaningful analytics with an interactive interface.
