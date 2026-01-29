# Eye Gaze Based Psychological State Detection

## Project Overview
This project is a computer vision–based system that analyzes **eye movements and gaze behavior** using a webcam to infer a user's psychological state. The system focuses on detecting indicators such as **stress, fatigue, concentration, and flow** by extracting eye-gaze features in real time and applying a pre-trained machine learning model.

The application is designed as an academic and experimental project, suitable for Final Year Projects (FYPs), research prototypes, and human–computer interaction studies.

---

## Key Features

- **Real-time Eye and Gaze Tracking**
  - Tracks eye position and gaze direction using a webcam.
- **Psychological State Prediction**
  - Predicts mental states such as stress, fatigue, concentration, and flow.
- **Blink and Saccade Detection**
  - Counts blinks and detects saccades, which are strong indicators of fatigue and stress.
- **Fixation Analysis**
  - Measures fixation duration and frequency to assess focus and engagement.
- **Session Recording and Data Storage**
  - Saves extracted eye-gaze parameters into CSV files for further analysis.
- **Exercise Recommendations**
  - Provides basic exercise or wellness recommendations based on detected psychological states.

---

## Technologies Used

- **Python**
- **OpenCV** – for real-time video processing
- **GazeTracking Library** – for gaze and eye movement detection
- **dlib** – for facial landmark detection
- **TensorFlow / Keras** – for machine learning model inference
- **NumPy & Pandas** – for data handling and feature processing

---

## System Workflow

1. Webcam captures real-time video.
2. Face and eye regions are detected.
3. Eye-gaze features (blinks, fixations, saccades) are extracted.
4. Features are passed to a pre-trained ML model.
5. Psychological state is predicted.
6. Results are displayed and logged to CSV.
7. Exercise recommendations are generated based on the prediction.

---

## Software Limitations

- **Accuracy Dependence**
  - Performance depends on webcam quality and lighting conditions.
- **Gaze Tracking Constraints**
  - Glasses or certain eye conditions may reduce accuracy.
- **Model Generalization**
  - The pre-trained model may not generalize well to all users, as it is trained on a limited dataset.

---

## Challenges Faced

- Integrating multiple libraries (OpenCV, GazeTracking, dlib, TensorFlow).
- Maintaining real-time performance with high-resolution video.
- Handling environmental variations such as lighting and background noise.
- Ensuring stable predictions across different users.

---

## Advantages

- Provides real-time feedback on psychological states.
- Records session data for long-term analysis.
- Monitors multiple eye-based indicators for comprehensive assessment.
- Useful for research, education, and well-being monitoring.

---

## Use Cases

- Academic Final Year Projects
- Human–Computer Interaction Research
- Cognitive Load and Fatigue Monitoring
- Productivity and Well-being Analysis

---

## Disclaimer

This project is intended for **educational and research purposes only**. It is not a medical diagnostic tool and should not be used for clinical decision-making.

---

## Author

**Fawaz Ahmed Dar**  
GitHub: https://github.com/fawazdar2196
