Driver Drowsiness Detection System

Introduction

The Driver Drowsiness Detection System is designed to enhance road safety by monitoring drivers and detecting signs of drowsiness. It uses computer vision techniques to analyze the driver's facial features in real-time, providing alerts if drowsiness is detected. This system aims to reduce the risk of accidents caused by driver fatigue.

Features

Real-time Monitoring: Continuously monitors the driver's face using a webcam.
Eye Blink Detection: Analyzes eye blinks to detect drowsiness.
Yawning Detection: Monitors mouth movements to detect yawning.
Head Pose Estimation: Tracks head movements to detect head nodding or head down positions.
Alert System: Provides audio alerts when signs of drowsiness are detected.

Prerequisites:

Python 3.x

OpenCV

dlib

imutils

numpy

scipy

pygame

pyttsx3

Installation

1.Clone the Repository
2.Create a Virtual Environment (Optional but Recommended)
3.Install Dependencies
4.Download Pre-trained Models

Download the required pre-trained model for facial landmark detection:

shape_predictor_68_face_landmarks.dat
Extract the file and place it in the project directory.
5.Run the Detection Script
python pgm.py




