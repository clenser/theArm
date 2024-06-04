# theArm Control with Mediapipe and OpenCV

This project uses Mediapipe and OpenCV to track hand movements and gestures through a webcam. The tracked hand data can be used to control various devices such as servos on a theArm robotic arm.

## Features

- Hand tracking and gesture recognition using Mediapipe.
- Real-time video capture and annotation using OpenCV.
- Calculates angles and distances based on hand landmarks for potential control applications.

## Requirements

- Python 3.x
- OpenCV
- Mediapipe

## Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/clenser/mearm-control.git
    cd mearm-control
    ```

2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

Run the script to start the hand tracking and gesture recognition:
```sh
python mearm_control.py
