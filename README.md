# Fire and Smoke Detection System

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-orange.svg)](https://ultralytics.com/)

A robust computer vision solution designed to detect fire and smoke in real-time using YOLO architecture. This project provides automated alerts and logging via Firebase.

## 🚀 Features
- **Real-time Detection:** High-speed inference for fire and smoke.
- **Firebase Integration:** Sends detection logs directly to the cloud.
- **Cross-Platform:** Can be deployed on edge devices or servers.
- **Easy Deployment:** Modular structure for quick setup.

## 📂 Project Structure
```text
fire-smoke-detection/
├── samples/          # Test images and videos
├── datasets/         # Training and validation data
├── runs/             # Model training outputs
├── detect.py         # Main detection script
├── firebase.py       # Firebase connection module
├── db.py             # Database operations
└── data.yaml         # Dataset configuration
🛠️ Installation
Clone the repository:

Bash
   git clone [https://github.com/eminosmanatci/fire-smoke-detection.git](https://github.com/eminosmanatci/fire-smoke-detection.git)
   cd fire-smoke-detection
Install dependencies:

Bash
   pip install -r requirements.txt
💡 Usage
To run the detection on a local file:

Bash
python detect.py --source samples/fire_video_01.mp4
📝 Note on Labels
The model was trained with the following classes:

Ateş (Fire)

Duman (Smoke)

🤝 Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.
