# 📌 ASL Alphabet Recognition with Machine Learning

## 📖 Overview
This project implements a machine learning model to recognize American Sign Language (ASL) alphabet letters from hand keypoints extracted using **MediaPipe**. The model is trained on a dataset of labeled images and uses **TensorFlow** for classification.

There is a skeleton for the implementation of a similar model for the MS-ASL dataset, which contains the tools for extraction and treatment of the initial dataset. 

## 🚀 Features
- **ASL Alphabet Recognition**: Detects letters A-Z.
- **Keypoint Extraction**: Uses **MediaPipe Hands** to extract 2D hand landmarks.
- **Deep Learning Model**: A TensorFlow-based neural network trained on extracted keypoints.
- **Streamlit Dashboard**: Visualizes model performance with statistics and plots.
- **Real-Time Prediction**: Supports live inference using a webcam.

## 📂 Project Structure
```
├── alphabet/             # Alphabet project main dir
├── models/               # Saved TensorFlow models (.h5) with statistics
├── dashboard.py/         # Streamlit dashboard
├── live-translator.py    # Live visualization tool using the webcam
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies
```

## 🔧 Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/eduardoparracho/ASL-TRANSLATION.git
   ```
2. **Create a virtual environment (optional)**
   ```bash
   python -m venv env
   source env/bin/activate 
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Training the Model
```
alphabet/model-development.ipynb
```
Download the dataset through https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset?resource=download

## 🎥 Running Live Prediction
```bash
python live-translator.py
```
This script uses your **webcam** to predict ASL letters in real-time.

## 📈 Running Streamlit Dashboard
```bash
streamlit run dashboard.py
```
This launches an **interactive dashboard** to visualize model performance.
You have to save your model to /models/(your-model-name)/(your-model)

## 🤝 Contributing
Feel free to open issues or submit PRs to improve this project!

Made with ❤️ for ASL accessibility! 🚀
Eduardo Parracho - Ironhack AD 2025

