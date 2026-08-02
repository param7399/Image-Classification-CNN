# 🖼️ Image Classification using CNN (CIFAR-10)

A Deep Learning project that classifies images into one of the ten CIFAR-10 categories using a Convolutional Neural Network (CNN). The project includes a Streamlit web application where users can upload an image and receive a prediction with confidence scores.

---

## 📌 Features

- Image Classification using CNN
- CIFAR-10 Dataset
- Upload Image
- Top Prediction
- Top-3 Predictions
- Confidence Score
- Streamlit Web Interface
- Accuracy & Loss Graphs

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- Streamlit
- OpenCV
- NumPy
- Matplotlib

---

## 📂 Dataset

The project uses the **CIFAR-10 Dataset**, which contains **60,000** color images across **10 classes**:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

---

## 📁 Project Structure

```
Image-Classification-CNN/

│── app.py
│── train_model.py
│── predict.py
│── requirements.txt
│── README.md
│── .gitignore
│── LICENSE

├── model/
│     cnn_model.keras

├── screenshots/

├── uploads/

└── assets/
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Image-Classification-CNN.git
```

Move into the project directory

```bash
cd Image-Classification-CNN
```

Install dependencies

```bash
pip install -r requirements.txt
```

Train the model

```bash
python train_model.py
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

## 📊 Model

- CNN Architecture
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Dataset: CIFAR-10

---

## 📈 Results

- Training Accuracy: ~80%
- Validation Accuracy: ~75%

---

## 🚀 Future Improvements

- Transfer Learning (ResNet, MobileNet)
- Webcam Prediction
- Image Drag & Drop
- Dark Theme UI
- Cloud Deployment

---

## 📜 License

This project is licensed under the MIT License.