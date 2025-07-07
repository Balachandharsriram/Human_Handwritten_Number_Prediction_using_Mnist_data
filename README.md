# 🖥️ Handwritten Digit Recognition with MNIST & Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📌 Overview
This project implements a deep learning model to classify handwritten digits using the **MNIST dataset**, stored here in an **HDF5 format (`mnist.hdf5`)**.  
It leverages **Convolutional Neural Networks (CNNs)** to achieve high accuracy in recognizing handwritten digits (0-9).

---

## 🗄 Dataset
- **MNIST**: A standard benchmark dataset containing 70,000 grayscale images of handwritten digits (28x28 pixels), split into:
  - **60,000 training images**
  - **10,000 testing images**
- Stored in `mnist.hdf5` for efficient loading.

---

## 🚀 Technologies Used
- **Python 3.8+**
- **TensorFlow / Keras** for building and training the model.
- **NumPy & Pandas** for data manipulation.
- **Matplotlib & Seaborn** for visualization.

---

## ⚙️ Project Workflow
1. **Load Data** from `mnist.hdf5`.
2. **Normalize & preprocess** the images.
3. **Build a CNN** architecture using Keras Sequential API.
4. **Train the model** on the training set.
5. **Evaluate the model** on the test set.
6. **Visualize training metrics** (accuracy & loss curves).
7. **Predict and display sample outputs.**

---

## 📊 Results
- Achieved **~99% accuracy** on the MNIST test dataset.
- The CNN effectively learns complex features, outperforming simple dense networks.

---

## 🚀 How to Run
1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/mnist-digit-classification.git
    cd mnist-digit-classification
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the training script or notebook:**
    ```bash
    python train_mnist.py
    ```
    OR
    ```bash
    jupyter notebook MNIST_Classification.ipynb
    ```

---

## 🔬 Future Enhancements
- Implement data augmentation for more robust learning.
- Try advanced architectures like ResNet or Capsule Networks.
- Deploy as a web app using **Flask / Streamlit** for live digit predictions.

---

## 🤝 Contributing
Pull requests are welcome. Please open an issue first to discuss changes.

---

## ✨ Acknowledgments
- [Yann LeCun et al.](http://yann.lecun.com/exdb/mnist/) for the MNIST dataset.
- TensorFlow & Keras contributors for powerful tools.

> ⭐ **If you like this project, please star the repository and follow for more!**
