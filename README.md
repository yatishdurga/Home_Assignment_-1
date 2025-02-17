# Home_Assignment_-1
# TensorFlow Experiments in Google Colab

## 📌 Overview
This repository contains TensorFlow-based implementations of key deep learning concepts, including tensor manipulations, loss functions, optimizers, and training a neural network with TensorBoard. The experiments are run in **Google Colab**.

## 📂 Project Structure
```
├── Tensor_Manipulations.ipynb      # Tensor reshaping and operations
├── Loss_Functions_Tuning.ipynb     # Comparing loss functions
├── Optimizers_Comparison.ipynb     # Training with Adam vs. SGD
├── Neural_Network_TensorBoard.ipynb # Training and logging with TensorBoard
├── README.md                        # Project Documentation
```

## 🔥 1. Tensor Manipulations & Reshaping
### ✅ Tasks:
- Create a random tensor of shape **(4,6)**.
- Find its **rank and shape**.
- Reshape it into **(2,3,4)** and then **transpose** it to **(3,2,4)**.
- Broadcast a smaller tensor **(1,4)** and add it to the larger tensor.
- Explain **broadcasting in TensorFlow**.

### 📌 Expected Output:
- Print tensor **rank and shape** before and after reshaping/transposing.

📖 **Concepts Covered:** Tensor Operations, Reshaping, Transposing, Broadcasting

---

## 🎯 2. Loss Functions & Hyperparameter Tuning
### ✅ Tasks:
- Define **true values (y_true)** and **model predictions (y_pred)**.
- Compute:
  - **Mean Squared Error (MSE)** – Used in regression.
  - **Categorical Cross-Entropy (CCE)** – Used in classification.
- Modify predictions and observe **loss changes**.
- Plot **MSE vs. CCE loss values** using Matplotlib.

### 📌 Expected Output:
- Loss values printed for different predictions.
- **Bar chart comparing MSE and CCE loss.**

📖 **Concepts Covered:** Loss Functions, Model Performance Metrics

---

## ⚡ 3. Train a Model with Different Optimizers
### ✅ Tasks:
- Load the **MNIST dataset**.
- Train two models:
  - One with **Adam Optimizer**.
  - One with **SGD Optimizer**.
- Compare **training and validation accuracy trends**.

### 📌 Expected Output:
- Accuracy plots comparing **Adam vs. SGD performance**.

📖 **Concepts Covered:** Optimizers, Model Training, Accuracy Comparison

---

## 📊 4. Train a Neural Network and Log to TensorBoard
### ✅ Tasks:
- Load the **MNIST dataset** and preprocess it.
- Train a **simple neural network** with TensorBoard logging.
- Launch **TensorBoard in Google Colab**.
- Analyze **training vs. validation accuracy and loss** trends.

### 📌 Expected Output:
- Model trains for **5 epochs**, logs stored in `logs/fit/`.
- Visualization of **training vs. validation accuracy and loss** in TensorBoard.

📖 **Concepts Covered:** Deep Learning Model Training, TensorBoard, Visualization

---

## 🔍 4.1 Questions Answered
1️⃣ **What patterns do we observe in accuracy curves?**
   - Training accuracy **increases**, but validation accuracy may **plateau or drop** due to overfitting.

2️⃣ **How to detect overfitting with TensorBoard?**
   - If **training loss keeps decreasing** but **validation loss increases**, the model is overfitting.
   - Solutions: **Dropout, L2 Regularization, Data Augmentation**.

3️⃣ **What happens if we increase epochs?**
   - Initially, **both accuracies improve**.
   - After too many epochs, the model **overfits** – training accuracy remains high, but validation accuracy drops.
   - Solutions: **Early Stopping, Reduce Learning Rate**.

---

## 🛠️ How to Run the Code in Google Colab
1. Open **Google Colab**.
2. Upload the notebooks from this repository.
3. Run the cells step by step.
4. For TensorBoard visualization, use:
   ```python
   %load_ext tensorboard
   %tensorboard --logdir logs/fit/
   ```

