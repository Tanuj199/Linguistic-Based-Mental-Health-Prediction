# 🧠 Linguistic-Based Mental Health Risk Prediction

## 📌 Project Overview
This project presents a machine learning solution to a critical social and public health problem: **the early identification of mental health risk in online text**.  

The model analyzes subtle **linguistic patterns** within self-reported text from a public dataset to predict a user's risk level.  
⚠️ **Note:** The goal is *not to diagnose*, but to provide a tool for early detection that can be integrated into online support systems.

---

## 📂 Project Files and Structure
This project is structured into four sequential Python scripts and several data files.  
👉 **You must run the scripts in order for the project to work correctly.**

- `1_data_preprocessing.py`  
  Handles data preparation:  
  - Loads raw text data  
  - Cleans text  
  - Tokenizes words → numerical sequences  
  - Splits dataset into training, validation, testing  
  - Saves processed data + tokenizer  

- `2_model_training.py`  
  - Builds & trains a **Long Short-Term Memory (LSTM)** neural network  
  - Uses callbacks to save the best-performing model based on validation accuracy  

- `3_model_evaluation.py`  
  - Evaluates the trained model on test data  
  - Generates metrics: **accuracy, precision, recall**  

- `4_interactive_app.py`  
  - Final product using **Gradio**  
  - Provides an interactive web interface to type text & get **real-time risk prediction**  

### 📁 Data Files
- `depression_dataset_reddit_cleaned.csv` → Primary dataset  
- `processed_data.npz` → Preprocessed train/val/test splits  
- `tokenizer.pkl` → Saved Keras Tokenizer  
- `best_model.h5` → Final trained model (weights + architecture)  

---

## 🚀 Getting Started
This project is designed to run in **Google Colab**.  

### Steps:
1. **Upload Dataset**  
   Upload `depression_dataset_reddit_cleaned.csv` into your Colab file browser.  

2. **Run Scripts Sequentially**  
   Create new code cells for each phase and run in order:  

---

## 📝 Phase 1: Data Preprocessing (`1_data_preprocessing.py`)
- Cleans and preprocesses dataset  
- Tokenizes and pads sequences  
- Splits data into train/val/test  
- Saves processed data + tokenizer  

---

## 🏋️ Phase 2: Model Training (`2_model_training.py`)
- Builds **LSTM neural network**  
- Trains on processed data  
- Uses **EarlyStopping & ModelCheckpoint**  
- Saves best model → `best_model.h5`  

---

## 📊 Phase 3: Model Evaluation (`3_model_evaluation.py`)
- Loads trained model + test data  
- Evaluates using:  
  - Accuracy  
  - Precision, Recall, F1-score  
  - Confusion Matrix (visualized with Seaborn)  

---

## 💻 Phase 4: Interactive Application (`4_interactive_app.py`)
- Loads trained model & tokenizer  
- Uses **Gradio** to build an interactive web app  
- Input: user text  
- Output:  
  - Risk Level (**High Risk / Low Risk**)  
  - Message for user guidance  

---

## ⚠️ Important Notes
- This tool is for **educational/research purposes only**  
- Not intended for medical diagnosis  
- Predictions indicate **risk patterns**, not clinical outcomes  

---

## 🛠️ Tech Stack
- **Python**  
- **TensorFlow / Keras** (LSTM model)  
- **scikit-learn** (evaluation)  
- **Gradio** (web interface)  
- **pandas, numpy, seaborn, matplotlib**  

---

## ✅ Example Workflow
1. Upload dataset → `depression_dataset_reddit_cleaned.csv`  
2. Run `1_data_preprocessing.py` → saves `processed_data.npz` & `tokenizer.pkl`  
3. Run `2_model_training.py` → trains & saves `best_model.h5`  
4. Run `3_model_evaluation.py` → check performance metrics  
5. Run `4_interactive_app.py` → launch Gradio app for predictions  

---

## 📌 Final Thoughts
This project demonstrates how **linguistic analysis with machine learning** can aid in **early detection of mental health risks**.  
With further development, it could be integrated into **online support platforms** for real-world impact.  

---
