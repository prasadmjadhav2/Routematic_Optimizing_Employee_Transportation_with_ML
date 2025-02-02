# Routematic Machine Learning Project: Optimizing Employee Transportation with AI

🚗 **Project Overview** 🚗

This project leverages machine learning to optimize employee transportation for **Routematic** by predicting ride schedules, estimating ride durations, and enhancing shift notifications. The goal was to streamline employee commutes, reduce manual scheduling, and improve overall transportation efficiency.

---

## 🚀 Key Features:

1. **Automated Shift-Based Communication System:**
   - Predicts and notifies employees about their upcoming shift rides via automated emails and SMS.
   - Includes ride scheduling, location tracking, and transportation details.

2. **Ride Schedule & Shift Notifications:**
   - Predicts shift-based ride schedules to notify employees in advance.

3. **Ride Duration & Distance Estimation:**
   - Accurately estimates ride distance and duration using machine learning models.
   - Incorporates real-time traffic and location data for better predictions.

4. **Cost Optimization & Sustainability:**
   - Optimizes ride costs and suggests environmentally friendly options (Electric Vehicle usage).
   - Focus on reducing carbon footprint by encouraging EV adoption.

---

## 📊 Project Overview

### Objective:

- **Predict** ride schedules based on employee shifts and locations.
- **Estimate** ride duration, distance, and cost for more accurate scheduling and planning.
- **Notify** employees automatically about their ride schedules using SMS and email.

### Dataset:

- **Size:** 50,000 records
- **Features:** 33 features including:
  - Pickup Location
  - Drop Location
  - Ride Time
  - Distance
  - Shift Type
  - Vehicle Type
  - Fuel Consumption
  - Ride Feedback Score
  - And more...

---

## 🚀 Key Solutions & Achievements:

### 1️⃣ **Shift Notification Prediction (Regression):**
   - Developed a **Random Forest Regressor** to predict shift notifications, achieving a **Mean Absolute Error (MAE)** of **149.11**.
   - Notifies employees about their shift ride schedules based on employee ID and shift type.

### 2️⃣ **Ride Duration & Distance Estimation (Regression):**
   - Created a **Random Forest Regressor** model for predicting ride distance and duration.
   - Example Predictions:
     - **Predicted Distance (km):** 30.85 km
     - **Predicted Duration (min):** 47.1 min

### 3️⃣ **Ride Cost Prediction (Regression):**
   - Built another **Random Forest Regressor** to predict ride costs, achieving an MAE of **0.0109**.
   - Enables optimized pricing strategies for employee transportation.

### 4️⃣ **Shift Type Classification (Binary Classification):**
   - Developed a **Random Forest Classifier** to classify employees into **First Shift** or **General Shift** with 100% accuracy.

### 5️⃣ **Feature Engineering & Optimization:**
   - Optimized routes and scheduling by engineering pickup and drop-off mappings.
   - Applied categorical encoding and feature selection to improve model predictions.

### 6️⃣ **Deployment Readiness & Real-Time Notifications:**
   - Packaged the model as a **Pickle (.pkl) file** for real-time notifications and predictions.
   - Built an interactive input-based prediction system for employees to get real-time ride details by entering their Employee ID and Shift Type.

### 7️⃣ **Sustainability & Carbon Emission Reduction:**
   - Integrated **Electric Vehicle (EV)** usage metrics to reduce corporate transportation’s carbon footprint.

---

## 📈 Results & Business Impact:

- ✅ **Automated Ride Notifications:** Reduced the need for manual scheduling and enhanced employee convenience.
- ✅ **Optimized Route & Time Predictions:** Improved efficiency by reducing wait times and delays.
- ✅ **Cost Optimization:** Accurate ride cost estimations for better transportation budget planning.
- ✅ **Sustainable Transportation:** Promoted **EV adoption** and helped reduce carbon emissions.

---

## 📁 Files & Structure

- **`model.pkl`** - Trained machine learning model saved for deployment.
- **`data/`** - Directory containing the dataset (50,000 records).
- **`notebooks/`** - Jupyter notebooks for exploratory data analysis, model training, and evaluation.
- **`src/`** - Python scripts for data preprocessing, model building, and deployment.

---

## 📃 License
### This project is intended for study purposes only. Do not promote or sell the dataset or the code.

---

## 💬 Contact
Feel free to connect or ask questions. Looking forward to collaborating on more innovative solutions!

[Linkedin](https://www.linkedin.com/in/prasadmjadhav2)
Mail: prasadmjadhav6161@gmail.com

