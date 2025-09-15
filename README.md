# Liver Disease Prediction Web Application

This is a web application that predicts the likelihood of liver disease based on various medical parameters using machine learning.

## Features

- Modern, responsive UI built with Tailwind CSS
- Real-time predictions using machine learning
- Easy-to-use form interface
- Clear visualization of results

## Setup Instructions

1. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train and save the model:
   - Place your trained model in the `models` directory as `liver_model.pkl`
   - The model should be compatible with the input features specified in the form

3. Run the application:
   ```
   python app.py
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Input Features

The application accepts the following medical parameters:
- Age
- Gender (0-Female, 1-Male)
- Total Bilirubin
- Direct Bilirubin
- Alkaline Phosphotase
- Alamine Aminotransferase
- Aspartate Aminotransferase
- Total Proteins
- Albumin
- Albumin/Globulin Ratio

## Technology Stack

- Backend: Flask (Python)
- Frontend: HTML, JavaScript, Tailwind CSS
- Machine Learning: scikit-learn
- Additional Libraries: pandas, numpy
