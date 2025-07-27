Mobile Price Prediction Project


Overview

The Mobile Price Prediction project is a machine learning-based system designed to predict mobile phone prices based on their technical specifications, such as resolution, PPI, CPU cores, RAM, battery capacity, and more. The project includes exploratory data analysis (EDA), model training with Linear Regression and Random Forest Regressor, and a Streamlit web application for user-friendly price predictions with USD to INR conversion. This project showcases proficiency in data analysis, machine learning, and web application development, serving as a strong proof of work for an internship application.

Features

Dataset Analysis: Comprehensive EDA to identify key features influencing mobile prices, using visualizations like correlation heatmaps and scatter plots.
Machine Learning Models: Trained and evaluated Linear Regression (R²: 81.63%, MAE: $177.58) and Random Forest Regressor (R²: 83.93%, MAE: $136.26).

Web Application: A Streamlit app (app.py) for real-time price predictions, with input validation, data saving to new_data.csv, and responsive design using Tailwind CSS.


Project Report: A professional report (Mobile_Price_Prediction_Report.pdf) documenting methodology, results, and code.


Prerequisites:

To run the project, ensure you have the following installed:

Python 3.8+
Dependencies (listed in requirements.txt):pandas
numpy
scikit-learn
streamlit
matplotlib
seaborn
joblib


Setup Instructions

Clone the Repository:
git clone <repository-url>
cd Mobile-Price-Prediction


Install Dependencies:
pip install -r requirements.txt

If requirements.txt is not provided, install the dependencies listed above:
pip install pandas numpy scikit-learn streamlit matplotlib seaborn joblib


Run the Streamlit App:
streamlit run app.py


Open the provided URL (e.g., http://localhost:8501) in a browser.
Enter mobile specifications and an exchange rate to predict prices in USD and INR.
User inputs are saved to new_data.csv.


View the Jupyter Notebook:

Open MobilePricePrediction.ipynb in Jupyter Notebook or JupyterLab to review the data analysis and model training.

jupyter notebook MobilePricePrediction.ipynb


View Visualizations:

Open Mobile_Price_Visualizations.html in a browser to view interactive charts.
Ensure Cellphone.csv is accessible, or embed the data directly in the HTML file.
Alternatively, host the HTML file using a local server:npx http-server

Navigate to http://localhost:8080/Mobile_Price_Visualizations.html.



Usage

Streamlit App:
Input mobile specifications (e.g., resolution, PPI, RAM) and an exchange rate.
Click "Predict Price" to view the predicted price in USD and INR.
View saved inputs in a table, stored in new_data.csv.


Jupyter Notebook:
Run cells to perform EDA, train models, and evaluate performance.
Visualizations include histograms, correlation heatmaps, and scatter plots.


Project Report:
The PDF report details the methodology, results, challenges, and code, suitable for submission as proof of work.



Results

Model Performance:

Random Forest Regressor: R² = 83.93%, MAE = 136.26
Linear Regression: R² = 81.63%, MAE = 177.58


Key Insights:


RAM, battery capacity, and PPI are the most influential features for price prediction.

Phones with high sales (>5000 units) tend to have lower prices (average ~$1800), likely due to economies of scale.


Challenges and Solutions


Outliers: Handled using Random Forest's robustness and feature scaling.

Non-linear Relationships: Addressed by using Random Forest Regressor over Linear Regression.

User-Friendly Deployment: Achieved with a Streamlit app featuring input validation and responsive design.

Future Improvements


Add features like brand or operating system for better accuracy.

Implement hyperparameter tuning for the Random Forest model.

Enhance the Streamlit app with real-time market data via APIs.

Acknowledgments


Dataset: Cellphone.csv (source not specified, assumed public domain).

Libraries: pandas, scikit-learn, Streamlit, matplotlib, seaborn, joblib.

Visualization: Recharts for interactive charts.

Author

Kirtana Aryasomyajula
