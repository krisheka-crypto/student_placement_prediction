# Student Placement Prediction System

## Overview

The Student Placement Prediction System is a machine learning application designed to predict whether a student is likely to be placed based on academic performance, technical skills, internships, projects, and other career-related factors.

The project implements multiple machine learning algorithms and compares their performance to identify the most suitable model for deployment. A Streamlit-based web application provides an interactive interface for users to enter student information and receive placement predictions.

---

## Objectives

* Analyze student-related data and identify factors influencing placement outcomes.
* Build and evaluate machine learning models for classification.
* Compare the performance of different algorithms.
* Deploy the best-performing model through a web-based interface.

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Joblib
* Streamlit

### Machine Learning Models

* Random Forest Classifier
* XGBoost Classifier

---

## Dataset Description

The dataset contains various student-related attributes, including:

* Age
* Gender
* CGPA
* Branch
* College Tier
* Number of Internships
* Number of Projects
* Number of Certifications
* Coding Skill Score
* Aptitude Score
* Communication Skill Score
* Logical Reasoning Score
* GitHub Repositories
* LinkedIn Connections
* Mock Interview Score
* Attendance Percentage
* Backlogs
* Extracurricular Activities Score
* Leadership Score
* Study Hours per Day
* Sleep Hours
* Volunteer Experience

### Target Variable

* Placement Status (Placed / Not Placed)

---

## Methodology

### Data Preprocessing

The dataset was cleaned and prepared using the following steps:

* Removal of irrelevant columns
* Target variable encoding
* Handling categorical variables
* One-Hot Encoding
* Feature engineering

### Feature Engineering

Additional features were created to improve model learning:

* Total Skills Score
* Activity Score

### Train-Test Split

The dataset was divided into:

* Training Set: 80%
* Testing Set: 20%

### Model Training

Two machine learning algorithms were implemented:

#### Random Forest Classifier

An ensemble learning algorithm that combines multiple decision trees to improve predictive performance and reduce overfitting.

#### XGBoost Classifier

A gradient boosting algorithm known for its efficiency and predictive capabilities on structured datasets.

---

## Model Evaluation

The models were evaluated using:

* Accuracy Score
* Precision
* Recall
* F1-Score

### Results

| Model         | Accuracy |
| ------------- | -------- |
| Random Forest | ~56%     |
| XGBoost       | ~55%     |

### Best Performing Model

Random Forest Classifier

The dataset represents a complex real-world classification problem. The primary objective of this project was to demonstrate a complete machine learning workflow, including preprocessing, model development, evaluation, and deployment.

---

## Web Application

A Streamlit-based web application was developed to allow users to:

* Enter student details
* Generate placement predictions
* View results in real time

Run the application using:

```bash
streamlit run app.py
```

---

## Project Structure

```text
student_placement_prediction/
│
├── Dataset/
│   └── placement.csv
│
├── model/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/krisheka-crypto/student_placement_prediction.git
```

### Navigate to the Project Directory

```bash
cd student_placement_prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python train.py
```

### Run the Application

```bash
streamlit run app.py
```

---

<img width="1600" height="847" alt="image" src="https://github.com/user-attachments/assets/958fdb53-d985-409b-b34d-f5d83920d97a" />

<img width="1600" height="845" alt="image" src="https://github.com/user-attachments/assets/74d561ad-ebf0-4a80-9482-75d181bb1daf" />

<img width="1600" height="850" alt="image" src="https://github.com/user-attachments/assets/8a329abb-9fad-4afe-bba5-c31e0da516e7" />


## Key Learning Outcomes

This project provided practical experience in:

* Data preprocessing
* Feature engineering
* One-Hot Encoding
* Machine learning model development
* Ensemble learning techniques
* Model evaluation and comparison
* Streamlit application development
* Git and GitHub workflow

---

## Future Enhancements

* Hyperparameter optimization
* Additional feature engineering techniques
* Model performance improvement
* Cloud deployment
* Interactive analytics dashboard
* Placement probability visualization
* Resume-based placement prediction

---

## Author

Krisheka

Bachelor of Engineering in Electronics and Communication Engineering (ECE)

Areas of Interest:

* Artificial Intelligence
* Machine Learning
* Data Science
* Software Development
