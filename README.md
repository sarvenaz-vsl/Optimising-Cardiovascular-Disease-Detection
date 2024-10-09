# Predicting Cardiovascular Disease

## Project Overview
This project focuses on the detection of cardiovascular disease (CVD) by leveraging machine learning models to enhance diagnostic accuracy and improve patient outcomes. The goal is to predict the likelihood of CVD using various clinical parameters, comparing multiple machine learning algorithms to identify the best-performing model.

## Cardiovascular Disease Targeted
The models aim to predict the presence of cardiovascular diseases based on patient data, addressing the global challenge of CVD diagnosis.

## Dataset
Data consists of patients information, with 14 distinct features including physiological and medical indicators:
- **Age**: Patient age in years (29 to 77).
- **Sex**: Gender of the patient (1 = male, 0 = female).
- **Chest Pain Type (CP)**: Four categories ranging from asymptomatic to severe pain.
- **Resting Blood Pressure (Trestbps)**: Blood pressure upon hospital admission.
- **Serum Cholesterol (Chol)**: Cholesterol level in mg/dl.
- **Fasting Blood Sugar (Fbs)**: Higher than 120 mg/dl (1 = true; 0 = false).
- **Electrocardiographic Results (Restecg)**: ECG results with values 0, 1, 2.
- **Maximum Heart Rate Achieved (Thalach)**.
- **Exercise-Induced Angina (Exang)**: Indicates if angina occurred (1 = yes, 0 = no).
- **ST Depression (Oldpeak)**: ST depression relative to rest.
- **Slope**: The slope of the peak exercise ST segment.
- **Number of Major Vessels (CA)**: Colored by fluoroscopy (0-3).
- **Thalassemia (Thal)**: A blood disorder (3 = normal, 6 = fixed defect, 7 = reversible defect).
- **Target**: Indicates diagnosis of heart disease (1 = present, 0 = not present).

## Modeling Approach
Several machine learning algorithms were employed to predict cardiovascular disease, including:
- **Logistic Regression**
- **Naive Bayes**
- **Random Forest**
- **XGBoost**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Classifier (SVC)**
- **Decision Tree**

### Data Preprocessing
- The dataset was split into training (80%) and test (20%) sets for effective model evaluation.
- **StandardScaler** was applied to normalize the feature values, ensuring better performance for algorithms sensitive to scale, such as KNN.

### Evaluation Metrics
The models were evaluated using various metrics, including:
- **Accuracy**: Overall percentage of correct predictions.
- **Confusion Matrix**: True positives, false positives, true negatives, and false negatives.
- **Precision, Recall, F1 Score**: Breakdown of model performance for each class.

## Results
The models achieved accuracies ranging from **81.97% to 88.52%**. Notably, **K-Nearest Neighbors (KNN)**, **XGBoost**, and **Support Vector Classifier (SVC)** performed the best, with an accuracy of **88.52%**.

The **Receiver Operating Characteristic (ROC)** curve analysis highlighted the **K-Nearest Neighbor model** as the top performer when considering both accuracy and clinical reliability.

## Key Features
The most important features for predicting cardiovascular disease include:
- **Chest Pain Type (CP)**
- **Thalassemia (Thal)**
- **Sex**

These features were identified as the most influential in the prediction models.

## Tools & Libraries
- **Python**: Core programming language.
- **Pandas, NumPy**: Data manipulation libraries.
- **Scikit-learn**: Machine learning library.
- **Matplotlib, Seaborn**: Visualization libraries for model evaluation.
- **StandardScaler**: Used for feature scaling.

## Conclusion
While deep learning models were explored, their performance closely matched that of traditional machine learning models, underscoring the importance of selecting the right model based on data characteristics. The final model, K-Nearest Neighbors, provides a robust solution for predicting cardiovascular disease, achieving high accuracy and reliability.
