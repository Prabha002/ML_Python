# Diabetes Prediction using SVM

# Tools & Libraries Used

- Python
- Pandas, NumPy
- Scikit-learn (SVM, preprocessing, metrics)
- Matplotlib , Seaborn


# Objective

To build a machine learning model that can predict whether a person is likely to have diabetes based on medical diagnostic measurements using a Support Vector Machine (SVM) classifier.

# Problem Statement

Early detection of diabetes is critical in preventing long-term health issues. The dataset contains diagnostic features such as glucose level, BMI, blood pressure, and more for female individuals. The goal is to develop a model that can classify individuals as diabetic or non-diabetic.

This project uses the **PIMA Indians Diabetes Dataset**:
- Explore and preprocess the data.
- Train a classification model using Support Vector Machine.
- Evaluate the model's performance using accuracy and other metrics.

# Solution

1. **Data Loading and Exploration**

- Used the PIMA dataset containing 768 entries and 8 input features.
- Checked for null values, class imbalance, and provided a statistical summary.

2. **Preprocessing**

- Applied feature scaling using `StandardScaler` for better SVM performance.
- Addressed potential data inconsistencies.

3. **Model Training**

- Chose **Support Vector Machine (SVM)** as the main model.
- Split the dataset into training and testing sets using an 80-20 ratio.
- Trained the SVM with a linear or RBF kernel.

4. **Model Evaluation**

- Evaluated using accuracy, confusion matrix, precision, recall, and F1-score.
- Tested the model with new inputs using the same scaler applied during training.

5. **Prediction**

- Implemented a function to input custom values and return whether the prediction is Diabetic or Not Diabetic.

# Conclusion

The SVM model achieved satisfactory accuracy in predicting diabetes, showing how effective machine learning can be in early disease detection. Proper preprocessing, especially feature scaling, was essential for the model's performance. This project emphasizes how data-driven models can help in proactive health screening and decision-making.

** This is a YouTube guided project were I got hands on experiance and more knowledge about Machine Learning and Models. **

