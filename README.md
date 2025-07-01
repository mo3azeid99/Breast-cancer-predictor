# Breast-cancer-predictor

# 🧠 Breast Cancer Prediction using Machine Learning

This project aims to classify breast cancer tumors as **Malignant (1)** or **Benign (0)** using various machine learning classification algorithms. It utilizes the **Breast Cancer Wisconsin Dataset**, and compares the performance of multiple models to find the most accurate classifier.

---

## 📌 Objective

- Apply and compare multiple machine learning algorithms.
- Evaluate their performance using standard classification metrics.
- Identify the best performing model.

---

## 🗃️ Dataset

- **Instances**: 569
- **Features**: 30 numerical features (mean radius, mean texture, etc.)
- **Target**: `diagnosis`  
  - B = Benign → 0  
  - M = Malignant → 1

> Note: Dataset should be in CSV format. Example: `breast_cancer_data.csv`

---

## 🧪 Algorithms Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Bagging Classifier

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Jupyter Notebook / VSCode

---

## ⚙️ How to Run

1. Clone the repo:
    ```bash
    git clone https://github.com/your-username/breast-cancer-prediction.git
    cd breast-cancer-prediction
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Place your dataset file (CSV) in the root directory.

4. Run the script:
    ```bash
    python cancer_prediction.py
    ```

---

## 📊 Evaluation Metrics

For each model, the following metrics are reported:
- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-Score

---

## 📈 Sample Output

         id  radius_mean  texture_mean  ...  symmetry_worst  fractal_dimension_worst  diagnosis
0    842302        17.99         10.38  ...          0.4601                  0.11890          M 
1    842517        20.57         17.77  ...          0.2750                  0.08902          M 
2  84300903        19.69         21.25  ...          0.3613                  0.08758          M 
3  84348301        11.42         20.38  ...          0.6638                  0.17300          M 
4  84358402        20.29         14.34  ...          0.2364                  0.07678          M 

[5 rows x 32 columns]

=== Logistic Regression ===
Accuracy: 0.9736842105263158
Confusion Matrix:
 [[70  1]
 [ 2 41]]
Classification Report:
               precision    recall  f1-score   support

           0       0.97      0.99      0.98        71
           1       0.98      0.95      0.96        43

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114


=== Decision Tree ===
Accuracy: 0.9473684210526315
Confusion Matrix:
 [[68  3]
 [ 3 40]]
Classification Report:
               precision    recall  f1-score   support

           0       0.96      0.96      0.96        71
           1       0.93      0.93      0.93        43

    accuracy                           0.95       114
   macro avg       0.94      0.94      0.94       114
weighted avg       0.95      0.95      0.95       114


=== Random Forest ===
Accuracy: 0.9649122807017544
Confusion Matrix:
 [[70  1]
 [ 3 40]]
Classification Report:
               precision    recall  f1-score   support

           0       0.96      0.99      0.97        71
           1       0.98      0.93      0.95        43

    accuracy                           0.96       114
   macro avg       0.97      0.96      0.96       114
weighted avg       0.97      0.96      0.96       114


=== KNN ===
Accuracy: 0.9473684210526315
Confusion Matrix:
 [[68  3]
 [ 3 40]]
Classification Report:
               precision    recall  f1-score   support

           0       0.96      0.96      0.96        71
           1       0.93      0.93      0.93        43

    accuracy                           0.95       114
   macro avg       0.94      0.94      0.94       114
weighted avg       0.95      0.95      0.95       114


=== SVM ===
Accuracy: 0.956140350877193
Confusion Matrix:
 [[68  3]
 [ 2 41]]
Classification Report:
               precision    recall  f1-score   support

           0       0.97      0.96      0.96        71
           1       0.93      0.95      0.94        43

    accuracy                           0.96       114
   macro avg       0.95      0.96      0.95       114
weighted avg       0.96      0.96      0.96       114


=== Bagging ===
Accuracy: 0.956140350877193
Confusion Matrix:
 [[69  2]
 [ 3 40]]
Classification Report:
               precision    recall  f1-score   support

           0       0.96      0.97      0.97        71
           1       0.95      0.93      0.94        43

    accuracy                           0.96       114
   macro avg       0.96      0.95      0.95       114
weighted avg       0.96      0.96      0.96       114



===== Summary Comparison =====
              Model  Accuracy  Precision   Recall  F1-Score
Logistic Regression  0.973684   0.973719 0.973684  0.973621
      Random Forest  0.964912   0.965205 0.964912  0.964738
                SVM  0.956140   0.956488 0.956140  0.956237
            Bagging  0.956140   0.956088 0.956140  0.956036
      Decision Tree  0.947368   0.947368 0.947368  0.947368
                KNN  0.947368   0.947368 0.947368  0.947368


