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

### 📊 Model Performance Comparison

| Model               | Accuracy | Precision | Recall  | F1-Score |
|---------------------|----------|-----------|---------|----------|
| Logistic Regression | 0.9737   | 0.9737    | 0.9737  | 0.9736   |
| Random Forest       | 0.9649   | 0.9652    | 0.9649  | 0.9647   |
| SVM                 | 0.9561   | 0.9565    | 0.9561  | 0.9562   |
| Bagging             | 0.9561   | 0.9561    | 0.9561  | 0.9560   |
| Decision Tree       | 0.9474   | 0.9474    | 0.9474  | 0.9474   |
| KNN                 | 0.9474   | 0.9474    | 0.9474  | 0.9474   |

