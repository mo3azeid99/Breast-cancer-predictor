# Breast Cancer Prediction from CSV Dataset

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Classification Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# === 1. Load Dataset from CSV ===
# Replace with your actual path
df = pd.read_csv("Dataset.csv")  # ← غيّر اسم الملف حسب اسم ملفك

# OPTIONAL: Show basic info
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# === 2. Preprocessing ===
# Assume target column is named 'diagnosis' or something like that
# Modify below according to your dataset
target_col = 'diagnosis'  # ← غير الاسم حسب اسم العمود عندك

# Convert target to 0/1 if needed (example: M = malignant → 1, B = benign → 0)
df[target_col] = df[target_col].map({'M': 1, 'B': 0})

# Drop unnecessary columns if any (like ID or unnamed)
df = df.drop(columns=[col for col in df.columns if "Unnamed" in col or col.lower() == "id"], errors='ignore')

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# === 3. Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4. Feature scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 5. Models ===
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(kernel='linear', random_state=42),
    "Bagging": BaggingClassifier(random_state=42)
}

results = []

for name, clf in classifiers.items():
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1-Score": report["weighted avg"]["f1-score"]
    })

    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_test, y_pred))

# === 6. Summary table ===
summary_df = pd.DataFrame(results)
print("\n\n===== Summary Comparison =====")
print(summary_df.sort_values(by="Accuracy", ascending=False).to_string(index=False))
