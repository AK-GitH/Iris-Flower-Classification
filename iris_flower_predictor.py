import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

# Load cleaned data
df = pd.read_csv("cleaned_iris.csv")

# Encode species labels
df["Species"] = df["Species"].astype("category")
df["Species_Code"] = df["Species"].cat.codes  # Keep original for visuals

# Split into features and target
X = df.drop(columns=["Species", "Species_Code"])  # Use Species_Code as y
y = df["Species_Code"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Print model performance
print("Random Forest")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf, average='macro'))
print("Recall:", recall_score(y_test, y_pred_rf, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred_rf, average='macro'))

# --- Detailed per-class evaluation ---
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=df["Species"].cat.categories))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='viridis', fmt='d', linewidths=0.5, linecolor='gray',
            xticklabels=df["Species"].cat.categories,
            yticklabels=df["Species"].cat.categories)
plt.title("Random Forest Confusion Matrix", fontsize=14, fontweight='bold')
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.grid(False)
plt.show()

# --- Feature Importances ---
importances = rf.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 6))
sns.barplot(x=importances, y=feature_names, palette='coolwarm')
plt.title("Feature Importances (Random Forest)", fontsize=14, fontweight='bold')
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

joblib.dump(rf, "random_forest_iris.joblib")  # Saves the model to a file
