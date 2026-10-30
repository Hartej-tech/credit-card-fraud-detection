import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,roc_auc_score,roc_curve, accuracy_score

# Step 1: Load dataset
data = pd.read_csv("creditcard.csv")

# Step 2: Explore dataset
print("Dataset shape:", data.shape)
print("Fraud cases:", sum(data['Class'] == 1))
print("Normal cases:", sum(data['Class'] == 0))

#Print first 5 elements of the dataset
print(data.head())

#Print last 5 elements of the dataset
print(data.tail())

#Print all the null values of the dataset 
print(data.isnull())
print(data.isnull().sum())

#Print all information about the dataset
print(data.info())

#this will give a detail description of the dataset
print(data.describe())


#-------------------------------------
# Step 3: Visualize class distribution
#-------------------------------------
plt.figure(figsize=(6,4))
data['Class'].value_counts().plot(kind='bar', color=['green','red'])
plt.title("Class Distribution (0 = Normal, 1 = Fraud)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(12,12))
sns.heatmap(data.corr(),cmap="magma")
plt.title("Correlation heatmap")
plt.show()

#**************************
# Step 4: Feature scaling
#**************************
X = data.drop('Class', axis=1)
y = data['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Step 6: Model training (Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 7: Predictions
y_pred = model.predict(X_test)


# Step 8: Evaluation
print("Classification Report: " )

cm = confusion_matrix(y_test, y_pred)
print(cm)
plt.figure(figsize=(6,4))
sns.heatmap(cm,annot=True,cmap="Blues",fmt="d",xticklabels=["Normal","Fraud"],yticklabels=["Normal","Fraud"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Step 9: Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")


# Step 10: ROC curve
y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
print(f"ROC AUC Score: {auc_score:.4f}")

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, color="darkorange", label=f"ROC curve (AUC = {auc_score:.4f})")
plt.plot([0,1], [0,1], color="navy", linestyle="--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()