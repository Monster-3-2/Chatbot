import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, classification_report
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

print(df.head())
print(df.describe())
print(df['species'].value_counts())  # Confirm 50 samples each (balanced)
X = iris.data   # Features (4 columns)
y = iris.target  # Labels (0, 1, 2)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Result: Mean=0, Variance=1 for each feature
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,      # 80% train, 20% test
    random_state=42,
    shuffle=True        # Remove order bias
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
error_rates = []
k_range = range(1, 31)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    error_rates.append(1 - f1_score(y_test, preds, average='weighted'))

plt.plot(k_range, error_rates, marker='o')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.title('Elbow Method: Choosing K')
plt.show()

best_k = k_range[error_rates.index(min(error_rates))]
print(f"Optimal K: {best_k}")
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)       # FIT: memorize the training map
predictions = model.predict(X_test)  # PREDICT: apply logic to test set
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
f1 = f1_score(y_test, predictions, average='weighted')
print(f"F1 Score: {f1:.4f}")
print(classification_report(y_test, predictions, target_names=iris.target_names))
# Example: predict a new flower
new_sample = [[5.1, 3.5, 1.4, 0.2]]  # Raw measurements
new_sample_scaled = scaler.transform(new_sample)
prediction = model.predict(new_sample_scaled)
print(f"Predicted species: {iris.target_names[prediction[0]]}")

