# SMS-CLASSIFIER
# THIS PROJECT IS DONE BY USING VS CCODE WITH PYTHON EXTENTIONS (NUMPY & PANDAS)
# importing the required lib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



# Load the dataset
data = pd.read_csv("spam.csv", encoding='latin-1')

# Drop unnecessary columns and rename the remaining columns
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v1": "label", "v2": "text"})

# Display first few rows of the dataset
print(data.head())
# Convert labels to binary (0 for ham, 1 for spam)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})



# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
# Convert text data into TF-IDF vectors
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# Initialize SVM classifier
svm_classifier = SVC(kernel='linear')



# Train the classifier
svm_classifier.fit(X_train_tfidf, y_train)
# Predictions on the testing set
y_pred = svm_classifier.predict(X_test_tfidf)



# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)



# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))



# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt='d', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

