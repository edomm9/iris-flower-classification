import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load the Iris dataset
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['species'] = df['species'].map(species_map)

# Prepare training and test data
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the k-NN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Streamlit App Layout
st.title('🌸 Iris Flower Classifier')
st.write('Predict the species of an Iris flower based on its measurements.')

# Input sliders
sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.1)
sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.5)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 1.4)
petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 0.2)

# Make prediction
sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(sample)
species = species_map[prediction[0]]

st.subheader('🌼 Prediction Result')
st.success(f'This flower is predicted to be **{species.capitalize()}**.')

# Add data visualizations
st.subheader('📊 Species Distribution')

# Petal Length vs Petal Width
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='species', ax=ax1, s=60, alpha=0.8, edgecolor='k')
ax1.scatter(petal_length, petal_width, color='red', marker='*', s=200, label='Your Flower')
plt.title('Petal Length vs Petal Width', fontsize=16)
plt.legend()
plt.xlabel('Petal Length (cm)', fontsize=14)
plt.ylabel('Petal Width (cm)', fontsize=14)
st.pyplot(fig1)

# Sepal Length vs Sepal Width
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='species', ax=ax2, s=60, alpha=0.8, edgecolor='k')
ax2.scatter(sepal_length, sepal_width, color='red', marker='*', s=200, label='Your Flower')
plt.title('Sepal Length vs Sepal Width', fontsize=16)
plt.legend()
plt.xlabel('Sepal Length (cm)', fontsize=14)
plt.ylabel('Sepal Width (cm)', fontsize=14)
st.pyplot(fig2)
