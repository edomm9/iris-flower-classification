import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Load and Explore Dataset
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Print first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Basic statistics
print("\nDataset Description:")
print(df.describe())

# 3. Visualize Data
# Pairplot
sns.pairplot(df, hue='species')
plt.title('Pairplot of Iris Dataset')
plt.show()

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title('Feature Correlation Heatmap')
plt.show()

# 4. Prepare Data for Modeling
X = iris.data
y = iris.target

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 5. Train the Model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 6. Evaluate the Model
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Confusion Matrix Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Iris Classification')
plt.show()


# 7. Conclusion
print("\nConclusion:")
print("The k-NN classifier achieved high accuracy on the Iris dataset.")
print("Model could be improved further by tuning 'k' or trying different algorithms.")

# ---- Predict from User Input ----

# Scatter plot function
def plot_user_input(sepal_length, sepal_width, petal_length, petal_width):
    # Scatter plot for existing data
    plt.figure(figsize=(8,6))

    # Plot each Iris class
    for i, species in enumerate(iris.target_names):
        plt.scatter(
            X[y==i, 2],  # petal length
            X[y==i, 3],  # petal width
            label=species
        )

    # Plot user input
    plt.scatter(
        petal_length, 
        petal_width, 
        color='red', 
        marker='*', 
        s=200, 
        label='Your Flower'
    )

    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.title('Iris Flowers vs Your Input')
    plt.legend()
    plt.grid(True)
    plt.savefig('user_input_plot.png')
    plt.show()

# Prediction function
def predict_iris():
    print("\n🌸 Enter flower measurements to predict Iris species:")

    try:
        sepal_length = float(input("Enter sepal length (cm): "))
        sepal_width = float(input("Enter sepal width (cm): "))
        petal_length = float(input("Enter petal length (cm): "))
        petal_width = float(input("Enter petal width (cm): "))

        # Basic range checking (based on real iris measurements)
        if not (4.0 <= sepal_length <= 8.0 and
                2.0 <= sepal_width <= 4.5 and
                1.0 <= petal_length <= 7.0 and
                0.1 <= petal_width <= 2.5):
            print("\n❌ Invalid input: Measurements out of Iris flower range!")
            return

        sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(sample)
        species = iris.target_names[prediction[0]]

        print(f"\n✅ Predicted Iris species: {species.capitalize()}")

        # Plot after successful prediction
        plot_user_input(sepal_length, sepal_width, petal_length, petal_width)

    except ValueError:
        print("\n❌ Invalid input: Please enter valid numbers.")

# Call the function
predict_iris()
