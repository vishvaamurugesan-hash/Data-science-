# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import os

# Step 1: Load Dataset (use script directory so relative paths work)
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'online_learning_data.csv')
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found at {csv_path}. Place 'online_learning_data.csv' next to this script.")

df = pd.read_csv(csv_path)

# Step 2: Preprocessing
# Clean column names
# use regex=False to avoid pandas future warning when replacing literal strings
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_', regex=False)
print("Columns in dataset:", df.columns.tolist())

# Handle missing values
df = df.dropna()

# If your dataset doesn’t have 'internet_access', we’ll use 'internet_type' or skip it safely
if 'internet_access' in df.columns:
    df['internet_access'] = df['internet_access'].map({'yes': 1, 'no': 0})
elif 'internet_type' in df.columns:
    # Convert categorical internet type (Wi-Fi, Mobile Data, etc.) to numeric codes
    df['internet_access'] = df['internet_type'].astype('category').cat.codes
    print("✅ Using 'internet_type' instead of 'internet_access'")
else:
    # If no related column exists, create a dummy one to prevent errors
    df['internet_access'] = 1
    print("⚠️ No 'internet_access' or 'internet_type' found — using dummy value.")

# If 'final_grade' not present, create a simulated target
if 'final_grade' not in df.columns:
    np.random.seed(42)
    df['final_grade'] = np.random.randint(50, 100, size=len(df))
    print("⚠️ 'final_grade' not found — generated random grades for demo.")

# If 'hours_studied' not present, create one
if 'hours_studied' not in df.columns:
    np.random.seed(42)
    df['hours_studied'] = np.random.randint(1, 10, size=len(df))
    print("⚠️ 'hours_studied' not found — generated random values for demo.")

# If 'attended_online_classes' not present, simulate it
if 'attended_online_classes' not in df.columns:
    np.random.seed(42)
    df['attended_online_classes'] = np.random.choice([0, 1], size=len(df))
    print("⚠️ 'attended_online_classes' not found — generated random yes/no data.")

# Step 3: Exploratory Data Analysis
plt.figure(figsize=(8, 6))
# Compute correlation only for numeric columns to avoid warnings in newer pandas/seaborn
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(df['final_grade'], kde=True, color='green')
plt.title("Distribution of Final Grades")
plt.xlabel("Final Grade")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x='attended_online_classes', y='final_grade', data=df)
plt.title("Final Grades by Online Class Attendance")
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

# Step 4: Modeling
features = ['hours_studied', 'internet_access', 'attended_online_classes']
X = df[features]
y = df['final_grade']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Step 5: Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics")
print("------------------------")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 6: Plot Predictions vs Actual
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")
plt.title("Actual vs Predicted Final Grades")
plt.grid(True)
plt.show()
