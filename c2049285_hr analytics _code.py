#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer


# In[2]:


# Load the data
data = pd.read_csv('HR CSV DATA SET.csv')
data.head()


# In[3]:


data.shape


# In[4]:


# find a missing value
data.isnull().sum()


# In[5]:


#handling a missing value
data.dropna(inplace=True)
missing_values = data.isnull().sum()
print(data.isnull().sum())


# In[6]:


# handling missing values with median
data['YearsWithCurrManager'].fillna(data['YearsWithCurrManager'].median(),inplace=True)
print(data.isnull().sum())


# In[7]:


# Data Cleaning
def clean_data(data):
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    data_numeric = data.select_dtypes(include=[np.number])
    data[data_numeric.columns] = imputer.fit_transform(data_numeric)
    
    # Convert categorical variables to numerical
    le = LabelEncoder()
    categorical_cols = ['Attrition', 'SalarySlab', 'AgeGroup', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
    
    # Drop unnecessary columns
    data = data.drop(['EmpID', 'EmployeeCount', 'EmployeeNumber', 'StandardHours'], axis=1)
    
    return data

cleaned_data = clean_data(data)


# In[20]:


cleaned_data.columns


# In[8]:


# Data Visualization
def visualize_data(data):
    # Correlation heatmap
    plt.figure(figsize=(20, 16))
    sns.heatmap(data.corr(), annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

    # Distribution of target variable (Attrition)
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Attrition', data=data)
    plt.title('Distribution of Attrition')
    plt.show()

    # Age distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Age'], kde=True)
    plt.title('Age Distribution')
    plt.show()

    # Job satisfaction by department
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Department', y='JobSatisfaction', data=data)
    plt.title('Job Satisfaction by Department')
    plt.show()

visualize_data(cleaned_data)


# In[9]:


# Prepare data for modeling
def prepare_data(df):
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

X, y = prepare_data(cleaned_data)


# In[10]:


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


# Model training and evaluation
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"{model_name} Results:")
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')


    return model, accuracy


# In[16]:


# Random Forest
rf_model, rf_accuracy = train_and_evaluate_model(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_train, X_test, y_train, y_test, "Random Forest"
)


# In[12]:


# Gradient Boosting
gb_model, gb_accuracy = train_and_evaluate_model(
    GradientBoostingClassifier(n_estimators=100, random_state=42),
    X_train, X_test, y_train, y_test, "Gradient Boosting"
)


# In[14]:


# Neural Network
nn_model, nn_accuracy = train_and_evaluate_model(
    MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
    X_train, X_test, y_train, y_test, "Neural Network"
)


# In[17]:


# Compare model performances
models = ['Random Forest', 'Gradient Boosting', 'Neural Network']
accuracies = [rf_accuracy, gb_accuracy, nn_accuracy]

plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accuracies)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()


# In[18]:


# Feature importance (for Random Forest)
feature_importance = pd.DataFrame({
    'feature': cleaned_data.drop('Attrition', axis=1).columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
plt.title('Top 15 Feature Importances (Random Forest)')
plt.tight_layout()
plt.show()


# In[19]:


joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(gb_model, 'gradient_boosting_model.pkl')
joblib.dump(nn_model, 'neural_network_model.pkl')


# In[ ]:


#creation of GUI


# In[20]:


import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Load the trained model (you can change the model filename as needed)
rf_model = joblib.load('random_forest_model.pkl')

# Define the features required for prediction
features = [
    'Age', 'AgeGroup', 'BusinessTravel', 'DailyRate',
       'Department', 'DistanceFromHome', 'Education', 'EducationField',
       'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement',
       'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
       'MonthlyIncome', 'SalarySlab', 'MonthlyRate', 'NumCompaniesWorked',
       'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
       'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
       'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
       'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager'
]

# Function to make a prediction
def make_prediction():
    try:
        # Collect input data
        input_data = [float(entries[feature].get()) for feature in features]

        # Reshape the input data
        input_data = np.array(input_data).reshape(1, -1)

        # Predict using the loaded model
        prediction = rf_model.predict(input_data)

        # Display the result
        result = "Attrition" if prediction[0] == 1 else "No Attrition"
        messagebox.showinfo("Prediction Result", f"Prediction: {result}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Initialize the GUI window
root = tk.Tk()
root.title("Attrition Prediction")
root.geometry("800x600")

# Create and place input fields in 3-column format
entries = {}
num_columns = 3
for i, feature in enumerate(features):
    col = i % num_columns  # Determine the column index
    row = i // num_columns  # Determine the row index
    
    label = ttk.Label(root, text=feature)
    label.grid(row=row, column=col*2, padx=10, pady=5, sticky='W')  # Adjust column index for spacing
    entry = ttk.Entry(root)
    entry.grid(row=row, column=col*2 + 1, padx=10, pady=5)
    entries[feature] = entry

# Create and place the Predict button
predict_button = ttk.Button(root, text="Predict", command=make_prediction)
predict_button.grid(row=(len(features) // num_columns) + 1, columnspan=num_columns*2, pady=20)

# Run the GUI event loop
root.mainloop()


# 
