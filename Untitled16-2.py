#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# In[2]:


data= pd.read_csv("credit.csv")


# In[3]:


data.head()


# In[31]:


population_count=len(data["Age"])
print(population_count)


# In[9]:


df = pd.DataFrame(data)

ages = df['Age']

age_counts = Counter(ages)

total_count = len(ages)
percentages = {age: (count / total_count) * 100 for age, count in age_counts.items()}

sorted_ages = sorted(percentages.keys())
percentage_values = [percentages[age] for age in sorted_ages]

plt.plot(sorted_ages, percentage_values)

plt.xlabel('Age')
plt.ylabel('Percentage of People (%)')
plt.title('Percentage of People by Age')

plt.show()


# In[27]:


purpose = df['Purpose']
fig, ax = plt.subplots()
purpose_counts = Counter(purpose)
total_count = len(purpose)
percentages = {purpose: (count / total_count) * 100 for purpose, count in purpose_counts.items()}
sorted_purpose = sorted(percentages.keys())
percentage_values = [percentages[purpose] for purpose in sorted_purpose]

ax.bar(sorted_purpose, percentage_values)


ax.set_ylabel('Percentage of use for loan')
ax.legend(title='What are loans being used for?')
plt.xticks(rotation=25)
plt.show()


# In[16]:


gender = df['Sex']
fig, ax = plt.subplots()
gender_counts = Counter(gender)
total_count = len(gender)
percentages = {gender: (count / total_count) * 100 for gender, count in gender_counts.items()}
sorted_gender = sorted(percentages.keys())
percentage_values = [percentages[gender] for gender in sorted_gender]

ax.bar(sorted_gender, percentage_values)


ax.set_ylabel('Percentage of People (%)')
ax.legend(title='Percentage of People by Gender')

plt.show()


# In[32]:


import seaborn as sns


gender_purpose_counts = df.groupby(['Sex', 'Purpose']).size().reset_index(name='counts')
total_counts = df.groupby('Sex').size().reset_index(name='total')
gender_purpose_counts = pd.merge(gender_purpose_counts, total_counts, on='Sex')
gender_purpose_counts['percentage'] = (gender_purpose_counts['counts'] / gender_purpose_counts['total']) * 100

sns.set(style="whitegrid")
g = sns.catplot(
    data=gender_purpose_counts, 
    x='Sex', 
    y='percentage', 
    hue='Purpose', 
    kind='bar', 
    height=6, 
    aspect=1.5
)

g.set_axis_labels("Gender", "Percentage of People (%)")
g.legend.set_title("Purpose")

plt.show()


# In[41]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



X = df.drop('Risk', axis=1)
y = df['Risk']

# Convert Risk to binary (good = 1, bad = 0)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Handle missing values and encode categorical features
numeric_features = ['Age', 'Job', 'Duration','Credit amount']
categorical_features = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']

numeric_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Step 2: Model Selection and Training

# Create a pipeline with preprocessor and model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(random_state=42))])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Step 3: Model Evaluation
# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[ ]:




