import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Print the name of the script
print("Titanic:", os.path.basename(__file__))

# Load the Titanic dataset
df = sns.load_dataset('titanic')

# Display the first few rows of the dataset
print(df.head())

# Check the dimensions of the dataset
print(df.shape)

# Get summary statistics of numerical variables
print(df.describe())

# Check the data types of variables
print(df.dtypes)

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values
df.dropna(inplace=True)

# Bar plot
sns.countplot(x='survived', data=df)
plt.xlabel('Survival Status')
plt.ylabel('Count')
plt.title('Survival Count')
plt.show()

# Histogram
plt.hist(df['age'], bins=10)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age')
plt.show()

# Scatter plot
plt.scatter(df['age'], df['fare'])
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Age vs. Fare')
plt.show()

# Box plot
sns.boxplot(x=df['survived'], y=df['fare'])
plt.xlabel('Survival Status')
plt.ylabel('Fare')
plt.title('Survival Status vs. Fare')
plt.show()

# Correlation analysis
correlation = df[['age', 'fare']].corr()
print(correlation)

# Cross-tabulation
cross_tab = pd.crosstab(df['pclass'], df['survived'])
print(cross_tab)
