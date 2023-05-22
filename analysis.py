import pandas as pd

df = pd.read_csv('titanic.csv')

print(df.isnull().sum())

df = df.dropna()

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

print(df[['Age', 'Fare']].describe())

import matplotlib.pyplot as plt

plt.hist(df['Age'])
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

import numpy as np

# Load data
df = pd.read_csv('titanic.csv')

# Compute mean, median, and mode of numeric variables
print('Mean age:', np.mean(df['Age']))
print('Median age:', np.median(df['Age']))
print('Mode of Pclass:', df['Pclass'].mode().iloc[0])

# Compute proportion of passengers who survived
survived = df['Survived'] == 1
print('Proportion survived:', np.mean(survived))

# Compute survival rates by gender
gender_survival = df.groupby('Sex')['Survived'].mean()
print('Survival rates by gender:')
print(gender_survival)

print('Summary statistics for numeric variables:')
print(df.describe())

# Compute mean, median, mode, and standard deviation of numeric variables
print('\nMean age:', np.mean(df['Age']))
print('Median age:', np.median(df['Age']))
print('Mode of Pclass:', df['Pclass'].mode().iloc[0])
print('Standard deviation of Fare:', np.std(df['Fare']))

# Compute proportion of passengers who survived
survived = df['Survived'] == 1
print('\nProportion survived:', np.mean(survived))

# Compute survival rates by gender
gender_survival = df.groupby('Sex')['Survived'].mean()
print('\nSurvival rates by gender:')
print(gender_survival)

# Compute range of Fare variable
print('\nRange of Fare:', max(df['Fare']) - min(df['Fare']))


import scipy.stats as stats

# Load the data
df = pd.read_csv('titanic.csv')

# Drop columns not needed for analysis
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Drop missing values
df = df.dropna()

# Convert categorical variables to numeric
df['Sex'] = df['Sex'].apply(lambda x: 0 if x == 'female' else 1)
df['Embarked'] = df['Embarked'].apply(lambda x: 0 if x == 'S' else (1 if x == 'C' else 2))

# Divide dataset into survived and not survived groups
survived = df[df['Survived'] == 1]
not_survived = df[df['Survived'] == 0]

# Calculate mean age for both groups
survived_mean_age = np.mean(survived['Age'])
not_survived_mean_age = np.mean(not_survived['Age'])

# Calculate two-sample t-test for difference in mean age
t_statistic, p_value = stats.ttest_ind(survived['Age'], not_survived['Age'], equal_var=False)

print('Mean age of those who survived:', survived_mean_age)
print('Mean age of those who did not survive:', not_survived_mean_age)
print('T-statistic:', t_statistic)
print('P-value:', p_value)


# df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Drop missing values
df = df.dropna()
import seaborn as sns

# Convert categorical variables to numeric
df['Sex'] = df['Sex'].apply(lambda x: 0 if x == 'female' else 1)
df['Embarked'] = df['Embarked'].apply(lambda x: 0 if x == 'S' else (1 if x == 'C' else 2))

# Visualize survival by gender and class
sns.catplot(x='Sex', y='Survived', hue='Pclass', kind='bar', data=df)
plt.title('Survival Rates by Gender and Class')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.show()

# Visualize distribution of age
sns.histplot(data=df, x='Age', hue='Survived', kde=True, bins=20)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Visualize correlation matrix
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True)
plt.title('Correlation Matrix')
plt.show()