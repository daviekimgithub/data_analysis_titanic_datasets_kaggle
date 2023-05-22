# import the neccessary packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns


# Read csv file
df = pd.read_csv('df.csv')
# df = df.dropna()


# Mean, median, mode, standard deviation, and range of the df csv file
# Compute mean, median, and mode of numeric variables
df_age = pd.notnull((df['Age']))
print('Mean age:', np.mean(df_age))
print('Median age:', np.median(df_age))
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
print('\n\n\nSurvival rates by gender:')
print(gender_survival)

# Compute range of Fare variable
print('\nRange of Fare:', max(df['Fare']) - min(df['Fare']))



# Convert categorical variables to numeric
df['Sex'] = df['Sex'].apply(lambda x: 0 if x == 'female' else 1)
df['Embarked'] = df['Embarked'].apply(lambda x: 0 if x == 'S' else (1 if x == 'C' else 2))

# Create subplots for survival analysis visualizations
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))

# Plot survival rate by sex
sns.barplot(x='Sex', y='Survived', data=df, ax=axs[0,0])
axs[0,0].set_title('Survival Rate by Sex')

# Plot survival rate by passenger class
sns.barplot(x='Pclass', y='Survived', data=df, ax=axs[0,1])
axs[0,1].set_title('Survival Rate by Passenger Class')

# Plot survival rate by age group
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 100])
sns.barplot(x='AgeGroup', y='Survived', data=df, ax=axs[1,0])
axs[1,0].set_title('Survival Rate by Age Group')

# Plot survival rate by number of siblings/spouses
sns.barplot(x='SibSp', y='Survived', data=df, ax=axs[1,1])
axs[1,1].set_title('Survival Rate by Number of Siblings/Spouses')

# Remove AgeGroup column
df.drop('AgeGroup', axis=1, inplace=True)

# Create subplots for descriptive statistics visualizations
fig, axs = plt.subplots(ncols=2, figsize=(12, 4))

# Plot distribution of ages
sns.histplot(x='Age', data=df, kde=False, ax=axs[0])
axs[0].set_title('Distribution of Ages')

# Plot distribution of fares
sns.histplot(x='Fare', data=df, kde=False, ax=axs[1])
axs[1].set_title('Distribution of Fares')

# Show the plots
plt.show()

# Visualization 1: Countplot of passenger class
sns.countplot(x='Pclass', data=df)
plt.title('Passenger Class Distribution')
plt.show()

# Visualization 2: Countplot of survival by passenger class
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title('Survival by Passenger Class')
plt.show()

# Visualization 3: Pie chart of survival rates
survival_counts = df['Survived'].value_counts()
plt.pie(survival_counts, labels=survival_counts.index, autopct='%1.1f%%')
plt.title('Survival Rate')
plt.show()

# Visualization 4: Boxplot of age by passenger class
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title('Age Distribution by Passenger Class')
plt.show()

# Visualization 5: Scatterplot of fare by age
sns.scatterplot(x='Age', y='Fare', data=df)
plt.title('Fare vs. Age')
plt.show()

# Visualization 6: Histogram of age distribution
sns.histplot(df, x='Age', kde=True)
plt.title('Age Distribution')
plt.show()

# Visualization 7: Bar chart of survival by sex
sns.catplot(x='Sex', hue='Survived', kind='count', data=df)
plt.title('Survival by Sex')
plt.show()

# Visualization 8: Stacked bar chart of survival by sex and passenger class
df_survived = df[df['Survived'] == 1].groupby(['Pclass', 'Sex'])['Survived'].count().reset_index()
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df_survived)
plt.title('Survival by Passenger Class and Sex')
plt.show()

# Visualization 9: Line chart of age distribution by passenger class
sns.lineplot(x='Age', y='Pclass', data=df)
plt.title('Age Distribution by Passenger Class')
plt.show()

# Visualization 10: Violin plot of age distribution by survival
sns.violinplot(x='Survived', y='Age', data=df)
plt.title('Age Distribution by Survival')
plt.show()


# Visualize distribution of age
sns.histplot(data=df, x='Age', hue='Survived', kde=True, bins=20)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# # Visualize correlation matrix
# corr_matrix = df.corr()
# sns.heatmap(corr_matrix, annot=True)
# plt.title('Correlation Matrix')
# plt.show()