import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# Load the Titanic dataset
df = pd.read_csv('titanic.csv')

# Clean the data
df = df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1)
df = df.dropna()

# Create a Kaplan-Meier estimator and fit the data
kmf = KaplanMeierFitter()
kmf.fit(df['Age'], event_observed=df['Survived'])

# # Plot the survival curve for all passengers
# plt.figure(figsize=(10, 5))
# kmf.plot()
# plt.title('Kaplan-Meier Survival Curve - All Passengers')
# plt.xlabel('Age')
# plt.ylabel('Survival Probability')
# plt.show()

# # Plot the survival curves for different groups
# groups = df.groupby(['Sex', 'Pclass'])
# plt.figure(figsize=(10, 5))
# for name, group in groups:
#     kmf.fit(group['Age'], event_observed=group['Survived'])
#     kmf.plot(label=name)
# plt.title('Kaplan-Meier Survival Curve - by Sex and Passenger Class')
# plt.xlabel('Age')
# plt.ylabel('Survival Probability')
# plt.legend()
# plt.show()
 

import pandas as pd

# Load the dataset
# df = pd.read_csv('titanic.csv')

# Analyze survival rates based on sex
sex_survival = df.groupby('Sex')['Survived'].mean()
print('Survival rates by sex:\n', sex_survival)

# Analyze survival rates based on passenger class
class_survival = df.groupby('Pclass')['Survived'].mean()
print('\nSurvival rates by passenger class:\n', class_survival)

# Analyze survival rates based on age
age_survival = pd.cut(df['Age'], [0, 18, 30, 50, 80])
age_survival = df.groupby(age_survival)['Survived'].mean()
print('\nSurvival rates by age group:\n', age_survival)

# Calculate summary statistics for age and fare
age_stats = df['Age'].describe()
fare_stats = df['Fare'].describe()

print('\nSummary statistics for Age:\n', age_stats)
print('\nSummary statistics for Fare:\n', fare_stats)




import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load Titanic dataset
titanic = pd.read_csv('titanic.csv')

# Create subplots for survival analysis visualizations
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))

# Plot survival rate by sex
sns.barplot(x='Sex', y='Survived', data=titanic, ax=axs[0,0])
axs[0,0].set_title('Survival Rate by Sex')

# Plot survival rate by passenger class
sns.barplot(x='Pclass', y='Survived', data=titanic, ax=axs[0,1])
axs[0,1].set_title('Survival Rate by Passenger Class')

# Plot survival rate by age group
titanic['AgeGroup'] = pd.cut(titanic['Age'], bins=[0, 18, 35, 50, 100])
sns.barplot(x='AgeGroup', y='Survived', data=titanic, ax=axs[1,0])
axs[1,0].set_title('Survival Rate by Age Group')

# Plot survival rate by number of siblings/spouses
sns.barplot(x='SibSp', y='Survived', data=titanic, ax=axs[1,1])
axs[1,1].set_title('Survival Rate by Number of Siblings/Spouses')

# Remove AgeGroup column
titanic.drop('AgeGroup', axis=1, inplace=True)

# Create subplots for descriptive statistics visualizations
fig, axs = plt.subplots(ncols=2, figsize=(12, 4))

# Plot distribution of ages
sns.histplot(x='Age', data=titanic, kde=False, ax=axs[0])
axs[0].set_title('Distribution of Ages')

# Plot distribution of fares
sns.histplot(x='Fare', data=titanic, kde=False, ax=axs[1])
axs[1].set_title('Distribution of Fares')

# Show the plots
plt.show()

# Calculate correlation matrix
corr_matrix = df.corr(numeric_only=True)

# Plot the correlation matrix as a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

import statsmodels.api as sm


# Select relevant columns
X = df[['Pclass']]
y = df['Fare']

# Add constant to predictor variable
X = sm.add_constant(X)

# Fit regression model
model = sm.OLS(y, X).fit()

# Print model summary
print(model.summary())

# Make prediction for first observation
prediction = model.predict([1, 2])
print("Predicted fare:", prediction[0])



# Create a pie chart showing the proportion of passengers by sex
sex_counts = df['Sex'].value_counts()
plt.pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%')
plt.title('Passengers by Sex')
plt.show()

# Create a bar chart showing the survival rate by passenger class
class_survival = df.groupby('Pclass')['Survived'].mean()
plt.bar(class_survival.index, class_survival)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()



from scipy.stats import ttest_ind

# Load Titanic dataset
df = pd.read_csv('titanic.csv')

# Filter dataset to only include male and female passengers
male_df = df[df['Sex'] == 'male']
female_df = df[df['Sex'] == 'female']

# Calculate the survival rate for male and female passengers
male_survival_rate = np.mean(male_df['Survived'])
female_survival_rate = np.mean(female_df['Survived'])

# Perform t-test to determine if there is a statistically significant difference
t_stat, p_val = ttest_ind(male_df['Survived'], female_df['Survived'], equal_var=False)

# Print the survival rate for male and female passengers
print("Male survival rate: ", male_survival_rate)
print("Female survival rate: ", female_survival_rate)

# Print the p-value to determine if there is a statistically significant difference
if p_val < 0.05:
    print("There is a statistically significant difference in survival rate between male and female passengers.")
else:
    print("There is no statistically significant difference in survival rate between male and female passengers.")