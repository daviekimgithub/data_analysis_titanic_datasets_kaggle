import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
df = pd.read_csv('titanic.csv')

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
