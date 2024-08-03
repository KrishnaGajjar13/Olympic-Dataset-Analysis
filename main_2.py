import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('athlete_events.csv')
rows, columns = df.shape
df = df.fillna({'Height': int(df['Height'].mean())})
df = df.fillna({'Age': int(df['Age'].mean())})
df = df.fillna({'Weight': int(df['Weight'].mean())})
Unique_df = df.drop_duplicates(subset=['ID'])

male_athletes = Unique_df[Unique_df['Sex'] == 'M']
female_athletes = Unique_df[Unique_df['Sex'] == 'F']


plt.figure(figsize=(27, 15))

# Question 8

plt.subplot(3, 3, 1)
plt.hist(Unique_df['Height'], bins=12, color='blue', alpha=0.4, label='Height(cm)')
plt.hist(Unique_df['Weight'], bins=12, color='green', alpha=0.4, label='Weight(kg)')
plt.title('Height and Weight Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend(loc='upper right')

# Question 9
plt.subplot(3, 3, 2)
age_bins = np.arange(min(male_athletes['Age']), max(male_athletes['Age']) + 1, 1)
f_age_bins = np.arange(min(female_athletes['Age']), max(female_athletes['Age']) + 1, 1)
plt.hist(male_athletes['Age'], bins=age_bins, color='blue', alpha=0.7, label='Age')
plt.title('Age-Male Distribution')
plt.xlabel('Male Athletes')
plt.ylabel('Count')
plt.legend()

plt.subplot(3, 3, 3)
plt.hist(male_athletes['Age'], bins=f_age_bins, color='green', alpha=0.7, label='Age')
plt.title('Age-Female Distribution')
plt.xlabel('Female Athletes')
plt.ylabel('Count')
plt.legend()

# Question 10
plt.subplot(3, 3, 4)
Team_count = Unique_df['Team'].value_counts()
Team_count = Team_count.nlargest(5)
Team_count.plot(kind='bar', color='skyblue', alpha=0.7)
plt.title('Country with most medals')
plt.xticks(rotation=0)
plt.xlabel('Country')
plt.ylabel('Number of Athletes')
plt.legend()

# Question 11
plt.subplot(3, 3, 5)
height = Unique_df['Height']
weight = Unique_df['Weight']
plt.scatter(height, weight, color='skyblue', marker='.', s=25,alpha=0.5, label='Athletes')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Height vs. Weight')
plt.legend()


# Question 12
plt.subplot(3, 3, 6)
medal_df = df[df['Medal'].notna()]
age_counts = medal_df['Age'].value_counts().sort_index()
plt.plot(age_counts.index, age_counts.values, marker='o', linestyle='-', color='brown', label='Age')
plt.xlabel('Age')
plt.ylabel('Medals')
plt.legend()

# Question 13
plt.subplot(3, 3, 7)
ath_df = Unique_df['Year'].value_counts().sort_index()
plt.plot(ath_df.index, ath_df.values, marker='.', linestyle='-', color='black', label='Participants')
plt.xlabel('Year')
plt.ylabel('Total Participation')
plt.legend()

# Question 14
plt.subplot(3, 3, 8)
Sport_count = df['Sport'].value_counts()
Sport_count = Sport_count.nlargest(5)
Sport_count.plot(kind='bar', color='red', alpha=1)
plt.title('Sport with most Athletes')
plt.xticks(rotation=0)
plt.xlabel('Sport')
plt.ylabel('Number of Athletes')
plt.legend()

# Question 15
print("Question 15")
plt.subplot(3, 3, 9)
City_count = df['City'].value_counts().sort_index()
print("Here is the list of cities hosted the event!")
print(City_count.to_string())
print("Let's see the graph of top 5 cities who hosted the most")
City_count = Sport_count.nlargest(5)
City_count.plot(kind='bar', color='blue', alpha=0.7)
plt.title('Sport with most Athletes')
plt.xticks(rotation=0)
plt.xlabel('City')
plt.ylabel('Number of times hosted')
plt.legend()


plt.tight_layout()
plt.show()
