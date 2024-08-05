import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('athlete_events.csv')
rows, columns = df.shape
Unique_df = df.drop_duplicates(subset=['ID'])
male_athletes = Unique_df[Unique_df['Sex'] == 'M']
female_athletes = Unique_df[Unique_df['Sex'] == 'F']

# Question 1
print("Question 1!")
print(f"There are {rows} rows and {columns} columns in the dataset.\n")

# Question 2
print("Question 2!")
Total_sports = df['Sport'].nunique()
print(f"There are {Total_sports} unique sports played.\n")

# Question 3
print("Question 3!")
tot_male = male_athletes['ID'].nunique()
tot_female = female_athletes['ID'].nunique()
print(f"There are total {tot_male} male and {tot_female} female athletes.\n")

# Question 4
print("Question 4!")
min_age = int(df['Age'].min())
max_age = int(df['Age'].max())
avg_age = int(df['Age'].mean())
print(f"Youngest athlete is : {min_age} years old.")
print(f"Eldest athlete is : {max_age} years old.")
print(f"Average age of an athlete is : {avg_age} years old.\n")

# Question 5
print("Question 5!")
tot_medal = df['Medal'].notna().sum()
print(f"Total Medals won are: {tot_medal}")
Gold_Winners = df[df['Medal'] == 'Gold']
Silver_Winners = df[df['Medal'] == 'Silver']
Bronze_Winners = df[df['Medal'] == 'Bronze']
print(f"There are total {Gold_Winners.shape[0]} Gold Medalists")
print(f"There are total {Silver_Winners.shape[0]} Silver Medalists")
print(f"There are total {Bronze_Winners.shape[0]} Bronze Medalists\n")

# Question 6
print("Question 6!")
Team_count = Unique_df['Team'].value_counts()
print(f"{Team_count.idxmax()} has the most athletes with total {Team_count.max()} athletes.\n")

# Question 7
print("Question 7!")
columns_with_missing = df.columns[df.isna().any()]  # Identify columns with missing values
missing_counts = df[columns_with_missing].isna().sum()  # Count missing values in each column
# Print columns with missing values and their counts
print("Columns with missing values:")
for column in columns_with_missing:
    count = missing_counts[column]
    print(f"{column}:\t{count}")

df.fillna({
    'Height': int(df['Height'].mean()),
    'Age': int(df['Age'].mean()),
    'Weight': int(df['Weight'].mean())
}, inplace=True)

# Figure 1
plt.figure(figsize=(18, 15))

# Question 8
plt.subplot(2, 2, 1)
plt.hist(Unique_df['Height'], bins=12, color='blue', alpha=0.4, label='Height(cm)')
plt.hist(Unique_df['Weight'], bins=12, color='green', alpha=0.4, label='Weight(kg)')
plt.title('Height and Weight Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend(loc='upper right')

# Question 9
plt.subplot(2, 2, 3)
age_bins = np.arange(min(male_athletes['Age']), max(male_athletes['Age']) + 1, 1)
plt.hist(male_athletes['Age'], bins=age_bins, color='blue', alpha=0.7, label='Age')
plt.title('Age-Male Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend()

plt.subplot(2, 2, 4)
f_age_bins = np.arange(min(female_athletes['Age']), max(female_athletes['Age']) + 1, 1)
plt.hist(female_athletes['Age'], bins=f_age_bins, color='green', alpha=0.7, label='Age')
plt.title('Age-Female Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend()

# Question 10
plt.subplot(2, 2, 2)
Team_count = Unique_df['Team'].value_counts()
Team_count = Team_count.nlargest(5)
Team_count.plot(kind='bar', color='skyblue', alpha=0.7)
plt.title('Country with most athletes')
plt.xticks(rotation=0)
plt.xlabel('Country')
plt.ylabel('Number of Athletes')
plt.legend()
plt.show(block=False)  # Non-blocking show

# Figure 2
plt.figure(figsize=(18, 15))

# Question 11
plt.subplot(2, 2, 1)
height = Unique_df['Height']
weight = Unique_df['Weight']
plt.scatter(height, weight, color='skyblue', marker='.', s=25, alpha=0.5, label='Athletes')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Height vs. Weight')
plt.legend()

# Question 12
plt.subplot(2, 2, 2)
medal_df = df[df['Medal'].notna()]
age_counts = medal_df['Age'].value_counts().sort_index()
plt.plot(age_counts.index, age_counts.values, marker='o', linestyle='-', color='brown', label='Age')
plt.xlabel('Age')
plt.ylabel('Medals')
plt.legend()

# Question 13
plt.subplot(2, 2, 3)
ath_df = Unique_df['Year'].value_counts().sort_index()
plt.plot(ath_df.index, ath_df.values, marker='.', linestyle='-', color='black', label='Participants')
plt.xlabel('Year')
plt.ylabel('Total Participation')
plt.legend()

# Question 14
plt.subplot(2, 2, 4)
Sport_count = df['Sport'].value_counts()
Sport_count = Sport_count.nlargest(5)
Sport_count.plot(kind='bar', color='red', alpha=1)
plt.title('Sport with most Athletes')
plt.xticks(rotation=0)
plt.xlabel('Sport')
plt.ylabel('Number of Athletes')
plt.legend()
plt.show(block=False)  # Non-blocking show

# Figure 3
plt.figure(figsize=(18, 15))

# Question 15
print("Question 15")
plt.subplot(2, 2, 1)
City_count = df['City'].value_counts().sort_index()
print("Here is the list of cities hosted the event!")
print(City_count.to_string())
print("Let's see the graph of top 5 cities who hosted the most")
City_count = City_count.nlargest(5)
City_count.plot(kind='bar', color='blue', alpha=0.7)
plt.title('Cities with most Events')
plt.xticks(rotation=0)
plt.xlabel('City')
plt.ylabel('Number of times hosted')
plt.legend()

# Question 16
print("Question 16!")
medal_count = df[df['Medal'].notna()]
sum_medal = medal_count[medal_count['Season'] == "Summer"]
win_medal = medal_count[medal_count['Season'] == "Winter"]
print(f"Total medals won in winter are {win_medal['ID'].count()} and in summer are {sum_medal['ID'].count()}.")

avg_height_weight = df.groupby('Year').agg({'Height': 'mean', 'Weight': 'mean'}).round(2)

# Question 17
print("Question 17")
grp_year = medal_count.groupby('Year')
for name, group in grp_year:
    print(f"In {name}, these are the top 5 teams who won the most medals!")
    group = group['Team'].value_counts()
    group = group.nlargest(5)
    print(group.to_string(), "\n")

# Question 18
print("Question 18!")
plt.subplot(2, 2, 2)
plt.bar(avg_height_weight.index, avg_height_weight['Height'], color='blue', alpha=0.7)
plt.title('Average Height Over Years')
plt.xlabel('Year')
plt.ylabel('Average Height')

plt.subplot(2, 2, 3)
plt.bar(avg_height_weight.index, avg_height_weight['Weight'], color='green', alpha=0.7)
plt.title('Average Weight Over Years')
plt.xlabel('Year')
plt.ylabel('Average Weight')
print("Graph shows that over the year the height and weight remain similar.")

# Question 20
print("Question 20!")
plt.subplot(2, 2, 4)
dummy_df = df[['Height', 'Weight', 'Age', 'Sport']]

# Normalize or standardize the numerical data (Height, Weight, Age)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(dummy_df[['Height', 'Weight', 'Age']])

# Choose the number of clusters (K)
k = 5

# Apply K-Means clustering
kmeans = KMeans(n_clusters=k, random_state=42)  # Create an instance of KMeans
# Create a copy of the DataFrame slice
dummy_df = df[['Height', 'Weight', 'Age', 'Sport']].copy()

# Normalize or standardize the numerical data (Height, Weight, Age)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(dummy_df[['Height', 'Weight', 'Age']])

# Choose the number of clusters (K)
k = 5

# Apply K-Means clustering
kmeans = KMeans(n_clusters=k, random_state=42)  # Create an instance of KMeans
dummy_df.loc[:, 'Cluster'] = kmeans.fit_predict(df_scaled)  # Use the instance to call fit_predict

# Visualize clusters
plt.scatter(dummy_df['Height'], dummy_df['Weight'], c=dummy_df['Cluster'], cmap='viridis', s=50, alpha=0.5)
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('K-Means Clustering of Athletes based on Height and Weight')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()  # Non-blocking show

plt.show(block=True)  # Ensure all figures are shown
