import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('athlete_events.csv')
df = df.fillna({'Height': int(df['Height'].mean())})
df = df.fillna({'Age': int(df['Age'].mean())})
df = df.fillna({'Weight': int(df['Weight'].mean())})
rows, columns = df.shape
Unique_df = df.drop_duplicates(subset=['ID'])

# Question 16
print("Question 16!")
medal_count = df[df['Medal'].notna()]
sum_medal = medal_count[medal_count['Season'] == "Summer"]
win_medal = medal_count[medal_count['Season'] == "Winter"]
print(f"Total medals won in winter are {win_medal['ID'].count()} and in summer are {sum_medal['ID'].count()}.")

avg_height_weight = df.groupby('Year').agg({'Height': 'mean', 'Weight': 'mean'}).round(2)


# Plotting
plt.figure(figsize=(20, 12))


# Question 18
# Plotting Average Height
print("Question 18!")
plt.subplot(2, 2, 1)  # (rows, columns, index)
plt.bar(avg_height_weight.index, avg_height_weight['Height'], color='blue', alpha=0.7)
plt.title('Average Height Over Years')
plt.xlabel('Year')
plt.ylabel('Average Height')

# Plotting Average Weight
plt.subplot(2, 2, 2)  # (rows, columns, index)
plt.bar(avg_height_weight.index, avg_height_weight['Weight'], color='green', alpha=0.7)
plt.title('Average Weight Over Years')
plt.xlabel('Year')
plt.ylabel('Average Weight')
print("Graph shows that over the year the height and weight over the remains similar.")

# Question 20
print("Question 20!")
plt.subplot(2, 2, 3)
dummy_df = df[['Height', 'Weight', 'Age', 'Sport']]

# Normalize or standardize the numerical data (Height, Weight, Age)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(dummy_df[['Height', 'Weight', 'Age']])

# Choose the number of clusters (K) - you may need to adjust this based on your data and analysis
k = 5

# Apply K-Means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
dummy_df['Cluster'] = kmeans.fit_predict(df_scaled)

# Visualize clusters
plt.scatter(dummy_df['Height'], dummy_df['Weight'], c=dummy_df['Cluster'], cmap='viridis', s=50, alpha=0.5)
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('K-Means Clustering of Sports based on Height and Weight')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()


plt.tight_layout()
plt.show()
