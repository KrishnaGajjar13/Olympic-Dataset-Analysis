import pandas as pd

# Question 17
print("Question 17")
df = pd.read_csv('athlete_events.csv')
df = df.fillna({'Height': int(df['Height'].mean())})
df = df.fillna({'Age': int(df['Age'].mean())})
df = df.fillna({'Weight': int(df['Weight'].mean())})
rows, columns = df.shape
Unique_df = df.drop_duplicates(subset=['ID'])
med_df = df[df['Medal'].notna()]
grp_year = med_df.groupby('Year')
for name, group in grp_year:
    print(f"In {name}, these are the top 5 teams who won the most medals!")
    group = group['Team'].value_counts()
    group = group.nlargest(5)
    group.describe()
    print(group.to_string(), "\n")
