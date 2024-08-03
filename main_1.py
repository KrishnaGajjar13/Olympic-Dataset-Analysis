import pandas as pd

df = pd.read_csv('athlete_events.csv')
rows, columns = df.shape
Unique_df = df.drop_duplicates(subset=['ID'])


# Question 1
print("Question 1!")
print(f"There are {rows} rows and {columns} columns in dataset.\n")

# Question 2
print("Question 2!")
Total_sports = df['Sport'].nunique()
print(f"There are {Total_sports} unique sports played.\n")

# Question 3
print("Question 3!")
male_athletes = df[df['Sex'] == 'M']
tot_male = male_athletes['ID'].nunique()
female_athletes = df[df['Sex'] == 'F']
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
tot_medal = rows - df['Medal'].isna().sum()
print(f"Total Medals won are: {tot_medal}")
Gold_Winners = df[df['Medal'] == 'Gold']
Silver_Winners = df[df['Medal'] == 'Silver']
Bronze_Winners = df[df['Medal'] == 'Bronze']
print(f"There are total {Gold_Winners.shape[0]} Gold Medalist")
print(f"There are total {Silver_Winners.shape[0]} Silver Medalist")
print(f"There are total {Bronze_Winners.shape[0]} Bronze Medalist\n")

# Question 6
print("Question 6!")
Team_count = Unique_df['Team'].value_counts()
print(f"{Team_count.idxmax()} has most athletes with total {Team_count.max()} athletes.\n")

# Question 7
print("Question 7!")
columns_with_missing = df.columns[df.isna().any()]  # Identify columns with missing values
missing_counts = df[columns_with_missing].isna().sum()  # Count missing values in each column
# Print columns with missing values and their counts
print("Columns with missing values:")
for column in columns_with_missing:
    count = missing_counts[column]
    print(f"{column}:\t{count}")
print()
