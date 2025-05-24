import pandas as pd

# Load the original dataset
df = pd.read_csv("Iris.csv")

# Drop 'Id' column (useless for classification)
df = df.drop(columns=["Id"], errors='ignore')

# Remove duplicate rows
df = df.drop_duplicates()

# Reset index
df = df.reset_index(drop=True)

# Define columns
cat_cols = ["Species"]
num_cols = [col for col in df.columns if col not in cat_cols]

# Handle missing values
for col in num_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = df[col].astype("category")

# Remove extreme outliers in PetalLengthCm (as an example)
length_threshold = df["PetalLengthCm"].quantile(0.99)
df = df[df["PetalLengthCm"] < length_threshold]

# Save cleaned dataset
df.to_csv("cleaned_iris.csv", index=False)
print("Data has been cleaned & saved as 'cleaned_iris.csv'")
