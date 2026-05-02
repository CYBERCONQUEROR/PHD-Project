import pandas as pd

# Load the dataset
df = pd.read_csv("Hugging_face text.csv")

# Print all unique labels
print("Labels found in dataset:")
print(df['status'].unique())

# Or see how many of each label there are
print("\nLabel counts:")
print(df['status'].value_counts())