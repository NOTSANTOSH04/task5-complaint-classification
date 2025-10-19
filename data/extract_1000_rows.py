import pandas as pd

# Specify the path to your CSV file
file_path = 'C:\_Main\complaint_classification\complaints.csv'

# Read only the first 1000 rows
df_first_1000 = pd.read_csv(file_path, nrows=1000)

# You can now work with df_first_1000, which contains the first 1000 rows
# For example, to save it to a new CSV:
df_first_1000.to_csv('first_1000_rows.csv', index=False)