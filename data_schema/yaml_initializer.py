import pandas as pd

loaded_df = pd.read_csv("../Input_Data_File/lending_club_loan.csv")

# Step 1: Create the 'columns' section
columns_section = "columns:\n"
for col in loaded_df.columns:
    columns_section += f"  - {col}: {loaded_df[col].dtype}\n"

# Step 2: Create the 'numerical_columns' section
numerical_cols = loaded_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
numerical_section = "\nnumerical_columns:\n"
for col in numerical_cols:
    numerical_section += f"  - {col}\n"

# Combine both sections
schema_yaml = columns_section + numerical_section

# save to the existing file named schema.yaml
with open("../data_schema/schema.yaml", "w") as file:
    file.write(schema_yaml)

