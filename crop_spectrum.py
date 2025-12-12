import os
import pandas as pd

# === Settings ===
input_folder = "calibrated_lineouts"       # Folder with your original CSVs
output_folder = "calibrated_lineouts_cropped"    # Folder to save updated CSVs
os.makedirs(output_folder, exist_ok=True)

# Define your energy range
energy_min = 880   # lower bound
energy_max = 5000   # upper bound

# === Loop through CSV files ===
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_folder, filename)

        # Read CSV
        df = pd.read_csv(file_path)

        # Filter by energy range
        df_cropped = df[(df["energy"] >= energy_min) & (df["energy"] <= energy_max)]

        # Save updated CSV
        output_path = os.path.join(output_folder, filename)
        df_cropped.to_csv(output_path, index=False)

        print(f"Cropped and saved: {output_path}")
