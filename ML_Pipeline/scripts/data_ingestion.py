import pandas as pd

def load_data(file_path: str):
    """Loads raw data from a CSV file."""
    df = pd.read_csv(file_path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    return df

if __name__ == "__main__":
    df = load_data("data/raw_data.csv")
    print("Data loaded successfully!")