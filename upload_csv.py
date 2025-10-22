import os
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, Repository, login

def upload_csv_to_hf(csv_filename: str, repo_id: str, private: bool = False):
    """
    Read a CSV file from the current directory and upload it to Hugging Face Hub.
    
    Args:
      csv_filename: str — the name of the CSV file (in the same directory as this script).
      repo_id: str — e.g. "your-username/your-dataset".
      private: bool — whether the dataset repository should be private.
    """
    # 1. Load the CSV into a 🤗 Datasets Dataset
    dataset = load_dataset(
        "csv",
        data_files=csv_filename,
        split="train"  # you may rename this split or handle multiple splits if needed
    )
    print(f"Loaded dataset: {dataset}")
    
    # 2. Authenticate (you must have run `huggingface-cli login` or set HF_TOKEN env var)
    # Optionally using `login(token=…)` but usually CLI does this
    # login()  # uncomment if you want to login programmatically
    
    # 3. Push to hub
    dataset.push_to_hub(repo_id, private=private)
    print(f"Dataset uploaded to: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    CSV_FILENAME = "unique - unique.csv.csv"       # <-- put your actual CSV filename here
    REPO_ID     = "your-username/your-dataset"  # <-- change this to your HF repo name
    PRIVATE     = False                   # Set True if you want it private
    
    # Check file exists
    if not os.path.exists(CSV_FILENAME):
        raise FileNotFoundError(f"CSV file {CSV_FILENAME} not found in current directory.")
    
    upload_csv_to_hf(CSV_FILENAME, REPO_ID, PRIVATE)
