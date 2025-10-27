import os
import psycopg2
from sqlalchemy import make_url
from llama_index.vector_stores.postgres import PGVectorStore
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi

# ------------------ CONFIGURATION ------------------
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv(find_dotenv())


def main():
    try:
        connection_string = os.getenv("CONNECTION_STRING")
        table_name = os.getenv("DB_TABLE_NAME")
        hf_token = os.getenv("HF_TOKEN")
        hf_repo_id = os.getenv("HF_REPO_ID", "ChamaraVishwajithRajapaksha/RAG-Evaluation-Dataset")

        if not connection_string or not table_name:
            raise ValueError("Missing CONNECTION_STRING or DB_TABLE_NAME environment variable")
        
        if not hf_token:
            raise ValueError("Missing HF_TOKEN environment variable")

        url = make_url(connection_string)

        # Initialize PGVectorStore (optional)
        print("Connecting to PGVectorStore...")
        vector_store = PGVectorStore.from_params(
            database=url.database,
            host=url.host,
            password=url.password,
            port=url.port,
            user=url.username,
            table_name=table_name,
            embed_dim=1536,
        )

        # Direct DB connection for reading
        print("Connecting to PostgreSQL database directly...")
        conn = psycopg2.connect(connection_string)
        cur = conn.cursor()

        # Query to retrieve all data
        cur.execute(f"SELECT id, node_id, text, metadata_, embedding FROM {table_name};")
        rows = cur.fetchall()

        if not rows:
            print("No records found in the vector store table.")
            cur.close()
            conn.close()
            return

        print(f"\n✅ Retrieved {len(rows)} records from database.")

        # Display sample records
        for i, (id_, node_id, text, metadata, embedding) in enumerate(rows[:3], start=1):
            print(f"\n--- Sample Chunk {i} ---")
            print(f"ID: {id_}")
            print(f"Node ID: {node_id}")
            print(f"Text: {text[:100]}...")  # Show first 100 chars
            print(f"Metadata: {metadata}")
            print(f"Embedding (first 10 dims): {embedding[:10] if embedding else 'None'}...")

        # Create DataFrame
        df = pd.DataFrame(rows, columns=["id", "node_id", "text", "metadata", "embedding"])
        
        # Convert embedding list to string representation for storage (optional)
        # If you want to keep embeddings as arrays, remove this line
        df['embedding'] = df['embedding'].apply(lambda x: str(x) if x is not None else None)
        
        print(f"\n📊 DataFrame shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Create Hugging Face Dataset
        ds = Dataset.from_pandas(df)
        
        print(f"\n🚀 Uploading dataset to Hugging Face: {hf_repo_id}")
        
        # Upload to Hugging Face Hub
        ds.push_to_hub(
            repo_id=hf_repo_id,
            token=hf_token,
            private=False  # Set to True if you want a private dataset
        )
        
        print(f"✅ Successfully uploaded dataset to: https://huggingface.co/datasets/{hf_repo_id}")

        cur.close()
        conn.close()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


# ------------------ ENTRY POINT ------------------
if __name__ == "__main__":
    main()