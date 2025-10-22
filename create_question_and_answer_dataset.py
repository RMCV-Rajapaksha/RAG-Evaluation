import os
import psycopg2
from sqlalchemy import make_url
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from datasets import Dataset
from openai import OpenAI
import time
from tqdm import tqdm
import json

# ------------------ CONFIGURATION ------------------
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv(find_dotenv())


class TestsetGenerator:
    """
    Generates evaluation testsets for RAG systems using context from vector store.
    Similar to Ragas testset generation approach.
    """
    
    def __init__(self, client, model="gpt-4o"):
        self.client = client
        self.model = model
        self.query_types = [
            "single_hop_specific",
            "multi_hop_abstract", 
            "multi_hop_specific"
        ]
    
    def generate_single_hop_specific_question(self, context):
        """
        Generate a specific question that requires single piece of information from context.
        """
        prompt = f"""Based on the following context, generate ONE specific factual question that can be answered directly using information from this context.

Requirements:
- The question should ask about a SPECIFIC fact, detail, or piece of information
- It should be answerable with a precise answer from the context
- Make it a straightforward, single-hop question (no reasoning chains needed)
- Do NOT create questions requiring information outside this context
- Question should be natural and practical

Context:
{context}

Generate only the question, nothing else:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at creating specific, fact-based questions from given context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating single-hop question: {e}")
            return None

    def generate_multi_hop_abstract_question(self, contexts):
        """
        Generate an abstract question requiring synthesis across multiple contexts.
        """
        combined_context = "\n\n---\n\n".join(contexts)
        
        prompt = f"""Based on the following contexts, generate ONE abstract question that requires understanding and synthesizing information from the provided contexts.

Requirements:
- The question should be conceptual or require reasoning across the contexts
- It should NOT ask about specific facts, but about relationships, implications, or broader understanding
- Make it require multi-hop reasoning (connecting multiple pieces of information)
- The answer should be derivable from the contexts provided
- Do NOT create questions requiring external knowledge

Contexts:
{combined_context}

Generate only the question, nothing else:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at creating reasoning-based questions that require synthesis of information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating multi-hop abstract question: {e}")
            return None

    def generate_multi_hop_specific_question(self, contexts):
        """
        Generate a specific question requiring information from multiple contexts.
        """
        combined_context = "\n\n---\n\n".join(contexts)
        
        prompt = f"""Based on the following contexts, generate ONE specific question that requires combining specific facts or details from multiple parts of the provided contexts.

Requirements:
- The question should ask for specific information that requires connecting details from different contexts
- It should have a precise, factual answer
- Make it require multi-hop reasoning (finding and connecting specific details)
- The answer should be clearly derivable from the contexts
- Do NOT create questions requiring external knowledge

Contexts:
{combined_context}

Generate only the question, nothing else:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at creating multi-hop factual questions from given contexts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating multi-hop specific question: {e}")
            return None

    def generate_ground_truth_answer(self, question, contexts):
        """
        Generate ground truth answer using ONLY the provided contexts.
        This is the reference answer for evaluation.
        """
        combined_context = "\n\n".join(contexts) if isinstance(contexts, list) else contexts
        
        prompt = f"""You are tasked with answering a question using ONLY the information provided in the contexts below.

CRITICAL RULES:
- Use ONLY information explicitly stated in the contexts
- Do NOT use any external knowledge or information not in the contexts
- If the contexts don't contain enough information, state this clearly
- Be accurate and precise
- Cite relevant parts of the context when appropriate

Contexts:
{combined_context}

Question: {question}

Provide a comprehensive answer based solely on the contexts above:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise assistant that answers questions using ONLY the provided contexts. You never use external knowledge."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=400
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating ground truth: {e}")
            return None


def generate_testset_from_chunks(rows, client, testset_size=10, model="gpt-4o"):
    """
    Generate a testset with different query types similar to Ragas.
    
    Query Distribution:
    - 50% Single-hop specific questions
    - 25% Multi-hop abstract questions  
    - 25% Multi-hop specific questions
    """
    generator = TestsetGenerator(client, model)
    testset = []
    
    # Filter out very short texts
    valid_rows = [(id_, node_id, text, metadata, embedding) 
                  for id_, node_id, text, metadata, embedding in rows 
                  if len(text.strip()) >= 100]
    
    if len(valid_rows) < 2:
        print("⚠️  Not enough valid chunks for testset generation")
        return []
    
    print(f"\n📝 Generating testset with {testset_size} samples...")
    print(f"Available chunks: {len(valid_rows)}")
    
    # Calculate distribution
    single_hop_count = int(testset_size * 0.5)
    multi_hop_abstract_count = int(testset_size * 0.25)
    multi_hop_specific_count = testset_size - single_hop_count - multi_hop_abstract_count
    
    print(f"\nQuery Distribution:")
    print(f"  - Single-hop specific: {single_hop_count}")
    print(f"  - Multi-hop abstract: {multi_hop_abstract_count}")
    print(f"  - Multi-hop specific: {multi_hop_specific_count}")
    
    with tqdm(total=testset_size, desc="Generating testset") as pbar:
        # Generate single-hop questions
        for i in range(single_hop_count):
            try:
                # Select a random chunk
                import random
                idx = random.randint(0, len(valid_rows) - 1)
                id_, node_id, text, metadata, embedding = valid_rows[idx]
                
                question = generator.generate_single_hop_specific_question(text)
                if not question:
                    continue
                
                time.sleep(0.5)
                
                ground_truth = generator.generate_ground_truth_answer(question, text)
                if not ground_truth:
                    continue
                
                testset.append({
                    'question': question,
                    'ground_truth': ground_truth,
                    'contexts': [text],
                    'query_type': 'single_hop_specific',
                    'chunk_ids': [id_],
                    'node_ids': [node_id]
                })
                
                pbar.update(1)
                time.sleep(0.5)
                
            except Exception as e:
                print(f"\n❌ Error generating single-hop question {i+1}: {e}")
                continue
        
        # Generate multi-hop abstract questions
        for i in range(multi_hop_abstract_count):
            try:
                # Select 2-3 random chunks
                import random
                num_chunks = random.randint(2, min(3, len(valid_rows)))
                selected_indices = random.sample(range(len(valid_rows)), num_chunks)
                selected_rows = [valid_rows[idx] for idx in selected_indices]
                
                contexts = [row[2] for row in selected_rows]
                question = generator.generate_multi_hop_abstract_question(contexts)
                if not question:
                    continue
                
                time.sleep(0.5)
                
                ground_truth = generator.generate_ground_truth_answer(question, contexts)
                if not ground_truth:
                    continue
                
                testset.append({
                    'question': question,
                    'ground_truth': ground_truth,
                    'contexts': contexts,
                    'query_type': 'multi_hop_abstract',
                    'chunk_ids': [row[0] for row in selected_rows],
                    'node_ids': [row[1] for row in selected_rows]
                })
                
                pbar.update(1)
                time.sleep(0.5)
                
            except Exception as e:
                print(f"\n❌ Error generating multi-hop abstract question {i+1}: {e}")
                continue
        
        # Generate multi-hop specific questions
        for i in range(multi_hop_specific_count):
            try:
                # Select 2-3 random chunks
                import random
                num_chunks = random.randint(2, min(3, len(valid_rows)))
                selected_indices = random.sample(range(len(valid_rows)), num_chunks)
                selected_rows = [valid_rows[idx] for idx in selected_indices]
                
                contexts = [row[2] for row in selected_rows]
                question = generator.generate_multi_hop_specific_question(contexts)
                if not question:
                    continue
                
                time.sleep(0.5)
                
                ground_truth = generator.generate_ground_truth_answer(question, contexts)
                if not ground_truth:
                    continue
                
                testset.append({
                    'question': question,
                    'ground_truth': ground_truth,
                    'contexts': contexts,
                    'query_type': 'multi_hop_specific',
                    'chunk_ids': [row[0] for row in selected_rows],
                    'node_ids': [row[1] for row in selected_rows]
                })
                
                pbar.update(1)
                time.sleep(0.5)
                
            except Exception as e:
                print(f"\n❌ Error generating multi-hop specific question {i+1}: {e}")
                continue
    
    return testset


def save_testset_to_csv(testset, filename="rag_testset.csv"):
    """
    Save testset to CSV file with proper formatting.
    """
    # Convert contexts list to string for CSV
    formatted_testset = []
    for item in testset:
        formatted_item = item.copy()
        # Join contexts with separator
        formatted_item['contexts'] = " ||| ".join(item['contexts'])
        formatted_item['chunk_ids'] = str(item['chunk_ids'])
        formatted_item['node_ids'] = str(item['node_ids'])
        formatted_testset.append(formatted_item)
    
    df = pd.DataFrame(formatted_testset)
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"\n💾 Saved testset with {len(testset)} samples to {filename}")
    return df


def upload_to_huggingface(df, hf_token, hf_repo_id):
    """
    Upload testset to Hugging Face Hub.
    """
    try:
        # Convert contexts back to proper format for HF
        df_copy = df.copy()
        
        # Convert to Hugging Face Dataset
        dataset = Dataset.from_pandas(df_copy)
        
        # Push to hub
        print(f"\n🚀 Uploading testset to Hugging Face: {hf_repo_id}")
        dataset.push_to_hub(
            hf_repo_id,
            token=hf_token,
            private=False
        )
        print(f"✅ Successfully uploaded to https://huggingface.co/datasets/{hf_repo_id}")
        
    except Exception as e:
        print(f"❌ Error uploading to Hugging Face: {e}")
        import traceback
        traceback.print_exc()


def main():
    try:
        # Load environment variables
        connection_string = os.getenv("CONNECTION_STRING")
        table_name = os.getenv("DB_TABLE_NAME")
        hf_token = os.getenv("HF_TOKEN")
        hf_repo_id = os.getenv("HF_REPO_ID", "ChamaraVishwajithRajapaksha/RAG-Evaluation-Dataset")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        testset_size = int(os.getenv("TESTSET_SIZE", "10"))

        # Validate environment variables
        if not connection_string or not table_name:
            raise ValueError("Missing CONNECTION_STRING or DB_TABLE_NAME environment variable")
        
        if not hf_token:
            raise ValueError("Missing HF_TOKEN environment variable")
        
        if not openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable")

        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)

        # Connect to database
        print("🔌 Connecting to PostgreSQL database...")
        conn = psycopg2.connect(connection_string)
        cur = conn.cursor()

        # Query to retrieve all data
        print(f"📊 Querying table: {table_name}")
        cur.execute(f"SELECT id, node_id, text, metadata_, embedding FROM {table_name};")
        rows = cur.fetchall()

        if not rows:
            print("⚠️  No records found in the vector store table.")
            cur.close()
            conn.close()
            return

        print(f"✅ Retrieved {len(rows)} records from database.\n")

        # Display sample
        print("--- Sample Record ---")
        sample = rows[0]
        print(f"ID: {sample[0]}")
        print(f"Text Preview: {sample[2][:200]}...")
        print()

        # Generate testset with different query types
        testset = generate_testset_from_chunks(rows, client, testset_size=testset_size)

        if not testset:
            print("❌ No testset samples were generated. Exiting.")
            cur.close()
            conn.close()
            return

        print(f"\n🎉 Generated {len(testset)} testset samples!")

        # Display sample from each query type
        print("\n" + "="*80)
        print("SAMPLE TESTSET ENTRIES")
        print("="*80)
        
        for query_type in ['single_hop_specific', 'multi_hop_abstract', 'multi_hop_specific']:
            samples = [t for t in testset if t['query_type'] == query_type]
            if samples:
                print(f"\n--- {query_type.upper().replace('_', ' ')} ---")
                sample = samples[0]
                print(f"Question: {sample['question']}")
                print(f"Ground Truth: {sample['ground_truth'][:200]}...")
                print(f"Number of Contexts: {len(sample['contexts'])}")

        # Save to CSV
        csv_filename = "rag_testset.csv"
        df = save_testset_to_csv(testset, csv_filename)

        # Upload to Hugging Face
        upload_to_huggingface(df, hf_token, hf_repo_id)

        # Cleanup
        cur.close()
        conn.close()
        print("\n✨ Testset generation completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


# ------------------ ENTRY POINT ------------------
if __name__ == "__main__":
    main()