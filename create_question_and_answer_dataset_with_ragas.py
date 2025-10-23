import os
import psycopg2
from sqlalchemy import make_url
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from datasets import Dataset

# Ragas imports
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset.synthesizers import default_query_distribution
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# ------------------ CONFIGURATION ------------------
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv(find_dotenv())


def load_chunks_from_vectorstore(connection_string, table_name):
    """
    Load all chunks from the PostgreSQL vector store.
    """
    print("🔌 Connecting to PostgreSQL database...")
    conn = psycopg2.connect(connection_string)
    cur = conn.cursor()

    print(f"📊 Querying table: {table_name}")
    cur.execute(f"SELECT id, node_id, text, metadata_, embedding FROM {table_name};")
    rows = cur.fetchall()

    cur.close()
    conn.close()

    if not rows:
        print("⚠️  No records found in the vector store table.")
        return []

    print(f"✅ Retrieved {len(rows)} chunks from database.\n")
    return rows


def combine_short_chunks(rows, min_tokens=150, max_combined=5):
    """
    Combine short chunks into longer documents to meet Ragas requirements.
    Groups consecutive chunks together until they reach minimum token count.
    """
    print(f"🔄 Combining short chunks (min {min_tokens} tokens)...")
    
    combined_docs = []
    current_text = []
    current_metadata = []
    current_token_count = 0
    chunks_in_group = 0
    
    for id_, node_id, text, metadata, embedding in rows:
        # Skip empty texts
        if not text or len(text.strip()) < 10:
            continue
        
        # Rough token estimation (1 token ≈ 4 characters)
        text_tokens = len(text) // 4
        
        current_text.append(text)
        current_metadata.append({
            "chunk_id": str(id_),
            "node_id": str(node_id),
            "original_metadata": metadata if metadata else {}
        })
        current_token_count += text_tokens
        chunks_in_group += 1
        
        # If we've reached minimum tokens or max chunks, create a combined document
        if current_token_count >= min_tokens or chunks_in_group >= max_combined:
            combined_content = "\n\n".join(current_text)
            combined_meta = {
                "source": "vector_store_combined",
                "num_chunks": chunks_in_group,
                "chunks": current_metadata
            }
            
            combined_docs.append(Document(
                page_content=combined_content,
                metadata=combined_meta
            ))
            
            # Reset for next group
            current_text = []
            current_metadata = []
            current_token_count = 0
            chunks_in_group = 0
    
    # Add any remaining chunks
    if current_text:
        combined_content = "\n\n".join(current_text)
        combined_meta = {
            "source": "vector_store_combined",
            "num_chunks": chunks_in_group,
            "chunks": current_metadata
        }
        combined_docs.append(Document(
            page_content=combined_content,
            metadata=combined_meta
        ))
    
    print(f"✅ Created {len(combined_docs)} combined documents")
    print(f"   Average tokens per document: {sum(len(d.page_content)//4 for d in combined_docs)//len(combined_docs) if combined_docs else 0}")
    
    return combined_docs


def create_knowledge_graph_from_documents(documents):
    """
    Create a Ragas KnowledgeGraph from combined documents.
    """
    print("🧠 Creating Knowledge Graph from documents...")
    
    kg = KnowledgeGraph()
    
    # Create nodes from documents
    for i, doc in enumerate(documents):
        # Create a DOCUMENT node (Ragas needs this for personas)
        node = Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": doc.page_content,
                "document_metadata": doc.metadata
            }
        )
        kg.nodes.append(node)
    
    print(f"✅ Created Knowledge Graph with {len(kg.nodes)} DOCUMENT nodes")
    return kg


def setup_llm_and_embeddings(model_name="gpt-4o"):
    """
    Setup LLM and embedding models for Ragas.
    """
    print(f"🤖 Setting up LLM ({model_name}) and Embeddings...")
    
    # Initialize OpenAI LLM wrapped for Ragas
    llm = LangchainLLMWrapper(ChatOpenAI(model=model_name, temperature=0.3))
    
    # Initialize OpenAI Embeddings wrapped for Ragas
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    
    print("✅ LLM and Embeddings ready")
    return llm, embeddings


def enrich_knowledge_graph(kg, documents, llm, embeddings):
    """
    Apply Ragas transformations to enrich the knowledge graph.
    This extracts entities, keyphrases, and builds relationships between nodes.
    """
    print("\n🔧 Enriching Knowledge Graph with transformations...")
    print("This will extract entities, keyphrases, and build relationships...")
    
    # Check document lengths before applying transforms
    avg_tokens = sum(len(d.page_content)//4 for d in documents) // len(documents) if documents else 0
    print(f"   Average document length: {avg_tokens} tokens")
    
    if avg_tokens < 100:
        print("⚠️  Warning: Documents still too short, skipping transforms")
        print("   Will use direct document-based generation instead")
        return kg, False
    
    try:
        # Create default transforms (extractors + relationship builders)
        transforms = default_transforms(
            documents=documents,
            llm=llm,
            embedding_model=embeddings
        )
        
        # Apply all transforms to the knowledge graph
        apply_transforms(kg, transforms)
        
        print(f"✅ Knowledge Graph enriched!")
        print(f"   Nodes: {len(kg.nodes)}")
        print(f"   Relationships: {len(kg.relationships)}")
        
        return kg, True
        
    except ValueError as e:
        if "too short" in str(e).lower():
            print(f"⚠️  Documents still too short: {e}")
            print("   Will use direct document-based generation instead")
            return kg, False
        raise


def generate_testset_with_ragas(kg, documents, llm, embeddings, testset_size=10, use_kg=True):
    """
    Generate testset using Ragas TestsetGenerator.
    Falls back to document-based generation if KG approach fails.
    """
    print(f"\n📝 Generating testset with {testset_size} samples using Ragas...")
    
    try:
        if use_kg and len(kg.relationships) > 0:
            # Try with Knowledge Graph first
            print("🔄 Using Knowledge Graph approach...")
            generator = TestsetGenerator(
                llm=llm,
                embedding_model=embeddings,
                knowledge_graph=kg
            )
            
            query_distribution = default_query_distribution(llm)
            
            print("\n📊 Query Distribution:")
            for synthesizer, probability in query_distribution:
                print(f"   - {synthesizer.__class__.__name__}: {probability*100}%")
            
            testset = generator.generate(
                testset_size=testset_size,
                query_distribution=query_distribution,
                with_debugging_logs=False
            )
            
        else:
            # Use direct document-based generation
            print("🔄 Using document-based approach...")
            generator = TestsetGenerator(
                llm=llm,
                embedding_model=embeddings
            )
            
            testset = generator.generate_with_langchain_docs(
                documents=documents,
                testset_size=testset_size
            )
        
        print(f"\n✅ Generated {len(testset)} testset samples!")
        return testset
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n⚠️  Knowledge Graph approach failed: {error_msg}")
        
        if use_kg:
            print("Switching to document-based generation method...")
            return generate_testset_with_ragas(kg, documents, llm, embeddings, testset_size, use_kg=False)
        else:
            raise


def save_testset_to_csv(testset, filename="ragas_testset.csv"):
    """
    Convert Ragas testset to pandas DataFrame and save to CSV.
    """
    print(f"\n💾 Saving testset to {filename}...")
    
    # Convert to pandas DataFrame
    df = testset.to_pandas()
    
    # Save to CSV
    df.to_csv(filename, index=False, encoding='utf-8')
    
    print(f"✅ Saved testset with {len(df)} samples")
    print(f"\nDataFrame columns: {list(df.columns)}")
    
    return df


def display_sample_questions(df):
    """
    Display sample questions from the testset.
    """
    print("\n" + "="*80)
    print("SAMPLE TESTSET ENTRIES")
    print("="*80)
    
    print(f"\nTotal samples: {len(df)}")
    
    if 'question' in df.columns:
        print("\n--- Sample Questions ---")
        for i, row in df.head(5).iterrows():
            print(f"\n{i+1}. Question: {row['question']}")
            if 'ground_truth' in df.columns and pd.notna(row['ground_truth']):
                gt_preview = str(row['ground_truth'])[:200]
                print(f"   Ground Truth: {gt_preview}{'...' if len(str(row['ground_truth'])) > 200 else ''}")
            if 'contexts' in df.columns:
                num_contexts = len(row['contexts']) if isinstance(row['contexts'], list) else 1
                print(f"   Number of contexts: {num_contexts}")


def upload_to_huggingface(df, hf_token, hf_repo_id):
    """
    Upload testset to Hugging Face Hub.
    """
    try:
        print(f"\n🚀 Uploading testset to Hugging Face: {hf_repo_id}")
        
        # Convert to Hugging Face Dataset
        dataset = Dataset.from_pandas(df)
        
        # Push to hub
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
        hf_repo_id = os.getenv("HF_REPO_ID")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        testset_size = int(os.getenv("TESTSET_SIZE", "10"))
        model_name = os.getenv("MODEL_NAME", "gpt-4o")
        min_tokens = int(os.getenv("MIN_TOKENS_PER_DOC", "150"))
        max_chunks_combined = int(os.getenv("MAX_CHUNKS_COMBINED", "5"))

        # Validate required environment variables
        if not connection_string or not table_name:
            raise ValueError("Missing CONNECTION_STRING or DB_TABLE_NAME environment variable")
        
        if not hf_token:
            raise ValueError("Missing HF_TOKEN environment variable")
        
        if not openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable")

        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = openai_api_key

        print("="*80)
        print("RAGAS TESTSET GENERATION FOR RAG EVALUATION")
        print("="*80)

        # Step 1: Load chunks from vector store
        rows = load_chunks_from_vectorstore(connection_string, table_name)
        if not rows:
            print("❌ No data to process. Exiting.")
            return

        # Display sample chunk
        print("\n--- Sample Chunk ---")
        sample = rows[0]
        print(f"ID: {sample[0]}")
        print(f"Text Preview: {sample[2][:200]}...")
        print(f"Estimated tokens: {len(sample[2])//4}")
        print()

        # Step 2: Combine short chunks into longer documents
        documents = combine_short_chunks(rows, min_tokens, max_chunks_combined)
        
        if not documents:
            print("❌ No valid documents created. Exiting.")
            return

        # Step 3: Create Knowledge Graph from combined documents
        kg = create_knowledge_graph_from_documents(documents)

        # Step 4: Setup LLM and Embeddings
        llm, embeddings = setup_llm_and_embeddings(model_name)

        # Step 5: Check if knowledge graph exists
        kg_filename = "knowledge_graph.json"
        use_kg = False
        
        if os.path.exists(kg_filename):
            print(f"\n📂 Loading existing Knowledge Graph from {kg_filename}")
            try:
                kg = KnowledgeGraph.load(kg_filename)
                print(f"✅ Loaded Knowledge Graph with {len(kg.nodes)} nodes")
                use_kg = len(kg.relationships) > 0
            except Exception as e:
                print(f"⚠️  Could not load KG: {e}, creating new one...")
                kg, use_kg = enrich_knowledge_graph(kg, documents, llm, embeddings)
                if use_kg:
                    kg.save(kg_filename)
                    print(f"\n💾 Saved Knowledge Graph to {kg_filename}")
        else:
            print("\n🔄 Creating new Knowledge Graph...")
            kg, use_kg = enrich_knowledge_graph(kg, documents, llm, embeddings)
            if use_kg:
                kg.save(kg_filename)
                print(f"\n💾 Saved Knowledge Graph to {kg_filename}")

        # Step 6: Generate testset using Ragas
        testset = generate_testset_with_ragas(kg, documents, llm, embeddings, testset_size, use_kg)

        # Step 7: Convert to DataFrame and save to CSV
        csv_filename = "ragas_testset.csv"
        df = save_testset_to_csv(testset, csv_filename)

        # Step 8: Display sample questions
        display_sample_questions(df)

        # Step 9: Upload to Hugging Face
        upload_to_huggingface(df, hf_token, hf_repo_id)

        print("\n" + "="*80)
        print("✨ TESTSET GENERATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\n📁 Files created:")
        print(f"   - {csv_filename}")
        if use_kg:
            print(f"   - {kg_filename}")
        print(f"\n🌐 Hugging Face: https://huggingface.co/datasets/{hf_repo_id}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    main()