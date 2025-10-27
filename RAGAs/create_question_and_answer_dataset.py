import os
import psycopg2
from sqlalchemy import make_url
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from datasets import Dataset
import asyncio

# Ragas imports
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset.synthesizers import default_query_distribution
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings

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


def create_knowledge_graph_from_chunks(rows):
    """
    Create a Ragas KnowledgeGraph from database chunks.
    Creates both DOCUMENT nodes (for persona generation) and CHUNK nodes.
    """
    print("🧠 Creating Knowledge Graph from chunks...")
    
    kg = KnowledgeGraph()
    documents = []  # Store documents for transforms
    
    # First, create a parent DOCUMENT node
    # Ragas needs DOCUMENT nodes for persona generation
    all_texts = []
    
    for id_, node_id, text, metadata, embedding in rows:
        # Skip very short texts
        if len(text.strip()) < 50:
            continue
        all_texts.append(text)
    
    # Create a parent document node (Ragas needs this for personas)
    parent_doc_node = Node(
        type=NodeType.DOCUMENT,
        properties={
            "page_content": "\n\n".join(all_texts[:5]),  # Combine first few chunks
            "document_metadata": {"source": "vector_store", "type": "combined"}
        }
    )
    kg.nodes.append(parent_doc_node)
    
    # Now add each chunk as a node to the knowledge graph
    for id_, node_id, text, metadata, embedding in rows:
        # Skip very short texts
        if len(text.strip()) < 50:
            continue
            
        # Create a document object for transforms
        from langchain_core.documents import Document
        doc = Document(
            page_content=text,
            metadata={
                "chunk_id": str(id_),
                "node_id": str(node_id),
                "original_metadata": metadata if metadata else {}
            }
        )
        documents.append(doc)
        
        # Create a CHUNK node (these will be used for question generation)
        node = Node(
            type=NodeType.CHUNK,
            properties={
                "page_content": text,
                "chunk_id": str(id_),
                "node_id": str(node_id),
                "metadata": metadata if metadata else {}
            }
        )
        kg.nodes.append(node)
    
    print(f"✅ Created Knowledge Graph with {len(kg.nodes)} nodes")
    print(f"   - 1 DOCUMENT node (for persona generation)")
    print(f"   - {len(kg.nodes)-1} CHUNK nodes (for question generation)")
    return kg, documents


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
    
    # Create default transforms (extractors + relationship builders)
    transforms = default_transforms(
        documents=documents,  # Pass the document objects
        llm=llm,
        embedding_model=embeddings
    )
    
    # Apply all transforms to the knowledge graph
    apply_transforms(kg, transforms)
    
    print(f"✅ Knowledge Graph enriched!")
    print(f"   Nodes: {len(kg.nodes)}")
    print(f"   Relationships: {len(kg.relationships)}")
    
    return kg


def generate_testset_with_ragas(kg, llm, embeddings, testset_size=10):
    """
    Generate testset using Ragas TestsetGenerator with different query types.
    
    Query Distribution (Ragas default):
    - 50% Single-hop Specific queries
    - 25% Multi-hop Abstract queries
    - 25% Multi-hop Specific queries
    """
    print(f"\n📝 Generating testset with {testset_size} samples using Ragas...")
    
    try:
        # Try with TestsetGenerator first
        generator = TestsetGenerator(
            llm=llm,
            embedding_model=embeddings,
            knowledge_graph=kg
        )
        
        # Get default query distribution
        query_distribution = default_query_distribution(llm)
        
        print("\n📊 Query Distribution:")
        for synthesizer, probability in query_distribution:
            print(f"   - {synthesizer.__class__.__name__}: {probability*100}%")
        
        # Generate the testset (without run_config to avoid the error)
        testset = generator.generate(
            testset_size=testset_size,
            query_distribution=query_distribution,
            with_debugging_logs=False
        )
        
        print(f"\n✅ Generated {len(testset)} testset samples!")
        return testset
        
    except (ValueError, Exception) as e:
        error_msg = str(e)
        if "No nodes that satisfied" in error_msg or "No relationships match" in error_msg or "Cannot form clusters" in error_msg:
            print(f"\n⚠️  Knowledge Graph approach failed: {error_msg}")
            print("Switching to simpler document-based generation method...")
            
            # Alternative: Use generate_with_langchain_docs
            # This bypasses the Knowledge Graph approach
            from langchain_core.documents import Document
            
            # Get documents from KG nodes
            docs = []
            for node in kg.nodes:
                if node.type == NodeType.CHUNK and "page_content" in node.properties:
                    doc = Document(
                        page_content=node.properties["page_content"],
                        metadata=node.properties.get("metadata", {})
                    )
                    docs.append(doc)
            
            if not docs:
                # If no CHUNK nodes, try DOCUMENT nodes
                for node in kg.nodes:
                    if "page_content" in node.properties:
                        doc = Document(
                            page_content=node.properties["page_content"],
                            metadata=node.properties.get("metadata", {})
                        )
                        docs.append(doc)
            
            print(f"📄 Using {len(docs)} documents for testset generation...")
            
            if not docs:
                raise ValueError("No documents found in Knowledge Graph to generate testset")
            
            # Create a fresh generator without KG
            generator = TestsetGenerator(
                llm=llm,
                embedding_model=embeddings
            )
            
            # Use the simpler generate_with_langchain_docs method
            print("🔄 Generating testset with document-based approach...")
            testset = generator.generate_with_langchain_docs(
                documents=docs,
                testset_size=testset_size
            )
            
            print(f"\n✅ Generated {len(testset)} testset samples using document-based method!")
            return testset
        else:
            # Re-raise if it's a different error
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
    Display sample questions from each query type.
    """
    print("\n" + "="*80)
    print("SAMPLE TESTSET ENTRIES")
    print("="*80)
    
    # Show basic stats
    print(f"\nTotal samples: {len(df)}")
    
    if 'question' in df.columns:
        print("\n--- Sample Questions ---")
        for i, row in df.head(3).iterrows():
            print(f"\n{i+1}. Question: {row['question']}")
            if 'ground_truth' in df.columns:
                print(f"   Ground Truth: {str(row['ground_truth'])[:200]}...")
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
        hf_repo_id = os.getenv("HF_REPO_ID", "ChamaraVishwajithRajapaksha/RAG-Evaluation-Dataset")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        testset_size = int(os.getenv("TESTSET_SIZE", "10"))
        model_name = os.getenv("MODEL_NAME", "gpt-4o")

        # Validate required environment variables
        if not connection_string or not table_name:
            raise ValueError("Missing CONNECTION_STRING or DB_TABLE_NAME environment variable")
        
        if not hf_token:
            raise ValueError("Missing HF_TOKEN environment variable")
        
        if not openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable")

        # Set OpenAI API key for langchain
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
        print()

        # Step 2: Create Knowledge Graph from chunks
        kg, documents = create_knowledge_graph_from_chunks(rows)

        # Step 3: Setup LLM and Embeddings
        llm, embeddings = setup_llm_and_embeddings(model_name)

        # Step 4: Check if knowledge graph exists
        kg_filename = "knowledge_graph.json"
        if os.path.exists(kg_filename):
            print(f"\n📂 Loading existing Knowledge Graph from {kg_filename}")
            kg = KnowledgeGraph.load(kg_filename)
            print(f"✅ Loaded Knowledge Graph with {len(kg.nodes)} nodes")
        else:
            print("\n🔄 Creating new Knowledge Graph...")
            # Enrich Knowledge Graph with transformations
            # This extracts entities, keyphrases, and builds relationships
            kg = enrich_knowledge_graph(kg, documents, llm, embeddings)
            # Save knowledge graph
            kg.save(kg_filename)
            print(f"\n💾 Saved Knowledge Graph to {kg_filename}")

        # Step 5: Generate testset using Ragas
        testset = generate_testset_with_ragas(kg, llm, embeddings, testset_size)

        # Step 6: Convert to DataFrame and save to CSV
        csv_filename = "ragas_testset.csv"
        df = save_testset_to_csv(testset, csv_filename)

        # Step 7: Display sample questions
        display_sample_questions(df)

        # Step 8: Upload to Hugging Face
        upload_to_huggingface(df, hf_token, hf_repo_id)

        print("\n" + "="*80)
        print("✨ TESTSET GENERATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\n📁 Files created:")
        print(f"   - {csv_filename}")
        print(f"   - {kg_filename}")
        print(f"\n🌐 Hugging Face: https://huggingface.co/datasets/{hf_repo_id}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


# ------------------ ENTRY POINT ------------------
if __name__ == "__main__":
    main()