import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from datasets import Dataset
import fitz  # PyMuPDF

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


# ------------------ PDF PROCESSING ------------------
def extract_text_from_pdf(pdf_path: str) -> dict:
    """
    Extract text from a PDF file using PyMuPDF (fitz).
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        dict with 'path', 'metadata', 'content'
    """
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        
        # Extract text from all pages
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        
        return {
            "path": pdf_path,
            "metadata": {
                "title": metadata.get("title", Path(pdf_path).stem),
                "author": metadata.get("author", "Unknown"),
                "subject": metadata.get("subject", ""),
                "keywords": metadata.get("keywords", "")
            },
            "content": full_text
        }
        
    except Exception as e:
        return {
            "path": pdf_path,
            "metadata": {
                "title": Path(pdf_path).stem,
                "error": str(e)
            },
            "content": f"Error: {e}"
        }
    finally:
        if 'doc' in locals():
            doc.close()


def process_multiple_pdfs(pdf_paths: list) -> list:
    """
    Process multiple PDF files and extract their content.
    
    Args:
        pdf_paths (list): List of paths to PDF files
        
    Returns:
        list of dicts, each with 'path', 'metadata', 'content'
    """
    results = []
    
    for i, path in enumerate(pdf_paths, 1):
        print(f"Processing {i}/{len(pdf_paths)}: {path}")
        result = extract_text_from_pdf(path)
        results.append(result)
        print(f"  ✓ Completed: {result['metadata']['title']}")
    
    return results


# ------------------ RAGAS Q&A GENERATION ------------------

def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """
    Split long text into overlapping chunks for better Q&A generation.
    
    Args:
        text: Full text content
        chunk_size: Number of characters per chunk
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            last_question = chunk.rfind('?')
            last_exclaim = chunk.rfind('!')
            
            boundary = max(last_period, last_question, last_exclaim)
            if boundary > chunk_size * 0.7:  # Only break if we're at least 70% through
                chunk = chunk[:boundary + 1]
                end = start + boundary + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks


def convert_pdfs_to_documents(pdf_results: list, chunk_content: bool = True) -> list:
    """
    Convert PDF results into Langchain Document objects.
    
    Args:
        pdf_results: List of PDF dictionaries from process_multiple_pdfs
        chunk_content: If True, split long content into chunks
        
    Returns:
        List of Langchain Document objects
    """
    print("\n📄 Converting PDFs to documents...")
    
    documents = []
    
    for result in pdf_results:
        content = result['content']
        metadata = result['metadata']
        path = result['path']
        
        # Skip error content
        if content.startswith("Error:"):
            print(f"⚠️  Skipping {metadata['title']} due to error")
            continue
        
        # Split into chunks if needed
        if chunk_content and len(content) > 1000:
            chunks = split_text_into_chunks(content, chunk_size=1500, overlap=200)
            print(f"   Split '{metadata['title']}' into {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": "pdf",
                        "path": path,
                        "title": metadata['title'],
                        "author": metadata.get('author', 'Unknown'),
                        "subject": metadata.get('subject', ''),
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                )
                documents.append(doc)
        else:
            doc = Document(
                page_content=content,
                metadata={
                    "source": "pdf",
                    "path": path,
                    "title": metadata['title'],
                    "author": metadata.get('author', 'Unknown'),
                    "subject": metadata.get('subject', '')
                }
            )
            documents.append(doc)
    
    print(f"✅ Created {len(documents)} documents")
    return documents


def create_knowledge_graph_from_documents(documents):
    """
    Create a Ragas KnowledgeGraph from documents.
    """
    print("🧠 Creating Knowledge Graph from documents...")
    
    kg = KnowledgeGraph()
    
    # Create nodes from documents
    for i, doc in enumerate(documents):
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
    
    llm = LangchainLLMWrapper(ChatOpenAI(model=model_name, temperature=0.3))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    
    print("✅ LLM and Embeddings ready")
    return llm, embeddings


def enrich_knowledge_graph(kg, documents, llm, embeddings):
    """
    Apply Ragas transformations to enrich the knowledge graph.
    """
    print("\n🔧 Enriching Knowledge Graph with transformations...")
    
    try:
        transforms = default_transforms(
            documents=documents,
            llm=llm,
            embedding_model=embeddings
        )
        
        apply_transforms(kg, transforms)
        
        print(f"✅ Knowledge Graph enriched!")
        print(f"   Nodes: {len(kg.nodes)}")
        print(f"   Relationships: {len(kg.relationships)}")
        
        return kg, True
        
    except Exception as e:
        print(f"⚠️  KG enrichment failed: {e}")
        print("   Will use direct document-based generation instead")
        return kg, False


def generate_testset_with_ragas(kg, documents, llm, embeddings, testset_size=10, use_kg=True):
    """
    Generate Q&A testset using Ragas TestsetGenerator.
    """
    print(f"\n📝 Generating testset with {testset_size} samples using Ragas...")
    
    try:
        if use_kg and len(kg.relationships) > 0:
            print("🔄 Using Knowledge Graph approach...")
            generator = TestsetGenerator(
                llm=llm,
                embedding_model=embeddings,
                knowledge_graph=kg
            )
            
            query_distribution = default_query_distribution(llm)
            
            testset = generator.generate(
                testset_size=testset_size,
                query_distribution=query_distribution,
                with_debugging_logs=False
            )
            
        else:
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


def save_testset_to_csv(testset, filename="youtube_qa_testset.csv"):
    """
    Convert Ragas testset to pandas DataFrame and save to CSV.
    """
    print(f"\n💾 Saving testset to {filename}...")
    
    df = testset.to_pandas()
    df.to_csv(filename, index=False, encoding='utf-8')
    
    print(f"✅ Saved testset with {len(df)} samples")
    print(f"\nDataFrame columns: {list(df.columns)}")
    
    return df


def display_sample_questions(df):
    """
    Display sample questions from the testset.
    """
    print("\n" + "="*80)
    print("SAMPLE Q&A PAIRS")
    print("="*80)
    
    print(f"\nTotal samples: {len(df)}")
    
    if 'question' in df.columns:
        print("\n--- Sample Questions ---")
        for i, row in df.head(5).iterrows():
            print(f"\n{i+1}. Question: {row['question']}")
            if 'ground_truth' in df.columns and pd.notna(row['ground_truth']):
                gt_preview = str(row['ground_truth'])[:200]
                print(f"   Ground Truth: {gt_preview}{'...' if len(str(row['ground_truth'])) > 200 else ''}")


def upload_to_huggingface(df, hf_token, hf_repo_id):
    """
    Upload testset to Hugging Face Hub.
    """
    try:
        print(f"\n🚀 Uploading testset to Hugging Face: {hf_repo_id}")
        
        dataset = Dataset.from_pandas(df)
        
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
        hf_token = os.getenv("HF_TOKEN")
        hf_repo_id = os.getenv("HF_REPO_ID")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        testset_size = int(os.getenv("TESTSET_SIZE", "10"))
        model_name = os.getenv("MODEL_NAME", "gpt-4o")

        # Validate required environment variables
        if not openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable")

        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = openai_api_key

        print("="*80)
        print("YOUTUBE TRANSCRIPT Q&A GENERATION WITH RAGAS")
        print("="*80)

        # Step 1: Define PDF paths
        pdf_paths = [
            "/home/vishwajith/Desktop/Project/RAG-Evaluation/data/A2A vs ACP.pdf"
            # Add more PDF paths here
        ]
        
        print(f"\n� Processing {len(pdf_paths)} PDF file(s)...")

        # Step 2: Get content from PDF files
        pdf_results = process_multiple_pdfs(pdf_paths)

        # Step 3: Convert PDF content to Langchain documents
        documents = convert_pdfs_to_documents(pdf_results, chunk_content=True)
        
        if not documents:
            print("❌ No valid documents created. Exiting.")
            return

        # Step 4: Create Knowledge Graph
        kg = create_knowledge_graph_from_documents(documents)

        # Step 5: Setup LLM and Embeddings
        llm, embeddings = setup_llm_and_embeddings(model_name)

        # Step 6: Enrich Knowledge Graph
        kg_filename = "pdf_knowledge_graph.json"
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
        else:
            print("\n🔄 Creating new Knowledge Graph...")
            kg, use_kg = enrich_knowledge_graph(kg, documents, llm, embeddings)
            if use_kg:
                kg.save(kg_filename)
                print(f"\n💾 Saved Knowledge Graph to {kg_filename}")

        # Step 7: Generate Q&A testset
        testset = generate_testset_with_ragas(kg, documents, llm, embeddings, testset_size, use_kg)

        # Step 8: Save to CSV
        csv_filename = "pdf_qa_testset.csv"
        df = save_testset_to_csv(testset, csv_filename)

        # Step 9: Display samples
        display_sample_questions(df)

        # Step 10: Upload to Hugging Face (optional)
        if hf_token and hf_repo_id:
            upload_to_huggingface(df, hf_token, hf_repo_id)

        print("\n" + "="*80)
        print("✨ Q&A GENERATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\n📁 Files created:")
        print(f"   - {csv_filename}")
        if use_kg:
            print(f"   - {kg_filename}")
        if hf_token and hf_repo_id:
            print(f"\n🌐 Hugging Face: https://huggingface.co/datasets/{hf_repo_id}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()