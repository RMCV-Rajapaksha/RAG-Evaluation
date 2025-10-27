import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from datasets import Dataset
import fitz  # PyMuPDF
import tempfile
import shutil

# DeepEval imports
from deepeval.synthesizer import Synthesizer, Evolution
from deepeval.synthesizer.config import EvolutionConfig

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
        
        doc.close()
        
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
        print(f"❌ Error processing {pdf_path}: {e}")
        return {
            "path": pdf_path,
            "metadata": {
                "title": Path(pdf_path).stem,
                "error": str(e)
            },
            "content": ""
        }


def process_multiple_pdfs(pdf_paths: list) -> list:
    """
    Process multiple PDF files and extract their content.
    
    Args:
        pdf_paths (list): List of paths to PDF files
        
    Returns:
        list of dicts, each with 'path', 'metadata', 'content'
    """
    results = []
    
    print(f"\n📚 Processing {len(pdf_paths)} PDF file(s)...")
    
    for i, path in enumerate(pdf_paths, 1):
        print(f"   {i}/{len(pdf_paths)}: {Path(path).name}")
        result = extract_text_from_pdf(path)
        
        if result['content']:
            results.append(result)
            print(f"      ✓ Extracted {len(result['content'])} characters")
        else:
            print(f"      ⚠️  No content extracted")
    
    print(f"\n✅ Successfully processed {len(results)} PDF(s)")
    return results


def split_text_into_chunks(text: str, chunk_size: int = 1500, overlap: int = 200) -> list:
    """
    Split long text into overlapping chunks.
    
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


def prepare_contexts_from_pdfs(pdf_results: list, chunk_content: bool = True) -> list:
    """
    Prepare contexts from PDF results for DeepEval.
    
    Args:
        pdf_results: List of PDF dictionaries from process_multiple_pdfs
        chunk_content: If True, split long content into chunks
        
    Returns:
        List of contexts (each context is a list of related text chunks)
    """
    print("\n📄 Preparing contexts from PDFs...")
    
    contexts = []
    
    for result in pdf_results:
        content = result['content']
        metadata = result['metadata']
        
        # Skip empty content
        if not content or len(content.strip()) < 100:
            print(f"⚠️  Skipping '{metadata['title']}' - insufficient content")
            continue
        
        # Split into chunks if needed
        if chunk_content and len(content) > 1500:
            chunks = split_text_into_chunks(content, chunk_size=1500, overlap=200)
            print(f"   Split '{metadata['title']}' into {len(chunks)} chunks")
            
            # Group chunks into contexts (each context contains related chunks)
            # For simplicity, we'll create contexts with 1-3 chunks each
            i = 0
            while i < len(chunks):
                # Take 1-3 chunks for each context
                context_size = min(3, len(chunks) - i)
                context = chunks[i:i + context_size]
                contexts.append(context)
                i += context_size
        else:
            # Single chunk context
            contexts.append([content])
    
    print(f"✅ Prepared {len(contexts)} contexts")
    return contexts


def save_pdfs_as_temp_files(pdf_results: list, temp_dir: str, chunk_content: bool = True) -> list:
    """
    Save PDF content as temporary text files for DeepEval document-based generation.
    
    Args:
        pdf_results: List of PDF dictionaries
        temp_dir: Temporary directory path
        chunk_content: If True, create separate files for chunks
        
    Returns:
        List of file paths
    """
    print("\n💾 Saving PDF content to temporary files...")
    
    file_paths = []
    
    for result in pdf_results:
        content = result['content']
        metadata = result['metadata']
        
        # Skip empty content
        if not content or len(content.strip()) < 100:
            continue
        
        if chunk_content and len(content) > 1500:
            # Split into chunks and save each as a separate file
            chunks = split_text_into_chunks(content, chunk_size=1500, overlap=200)
            
            for idx, chunk in enumerate(chunks):
                file_path = os.path.join(temp_dir, f"{metadata['title']}_chunk_{idx}.txt")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(chunk)
                file_paths.append(file_path)
        else:
            # Save entire content as one file
            file_path = os.path.join(temp_dir, f"{metadata['title']}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            file_paths.append(file_path)
    
    print(f"✅ Saved {len(file_paths)} text files")
    return file_paths


def setup_synthesizer(model_name="gpt-4o", num_evolutions=2):
    """
    Setup DeepEval Synthesizer with specified model and evolution config.
    """
    print(f"\n🤖 Setting up DeepEval Synthesizer with model: {model_name}")
    
    # Define evolution distribution
    evolutions = {
        Evolution.REASONING: 0.15,
        Evolution.MULTICONTEXT: 0.10,
        Evolution.CONCRETIZING: 0.10,
        Evolution.CONSTRAINED: 0.10,
        Evolution.COMPARATIVE: 0.10,
        Evolution.HYPOTHETICAL: 0.10,
        Evolution.IN_BREADTH: 0.35,
    }
    
    print("\n📊 Evolution Distribution:")
    for evolution_type, probability in evolutions.items():
        print(f"   - {evolution_type.value}: {probability*100}%")
    
    # Create evolution config
    evolution_config = EvolutionConfig(
        evolutions=evolutions,
        num_evolutions=num_evolutions
    )
    
    # Initialize synthesizer with evolution config
    synthesizer = Synthesizer(
        model=model_name,
        evolution_config=evolution_config
    )
    
    print(f"✅ Synthesizer ready with {num_evolutions} evolution steps")
    return synthesizer


def generate_testset_from_contexts(synthesizer, contexts, max_goldens_per_context=2):
    """
    Generate testset using DeepEval from pre-prepared contexts.
    This is faster and more direct than document-based generation.
    """
    print(f"\n📝 Generating testset using DeepEval (context-based method)...")
    print(f"   Using {len(contexts)} contexts")
    print(f"   Max goldens per context: {max_goldens_per_context}")
    
    # Generate goldens from contexts
    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=contexts,
        max_goldens_per_context=max_goldens_per_context,
        include_expected_output=True
    )
    
    print(f"\n✅ Generated {len(goldens)} goldens!")
    return goldens


def generate_testset_from_docs(synthesizer, document_paths, chunk_size=1024, 
                               chunk_overlap=100, max_contexts_per_document=3,
                               max_goldens_per_context=2):
    """
    Generate testset using DeepEval from document files.
    This includes automatic chunking and context generation.
    """
    print(f"\n📝 Generating testset using DeepEval (document-based method)...")
    print(f"   Documents: {len(document_paths)}")
    print(f"   Chunk size: {chunk_size} tokens")
    print(f"   Chunk overlap: {chunk_overlap} tokens")
    print(f"   Max contexts per document: {max_contexts_per_document}")
    print(f"   Max goldens per context: {max_goldens_per_context}")
    
    # Generate goldens from documents
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=document_paths,
        max_goldens_per_context=max_goldens_per_context,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_contexts_per_document=max_contexts_per_document,
        include_expected_output=True
    )
    
    print(f"\n✅ Generated {len(goldens)} goldens!")
    return goldens


def goldens_to_dataframe(synthesizer, goldens):
    """
    Convert DeepEval goldens to pandas DataFrame.
    """
    print("\n📊 Converting goldens to DataFrame...")
    
    # DeepEval provides a built-in method to convert to pandas
    df = synthesizer.to_pandas()
    
    print(f"✅ Created DataFrame with {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")
    
    return df


def save_testset_to_csv(df, filename="deepeval_pdf_testset.csv"):
    """
    Save testset DataFrame to CSV.
    """
    print(f"\n💾 Saving testset to {filename}...")
    
    # Save to CSV
    df.to_csv(filename, index=False, encoding='utf-8')
    
    print(f"✅ Saved testset with {len(df)} samples")
    
    return df


def display_sample_goldens(df, goldens):
    """
    Display sample goldens from the testset.
    """
    print("\n" + "="*80)
    print("SAMPLE Q&A PAIRS FROM TESTSET")
    print("="*80)
    
    # Show basic stats
    print(f"\nTotal samples: {len(df)}")
    
    if 'input' in df.columns:
        print("\n--- Sample Goldens ---")
        num_samples = min(5, len(goldens))
        
        for i in range(num_samples):
            golden = goldens[i]
            print(f"\n{i+1}. Question: {golden.input}")
            
            # Show expected output preview
            expected = golden.expected_output
            if len(expected) > 300:
                print(f"   Expected Output: {expected[:300]}...")
            else:
                print(f"   Expected Output: {expected}")
            
            # Show context info
            print(f"   Number of context chunks: {len(golden.context)}")
            
            # Show evolution information if available
            if hasattr(golden, 'additional_metadata') and golden.additional_metadata:
                if 'evolutions' in golden.additional_metadata:
                    evolutions = golden.additional_metadata['evolutions']
                    print(f"   Evolutions applied: {', '.join(evolutions) if evolutions else 'None'}")
                
                if 'synthetic_input_quality' in golden.additional_metadata:
                    quality = golden.additional_metadata['synthetic_input_quality']
                    print(f"   Input Quality Score: {quality}")


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
        hf_token = os.getenv("HF_TOKEN")
        hf_repo_id = os.getenv("HF_REPO_ID", "YourUsername/PDF-QA-Dataset")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        testset_size = int(os.getenv("TESTSET_SIZE", "10"))
        model_name = os.getenv("MODEL_NAME", "gpt-4o")
        
        # DeepEval specific parameters
        chunk_size = int(os.getenv("CHUNK_SIZE", "1024"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
        num_evolutions = int(os.getenv("NUM_EVOLUTIONS", "2"))
        max_contexts_per_document = int(os.getenv("MAX_CONTEXTS_PER_DOCUMENT", "3"))
        max_goldens_per_context = int(os.getenv("MAX_GOLDENS_PER_CONTEXT", "2"))
        use_contexts = os.getenv("USE_CONTEXTS", "true").lower() == "true"

        # Validate required environment variables
        if not openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable")

        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = openai_api_key

        print("="*80)
        print("DEEPEVAL TESTSET GENERATION FROM PDF FILES")
        print("="*80)

        # Step 1: Define PDF paths
        # You can modify this list or read from environment variable
        pdf_paths_str = os.getenv("PDF_PATHS", "")
        
        if pdf_paths_str:
            # Read from environment variable (comma-separated)
            pdf_paths = [p.strip() for p in pdf_paths_str.split(",")]
        else:
            # Default PDF paths
            pdf_paths = [
                "/home/vishwajith/Desktop/Project/RAG-Evaluation/data/A2A vs ACP.pdf"
                # Add more PDF paths here or use environment variable
            ]
        
        # Validate PDF paths
        valid_pdf_paths = []
        for path in pdf_paths:
            if os.path.exists(path):
                valid_pdf_paths.append(path)
            else:
                print(f"⚠️  Warning: PDF not found: {path}")
        
        if not valid_pdf_paths:
            raise ValueError("No valid PDF paths provided. Please check PDF_PATHS or update the script.")
        
        print(f"\n📚 Found {len(valid_pdf_paths)} valid PDF file(s)")

        # Step 2: Extract content from PDF files
        pdf_results = process_multiple_pdfs(valid_pdf_paths)
        
        if not pdf_results:
            print("❌ No PDF content extracted. Exiting.")
            return

        # Step 3: Setup DeepEval Synthesizer
        synthesizer = setup_synthesizer(model_name, num_evolutions)

        # Step 4: Generate testset using DeepEval
        if use_contexts:
            print("\n🔄 Using context-based generation method...")
            # Prepare contexts from PDFs (faster, more direct)
            contexts = prepare_contexts_from_pdfs(pdf_results, chunk_content=True)
            
            # Generate goldens from contexts
            goldens = generate_testset_from_contexts(
                synthesizer=synthesizer,
                contexts=contexts,
                max_goldens_per_context=max_goldens_per_context
            )
        else:
            print("\n🔄 Using document-based generation method...")
            # Create temporary directory for PDF content files
            temp_dir = tempfile.mkdtemp(prefix="deepeval_pdfs_")
            
            try:
                # Save PDF content to temporary files
                document_paths = save_pdfs_as_temp_files(pdf_results, temp_dir, chunk_content=True)
                
                # Generate goldens from documents
                goldens = generate_testset_from_docs(
                    synthesizer=synthesizer,
                    document_paths=document_paths,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    max_contexts_per_document=max_contexts_per_document,
                    max_goldens_per_context=max_goldens_per_context
                )
            finally:
                # Cleanup temporary files
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"🧹 Cleaned up temporary files")

        # Step 5: Convert to DataFrame
        df = goldens_to_dataframe(synthesizer, goldens)

        # Step 6: Save to CSV
        csv_filename = "deepeval_pdf_testset.csv"
        save_testset_to_csv(df, csv_filename)

        # Step 7: Display sample goldens
        display_sample_goldens(df, goldens)

        # Step 8: Upload to Hugging Face (optional)
        if hf_token and hf_repo_id:
            upload_to_huggingface(df, hf_token, hf_repo_id)

        print("\n" + "="*80)
        print("✨ TESTSET GENERATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\n📁 Files created:")
        print(f"   - {csv_filename}")
        
        if hf_token and hf_repo_id:
            print(f"\n🌐 Hugging Face: https://huggingface.co/datasets/{hf_repo_id}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


# ------------------ ENTRY POINT ------------------
if __name__ == "__main__":
    main()