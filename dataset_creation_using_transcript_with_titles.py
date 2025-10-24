import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from datasets import Dataset

# YouTube Transcript imports
from youtube_transcript_api import YouTubeTranscriptApi
import requests
from bs4 import BeautifulSoup

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


# ------------------ YOUTUBE TRANSCRIPT SCRAPER ------------------
class YouTubeTranscriptScraper:
    """Scraper to fetch YouTube video transcripts as paragraphs."""

    def __init__(self, language="en"):
        self.language = language

    def _get_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL"""
        try:
            if "v=" in url:
                return url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in url:
                return url.split("youtu.be/")[1].split("?")[0]
            else:
                raise ValueError("Invalid YouTube URL format")
        except Exception as e:
            raise ValueError(f"Error extracting video ID: {e}")

    def _fetch_metadata(self, url: str) -> dict:
        """Scrape title & description from YouTube page"""
        try:
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.title.string if soup.title else "No title found"
            description_tag = soup.find("meta", attrs={"name": "description"})
            description = description_tag["content"] if description_tag else "No description found"
            return {
                "title": title.strip(),
                "description": description.strip(),
                "url": url
            }
        except Exception as e:
            return {
                "title": "Unknown",
                "description": f"Error fetching metadata: {e}",
                "url": url
            }


def get_transcript_as_paragraph(url: str, language: str = "en") -> dict:
    """
    Get YouTube video transcript as a single paragraph with metadata.
    
    Args:
        url (str): YouTube video URL
        language (str): Language code (default: "en")
        
    Returns:
        dict with 'url', 'metadata', 'transcript_paragraph'
    """
    scraper = YouTubeTranscriptScraper(language=language)
    
    try:
        video_id = scraper._get_video_id(url)
        metadata = scraper._fetch_metadata(url)
        
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=[language])
        
        # Combine all text into a single paragraph
        full_text = " ".join([entry.text for entry in transcript])
        
        return {
            "url": url,
            "metadata": metadata,
            "transcript_paragraph": full_text
        }
        
    except Exception as e:
        metadata = scraper._fetch_metadata(url)
        return {
            "url": url,
            "metadata": metadata,
            "transcript_paragraph": f"Error: {e}"
        }


def get_multiple_transcripts_as_paragraphs(urls: list, language: str = "en") -> list:
    """
    Get transcripts from multiple YouTube videos as paragraphs.
    
    Args:
        urls (list): List of YouTube video URLs
        language (str): Language code (default: "en")
        
    Returns:
        list of dicts, each with 'url', 'metadata', 'transcript_paragraph'
    """
    results = []
    
    for i, url in enumerate(urls, 1):
        print(f"Processing {i}/{len(urls)}: {url}")
        result = get_transcript_as_paragraph(url, language)
        results.append(result)
        print(f"  ✓ Completed: {result['metadata']['title']}")
    
    return results


# ------------------ RAGAS Q&A GENERATION ------------------

def split_transcript_into_chunks(transcript_paragraph: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """
    Split long transcript into overlapping chunks for better Q&A generation.
    
    Args:
        transcript_paragraph: Full transcript text
        chunk_size: Number of characters per chunk
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List of text chunks
    """
    if len(transcript_paragraph) <= chunk_size:
        return [transcript_paragraph]
    
    chunks = []
    start = 0
    
    while start < len(transcript_paragraph):
        end = start + chunk_size
        chunk = transcript_paragraph[start:end]
        
        # Try to break at sentence boundary
        if end < len(transcript_paragraph):
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


def convert_transcripts_to_documents(transcript_results: list, chunk_transcripts: bool = True) -> list:
    """
    Convert YouTube transcript results into Langchain Document objects.
    
    Args:
        transcript_results: List of transcript dictionaries from get_multiple_transcripts_as_paragraphs
        chunk_transcripts: If True, split long transcripts into chunks
        
    Returns:
        List of Langchain Document objects
    """
    print("\n📄 Converting transcripts to documents...")
    
    documents = []
    
    for result in transcript_results:
        transcript = result['transcript_paragraph']
        metadata = result['metadata']
        url = result['url']
        
        # Skip error transcripts
        if transcript.startswith("Error:"):
            print(f"⚠️  Skipping {metadata['title']} due to error")
            continue
        
        # Split into chunks if needed
        if chunk_transcripts and len(transcript) > 1000:
            chunks = split_transcript_into_chunks(transcript, chunk_size=1500, overlap=200)
            print(f"   Split '{metadata['title']}' into {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": "youtube_transcript",
                        "url": url,
                        "title": metadata['title'],
                        "video_title": metadata['title'],  # Added for easy access
                        "description": metadata['description'],
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                )
                documents.append(doc)
        else:
            doc = Document(
                page_content=transcript,
                metadata={
                    "source": "youtube_transcript",
                    "url": url,
                    "title": metadata['title'],
                    "video_title": metadata['title'],  # Added for easy access
                    "description": metadata['description']
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


def extract_video_title_from_contexts(contexts):
    """
    Extract video title from contexts metadata.
    
    Args:
        contexts: List of context objects or strings
        
    Returns:
        Video title or None
    """
    if not contexts:
        return None
    
    # Handle different context formats
    for context in contexts:
        try:
            # If context is a Document object
            if hasattr(context, 'metadata'):
                if 'video_title' in context.metadata:
                    return context.metadata['video_title']
                elif 'title' in context.metadata:
                    return context.metadata['title']
            
            # If context is a dict
            elif isinstance(context, dict):
                if 'metadata' in context:
                    if 'video_title' in context['metadata']:
                        return context['metadata']['video_title']
                    elif 'title' in context['metadata']:
                        return context['metadata']['title']
        except Exception:
            continue
    
    return None


def save_testset_to_csv(testset, filename="youtube_qa_testset.csv"):
    """
    Convert Ragas testset to pandas DataFrame and save to CSV.
    Adds video_title column by extracting from contexts.
    """
    print(f"\n💾 Saving testset to {filename}...")
    
    df = testset.to_pandas()
    
    # Add video_title column
    if 'contexts' in df.columns:
        print("📋 Extracting video titles from contexts...")
        df['video_title'] = df['contexts'].apply(extract_video_title_from_contexts)
        
        # Count how many titles were extracted
        titles_found = df['video_title'].notna().sum()
        print(f"   ✓ Extracted video titles for {titles_found}/{len(df)} samples")
    else:
        print("⚠️  No 'contexts' column found, adding empty video_title column")
        df['video_title'] = None
    
    # Reorder columns to put video_title near the beginning
    cols = df.columns.tolist()
    if 'video_title' in cols:
        # Move video_title to be after question
        cols.remove('video_title')
        if 'question' in cols:
            question_idx = cols.index('question')
            cols.insert(question_idx + 1, 'video_title')
        else:
            cols.insert(0, 'video_title')
        df = df[cols]
    
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
            if 'video_title' in df.columns and pd.notna(row['video_title']):
                print(f"   Video: {row['video_title']}")
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

        # Step 1: Define YouTube URLs
        youtube_urls = [
            "https://www.youtube.com/watch?v=X5eC3Rk9FBQ"
            #   "https://www.youtube.com/watch?v=-nwIoiPB8CE",
            #     "https://www.youtube.com/watch?v=GoYR-iK2UUk",
            #     "https://www.youtube.com/watch?v=CYii_zExySA",
            #     "https://www.youtube.com/watch?v=banNxyyTSI4",
            #     "https://www.youtube.com/watch?v=wobNffok7nc"
            # Add more URLs here
        ]
        
        print(f"\n📺 Processing {len(youtube_urls)} YouTube video(s)...")

        # Step 2: Get transcripts from YouTube videos
        transcript_results = get_multiple_transcripts_as_paragraphs(youtube_urls)

        # Step 3: Convert transcripts to Langchain documents
        documents = convert_transcripts_to_documents(transcript_results, chunk_transcripts=True)
        
        if not documents:
            print("❌ No valid documents created. Exiting.")
            return

        # Step 4: Create Knowledge Graph
        kg = create_knowledge_graph_from_documents(documents)

        # Step 5: Setup LLM and Embeddings
        llm, embeddings = setup_llm_and_embeddings(model_name)

        # Step 6: Enrich Knowledge Graph
        kg_filename = "youtube_knowledge_graph.json"
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

        # Step 8: Save to CSV (now includes video_title extraction)
        csv_filename = "youtube_qa_testset.csv"
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