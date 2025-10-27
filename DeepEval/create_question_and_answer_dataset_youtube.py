import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from datasets import Dataset

# YouTube Transcript imports
from youtube_transcript_api import YouTubeTranscriptApi
import requests
from bs4 import BeautifulSoup

# DeepEval imports
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import Evolution

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


# ------------------ DEEPEVAL Q&A GENERATION ------------------

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


def prepare_contexts_from_transcripts(transcript_results: list, chunk_transcripts: bool = True) -> list:
    """
    Prepare contexts from YouTube transcripts for DeepEval.
    Each context is a list of related text chunks from the same video.
    
    Args:
        transcript_results: List of transcript dictionaries
        chunk_transcripts: If True, split long transcripts into chunks
        
    Returns:
        List of contexts (each context is a list of text strings)
    """
    print("\n📄 Preparing contexts from transcripts...")
    
    contexts = []
    
    for result in transcript_results:
        transcript = result['transcript_paragraph']
        metadata = result['metadata']
        
        # Skip error transcripts
        if transcript.startswith("Error:"):
            print(f"⚠️  Skipping {metadata['title']} due to error")
            continue
        
        # Split into chunks if needed
        if chunk_transcripts and len(transcript) > 1500:
            chunks = split_transcript_into_chunks(transcript, chunk_size=1500, overlap=200)
            print(f"   Split '{metadata['title']}' into {len(chunks)} chunks")
            
            # Create contexts from chunks (group 2-3 chunks per context)
            context_size = 3
            for i in range(0, len(chunks), context_size):
                context = chunks[i:i + context_size]
                if context:
                    contexts.append(context)
        else:
            # Use full transcript as single context
            contexts.append([transcript])
            print(f"   Added '{metadata['title']}' as single context")
    
    print(f"✅ Created {len(contexts)} contexts")
    return contexts


def save_transcripts_to_temp_files(transcript_results: list, temp_dir="temp_youtube_docs") -> list:
    """
    Save YouTube transcripts to temporary text files for document-based generation.
    
    Args:
        transcript_results: List of transcript dictionaries
        temp_dir: Directory to save temporary files
        
    Returns:
        List of document file paths
    """
    print(f"\n💾 Saving transcripts to temporary files in {temp_dir}/...")
    
    # Create temp directory
    Path(temp_dir).mkdir(exist_ok=True)
    
    document_paths = []
    
    for i, result in enumerate(transcript_results):
        transcript = result['transcript_paragraph']
        metadata = result['metadata']
        
        # Skip error transcripts
        if transcript.startswith("Error:"):
            print(f"⚠️  Skipping {metadata['title']} due to error")
            continue
        
        # Create filename from video title (sanitized)
        safe_title = "".join(c for c in metadata['title'] if c.isalnum() or c in (' ', '-', '_'))[:50]
        file_path = Path(temp_dir) / f"video_{i+1}_{safe_title}.txt"
        
        # Write transcript to file with metadata header
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"Title: {metadata['title']}\n")
            f.write(f"URL: {result['url']}\n")
            f.write(f"Description: {metadata['description']}\n\n")
            f.write("=" * 80 + "\n")
            f.write("TRANSCRIPT\n")
            f.write("=" * 80 + "\n\n")
            f.write(transcript)
        
        document_paths.append(str(file_path))
        print(f"   Saved: {safe_title}")
    
    print(f"✅ Created {len(document_paths)} temporary document files")
    return document_paths


def generate_testset_with_deepeval(
    contexts=None,
    document_paths=None,
    testset_size=10,
    model_name="gpt-4o",
    num_evolutions=2,
    enable_breadth_evolution=True
):
    """
    Generate Q&A testset using DeepEval Synthesizer.
    
    Args:
        contexts: List of context lists for context-based generation
        document_paths: List of document paths for document-based generation
        testset_size: Number of goldens to generate
        model_name: OpenAI model to use
        num_evolutions: Number of evolution steps
        enable_breadth_evolution: Whether to emphasize breadth (coverage) over complexity
    """
    print(f"\n📝 Generating testset with {testset_size} samples using DeepEval...")
    print(f"   Model: {model_name}")
    print(f"   Evolutions: {num_evolutions}")
    
    # Initialize synthesizer with custom model
    synthesizer = Synthesizer(model=model_name)
    
    # Configure evolution distribution
    if enable_breadth_evolution:
        # Emphasize breadth for better coverage
        evolutions = {
            Evolution.REASONING: 0.15,
            Evolution.MULTICONTEXT: 0.15,
            Evolution.CONCRETIZING: 0.05,
            Evolution.CONSTRAINED: 0.05,
            Evolution.COMPARATIVE: 0.1,
            Evolution.HYPOTHETICAL: 0.1,
            Evolution.IN_BREADTH: 0.4,  # High breadth for coverage
        }
    else:
        # Balanced distribution
        evolutions = {
            Evolution.REASONING: 0.15,
            Evolution.MULTICONTEXT: 0.15,
            Evolution.CONCRETIZING: 0.15,
            Evolution.CONSTRAINED: 0.1,
            Evolution.COMPARATIVE: 0.15,
            Evolution.HYPOTHETICAL: 0.1,
            Evolution.IN_BREADTH: 0.2,
        }
    
    print("\n📊 Evolution Distribution:")
    for evolution_type, probability in evolutions.items():
        print(f"   - {evolution_type.name}: {probability*100}%")
    
    try:
        # Generate goldens based on input type
        if contexts is not None:
            print(f"\n🔄 Generating goldens from {len(contexts)} contexts...")
            goldens = synthesizer.generate_goldens_from_contexts(
                contexts=contexts,
                max_goldens_per_context=max(1, testset_size // len(contexts))
            )
        elif document_paths is not None:
            print(f"\n🔄 Generating goldens from {len(document_paths)} documents...")
            goldens = synthesizer.generate_goldens_from_docs(
                document_paths=document_paths,
                max_goldens_per_document=max(1, testset_size // len(document_paths)),
                chunk_size=1024,
                chunk_overlap=100
            )
        else:
            raise ValueError("Either contexts or document_paths must be provided")
        
        print(f"\n✅ Generated {len(goldens)} goldens!")
        return goldens, synthesizer
        
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        raise


def save_testset_to_csv(goldens, synthesizer, filename="youtube_qa_testset.csv"):
    """
    Convert DeepEval goldens to pandas DataFrame and save to CSV.
    """
    print(f"\n💾 Saving testset to {filename}...")
    
    # Convert to pandas DataFrame
    df = synthesizer.to_pandas()
    
    # Save to CSV
    df.to_csv(filename, index=False, encoding='utf-8')
    
    print(f"✅ Saved testset with {len(df)} samples")
    print(f"\nDataFrame columns: {list(df.columns)}")
    
    return df


def display_sample_goldens(df, goldens):
    """
    Display sample goldens from the testset with quality metrics.
    """
    print("\n" + "="*80)
    print("SAMPLE Q&A PAIRS")
    print("="*80)
    
    print(f"\nTotal goldens: {len(df)}")
    
    if 'input' in df.columns:
        print("\n--- Sample Questions with Quality Scores ---")
        for i in range(min(5, len(goldens))):
            golden = goldens[i]
            print(f"\n{i+1}. Input: {golden.input}")
            
            # Show expected output preview
            output_preview = golden.expected_output[:200] if len(golden.expected_output) > 200 else golden.expected_output
            print(f"   Expected Output: {output_preview}{'...' if len(golden.expected_output) > 200 else ''}")
            
            # Display quality scores if available
            if hasattr(golden, 'additional_metadata') and golden.additional_metadata:
                print(f"   Quality Metrics:")
                if 'context_quality' in golden.additional_metadata:
                    print(f"      - Context Quality: {golden.additional_metadata['context_quality']}")
                if 'synthetic_input_quality' in golden.additional_metadata:
                    print(f"      - Input Quality: {golden.additional_metadata['synthetic_input_quality']}")
                if 'evolutions' in golden.additional_metadata:
                    evols = golden.additional_metadata['evolutions']
                    print(f"      - Evolutions Applied: {', '.join(evols) if isinstance(evols, list) else evols}")


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


def cleanup_temp_files(temp_dir="temp_youtube_docs"):
    """
    Clean up temporary document files.
    """
    import shutil
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)
        print(f"🧹 Cleaned up temporary directory: {temp_dir}")


def main():
    try:
        # Load environment variables
        hf_token = os.getenv("HF_TOKEN")
        hf_repo_id = os.getenv("HF_REPO_ID", "your_username/youtube-qa-dataset")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        testset_size = int(os.getenv("TESTSET_SIZE", "10"))
        model_name = os.getenv("MODEL_NAME", "gpt-4o")
        num_evolutions = int(os.getenv("NUM_EVOLUTIONS", "2"))
        
        # Generation method: 'contexts' or 'documents'
        generation_method = os.getenv("GENERATION_METHOD", "contexts")

        # Validate required environment variables
        if not openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable")

        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = openai_api_key

        print("="*80)
        print("YOUTUBE TRANSCRIPT Q&A GENERATION WITH DEEPEVAL")
        print("="*80)

        # Step 1: Define YouTube URLs
        youtube_urls = [
            "https://www.youtube.com/watch?v=GoYR-iK2UUk",
            # Add more URLs here
            # "https://www.youtube.com/watch?v=X5eC3Rk9FBQ",
            # "https://www.youtube.com/watch?v=-nwIoiPB8CE",
            # "https://www.youtube.com/watch?v=CYii_zExySA",
            # "https://www.youtube.com/watch?v=banNxyyTSI4",
            # "https://www.youtube.com/watch?v=wobNffok7nc",
            # "https://www.youtube.com/watch?v=bTj0h5x8W70"
        ]
        
        print(f"\n📺 Processing {len(youtube_urls)} YouTube video(s)...")

        # Step 2: Get transcripts from YouTube videos
        transcript_results = get_multiple_transcripts_as_paragraphs(youtube_urls)
        
        # Check if we have valid transcripts
        valid_transcripts = [t for t in transcript_results if not t['transcript_paragraph'].startswith("Error:")]
        if not valid_transcripts:
            print("❌ No valid transcripts obtained. Exiting.")
            return

        # Step 3: Generate testset based on method
        goldens = None
        synthesizer = None
        
        if generation_method == "contexts":
            print("\n📋 Using context-based generation method")
            contexts = prepare_contexts_from_transcripts(transcript_results, chunk_transcripts=True)
            
            if not contexts:
                print("❌ No contexts created. Exiting.")
                return
            
            # Generate testset from contexts
            goldens, synthesizer = generate_testset_with_deepeval(
                contexts=contexts,
                testset_size=testset_size,
                model_name=model_name,
                num_evolutions=num_evolutions,
                enable_breadth_evolution=True
            )
        
        elif generation_method == "documents":
            print("\n📄 Using document-based generation method")
            document_paths = save_transcripts_to_temp_files(transcript_results)
            
            if not document_paths:
                print("❌ No documents created. Exiting.")
                return
            
            # Generate testset from documents
            goldens, synthesizer = generate_testset_with_deepeval(
                document_paths=document_paths,
                testset_size=testset_size,
                model_name=model_name,
                num_evolutions=num_evolutions,
                enable_breadth_evolution=True
            )
            
            # Cleanup temporary files
            cleanup_temp_files()
        
        else:
            raise ValueError(f"Invalid GENERATION_METHOD: {generation_method}. Use 'contexts' or 'documents'")

        # Step 4: Save to CSV
        csv_filename = "youtube_qa_testset.csv"
        df = save_testset_to_csv(goldens, synthesizer, csv_filename)

        # Step 5: Display samples
        display_sample_goldens(df, goldens)

        # Step 6: Upload to Hugging Face (optional)
        if hf_token and hf_repo_id:
            upload_to_huggingface(df, hf_token, hf_repo_id)

        print("\n" + "="*80)
        print("✨ Q&A GENERATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\n📁 Files created:")
        print(f"   - {csv_filename}")
        print(f"\n💡 Generation Statistics:")
        print(f"   - Total goldens: {len(goldens)}")
        print(f"   - Videos processed: {len(valid_transcripts)}")
        print(f"   - Method: {generation_method}")
        print(f"   - Evolutions applied: {num_evolutions}")
        
        if hf_token and hf_repo_id:
            print(f"\n🌐 Hugging Face: https://huggingface.co/datasets/{hf_repo_id}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()