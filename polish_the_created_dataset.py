import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm
import time

# ------------------ CONFIGURATION ------------------
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv(find_dotenv())


# ------------------ POLISHING FUNCTIONS ------------------

def create_polishing_prompt():
    """
    Create a prompt template for polishing Q&A pairs.
    """
    system_message = """
You are an expert editor specializing in polishing question-answer (QA) pairs for datasets.

Your task is to:
1. Fix spelling errors (typos, misspellings)
2. Correct grammar and punctuation
3. Fix capitalization errors (e.g., proper nouns, product names, people's names)
   - Example: "Malit Jing" → "Malith Jayasinhe" / nadish → Nadeesh / Rana Kloff → Raniya Khalaf
   - Check for similar mistakes and correct them
4. Correct incorrect product names based on the following list:
   - WSO2 API Manager
   - WSO2 Identity Server
   - WSO2 Integrator
   - WSO2 Micro Integrator
   - Choreo
   - WSO2 API Platform for Kubernetes
   (e.g., "coro" → "Choreo")
5. Ensure consistency in terminology
6. Improve clarity and readability
7. Fix any factual errors in names (people, products, companies, places)
8. If the answer is not relevant to the question, modify either the question or the answer so that the question can be correctly answered using the given content.

CRITICAL RULES:
- Preserve the meaning and intent of the original text
- Do NOT add new information or change facts
- Do NOT make the text unnecessarily longer
- Keep technical and domain-specific terms intact
- Maintain the same tone and style
- Only fix actual errors — do not rewrite correct text

Return ONLY the polished text without any explanations or comments.
"""



    user_template = """Please polish the following {content_type}:

Original {content_type}:
{original_text}

Context (for reference only, DO NOT modify):
{context}

Polished {content_type}:"""

    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", user_template)
    ])


def detect_column_names(df):
    """
    Detect the actual column names for question, answer, and context.
    """
    columns = df.columns.tolist()
    
    # Detect question column
    question_col = None
    for col in ['user_input', 'question', 'query', 'questions']:
        if col in columns:
            question_col = col
            break
    
    # Detect answer column
    answer_col = None
    for col in ['reference', 'ground_truth', 'answer', 'response', 'ground_truths']:
        if col in columns:
            answer_col = col
            break
    
    # Detect context column (for providing context, NOT for polishing)
    context_col = None
    for col in ['reference_contexts', 'contexts', 'context']:
        if col in columns:
            context_col = col
            break
    
    return question_col, answer_col, context_col


def safe_get_context(row, context_col):
    """
    Safely extract context value from a row, handling arrays, lists, and NaN.
    """
    if not context_col:
        return ""
    
    try:
        context_value = row[context_col]
        
        # Handle None or NaN
        if context_value is None:
            return ""
        
        # Check for pandas NA types
        if pd.isna(context_value) if not isinstance(context_value, (list, np.ndarray)) else False:
            return ""
        
        # Handle lists and arrays
        if isinstance(context_value, (list, np.ndarray)):
            # Filter out None/NaN values and convert to strings
            clean_items = [str(item) for item in context_value if item is not None and (not isinstance(item, float) or not pd.isna(item))]
            return " ".join(clean_items)
        
        # Handle regular strings
        return str(context_value)
        
    except Exception as e:
        print(f"   ⚠️  Error extracting context: {e}")
        return ""


def polish_text(text, content_type, context, llm, prompt_template, max_retries=3):
    """
    Polish a single piece of text (question or answer) using LLM.
    
    Args:
        text: Text to polish
        content_type: "question" or "answer"
        context: Reference context for understanding
        llm: Language model
        prompt_template: Prompt template
        max_retries: Maximum number of retry attempts
        
    Returns:
        Polished text
    """
    if pd.isna(text) or text is None or str(text).strip() == "":
        return text
    
    for attempt in range(max_retries):
        try:
            # Create the prompt
            prompt = prompt_template.format_messages(
                content_type=content_type,
                original_text=text,
                context=context[:500] if context else "No context available"  # Limit context length
            )
            
            # Get LLM response
            response = llm.invoke(prompt)
            polished = response.content.strip()
            
            # Basic validation
            if polished and len(polished) > 0:
                return polished
            else:
                print(f"   ⚠️  Empty response on attempt {attempt + 1}, retrying...")
                time.sleep(1)
                
        except Exception as e:
            print(f"   ⚠️  Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"   ⚠️  Max retries reached, returning original text")
                return text
    
    return text


def polish_dataset(df, llm, prompt_template, batch_size=5):
    """
    Polish all questions and answers in the dataset.
    
    Args:
        df: DataFrame with questions and answers
        llm: Language model
        prompt_template: Prompt template
        batch_size: Number of items to process before saving checkpoint
        
    Returns:
        DataFrame with polished questions and answers
    """
    print("\n" + "="*80)
    print("POLISHING DATASET")
    print("="*80)
    
    # Detect column names
    question_col, answer_col, context_col = detect_column_names(df)
    
    print(f"\n📋 Detected columns:")
    print(f"   Question column: {question_col}")
    print(f"   Answer column: {answer_col}")
    print(f"   Context column: {context_col}")
    
    if not question_col:
        raise ValueError(f"Could not find question column. Available columns: {df.columns.tolist()}")
    if not answer_col:
        raise ValueError(f"Could not find answer column. Available columns: {df.columns.tolist()}")
    
    # Create new columns for polished content
    df['polished_question'] = df[question_col].copy()
    df['polished_answer'] = df[answer_col].copy()
    
    total_items = len(df) * 2  # Questions + Answers
    
    print(f"\n📊 Processing {len(df)} Q&A pairs ({total_items} items total)")
    
    with tqdm(total=total_items, desc="Polishing") as pbar:
        for idx, row in df.iterrows():
            # Get context safely
            context = safe_get_context(row, context_col)
            
            # Polish question
            if not pd.isna(row[question_col]):
                polished_q = polish_text(
                    text=row[question_col],
                    content_type="question",
                    context=context,
                    llm=llm,
                    prompt_template=prompt_template
                )
                df.at[idx, 'polished_question'] = polished_q
            pbar.update(1)
            
            # Polish answer
            if not pd.isna(row[answer_col]):
                polished_a = polish_text(
                    text=row[answer_col],
                    content_type="answer",
                    context=context,
                    llm=llm,
                    prompt_template=prompt_template
                )
                df.at[idx, 'polished_answer'] = polished_a
            pbar.update(1)
            
            # Save checkpoint every batch_size items
            if (idx + 1) % batch_size == 0:
                checkpoint_file = "checkpoint_polished_dataset.csv"
                df.to_csv(checkpoint_file, index=False, encoding='utf-8')
                print(f"\n   💾 Checkpoint saved at {idx + 1}/{len(df)} items")
    
    print("\n✅ Polishing completed!")
    return df, question_col, answer_col, context_col


def compare_changes(df, question_col, answer_col, sample_size=5):
    """
    Display comparison between original and polished text.
    """
    print("\n" + "="*80)
    print("SAMPLE COMPARISONS (Original vs Polished)")
    print("="*80)
    
    sample_df = df.head(sample_size)
    
    for idx, row in sample_df.iterrows():
        print(f"\n--- Example {idx + 1} ---")
        
        # Question comparison
        if row[question_col] != row['polished_question']:
            print(f"\n❌ ORIGINAL Q: {row[question_col]}")
            print(f"✅ POLISHED Q: {row['polished_question']}")
        else:
            print(f"\n✓ Question unchanged: {row[question_col]}")
        
        # Answer comparison
        original_ans = str(row[answer_col])[:200]
        polished_ans = str(row['polished_answer'])[:200]
        
        if row[answer_col] != row['polished_answer']:
            print(f"\n❌ ORIGINAL A: {original_ans}...")
            print(f"✅ POLISHED A: {polished_ans}...")
        else:
            print(f"\n✓ Answer unchanged: {original_ans}...")


def create_final_dataset(df, question_col, answer_col):
    """
    Create final dataset with polished content replacing original.
    """
    print("\n📋 Creating final polished dataset...")
    
    # Replace original columns with polished versions
    df[question_col] = df['polished_question']
    df[answer_col] = df['polished_answer']
    
    # Remove temporary polished columns
    df = df.drop(columns=['polished_question', 'polished_answer'])
    
    print("✅ Final dataset created")
    return df


def upload_to_huggingface(df, hf_token, hf_repo_id, suffix="_polished"):
    """
    Upload polished dataset to Hugging Face Hub.
    """
    try:
        # Create new repo ID with suffix
        polished_repo_id = f"{hf_repo_id}{suffix}"
        
        print(f"\n🚀 Uploading polished dataset to Hugging Face: {polished_repo_id}")
        
        dataset = Dataset.from_pandas(df)
        
        dataset.push_to_hub(
            polished_repo_id,
            token=hf_token,
            private=False
        )
        
        print(f"✅ Successfully uploaded to https://huggingface.co/datasets/{polished_repo_id}")
        
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
        model_name = os.getenv("POLISH_MODEL_NAME", "gpt-4o")  # Model for polishing
        
        # Validate required environment variables
        if not openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable")
        if not hf_repo_id:
            raise ValueError("Missing HF_REPO_ID environment variable. Please specify the dataset to polish.")
        
        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        print("="*80)
        print("HUGGING FACE DATASET POLISHER")
        print("="*80)
        print(f"\n📦 Source Dataset: {hf_repo_id}")
        print(f"🤖 Model: {model_name}")
        
        # Step 1: Download dataset from Hugging Face
        print(f"\n⬇️  Downloading dataset from Hugging Face...")
        dataset = load_dataset(hf_repo_id, split='train')
        df = dataset.to_pandas()
        print(f"✅ Downloaded {len(df)} samples")
        print(f"   Columns: {list(df.columns)}")
        
        # Show first few rows to understand the data structure
        print(f"\n📋 Dataset preview:")
        print(df.head(2))
        
        # Step 2: Setup LLM for polishing
        print(f"\n🤖 Setting up LLM for polishing...")
        llm = ChatOpenAI(model=model_name, temperature=0.1)  # Low temperature for consistency
        prompt_template = create_polishing_prompt()
        print("✅ LLM ready")
        
        # Step 3: Polish the dataset
        df_polished, question_col, answer_col, context_col = polish_dataset(df, llm, prompt_template)
        
        # Step 4: Show comparisons
        compare_changes(df_polished, question_col, answer_col, sample_size=5)
        
        # Step 5: Create final dataset
        df_final = create_final_dataset(df_polished, question_col, answer_col)
        
        # Step 6: Save locally
        output_filename = "polished_qa_dataset.csv"
        df_final.to_csv(output_filename, index=False, encoding='utf-8')
        print(f"\n💾 Saved polished dataset to {output_filename}")
        
        # Step 7: Upload to Hugging Face (optional)
        if hf_token:
            upload_to_huggingface(df_final, hf_token, hf_repo_id, suffix="_polished")
        
        # Step 8: Summary
        print("\n" + "="*80)
        print("✨ POLISHING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\n📁 Files created:")
        print(f"   - {output_filename}")
        if hf_token:
            print(f"\n🌐 Polished Dataset: https://huggingface.co/datasets/{hf_repo_id}_polished")
        
        # Statistics
        print(f"\n📊 Statistics:")
        print(f"   Total samples: {len(df_final)}")
        print(f"   Columns: {list(df_final.columns)}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()