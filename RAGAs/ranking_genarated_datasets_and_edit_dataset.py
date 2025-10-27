import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from datasets import load_dataset, Dataset
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import time
from tqdm import tqdm


# ------------------ CONFIGURATION ------------------
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv(find_dotenv())


# ------------------ PROMPTS ------------------

def create_sync_prompt():
    """
    Prompt for synchronizing Q&A with context.
    """
    system_message = """
You are an expert dataset curator.

Your task:
- Review the given question (user_input) and answer (reference) using the reference_contexts.
- If the question or answer does NOT align with the context, modify or regenerate them.
- Ensure both question and answer are directly based on and answerable from the context.
- Keep them natural, clear, and concise.
- Preserve the same meaning when possible, but fix factual mismatches.
- Output must include both 'question' and 'answer' clearly labeled.

Return only in this format:
Question: <updated question>
Answer: <updated answer>
"""

    user_template = """Context:
{context}

Original Question: {question}
Original Answer: {answer}

Now verify and fix them if needed.
"""

    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", user_template)
    ])


def create_polish_prompt():
    """
    Prompt for polishing text.
    """
    system_message = """
You are an expert editor specializing in polishing question-answer (QA) pairs.

Fix:
1. Grammar, spelling, and punctuation
2. Capitalization and proper names
3. Consistency and readability
4. WSO2 product names:
   - WSO2 API Manager
   - WSO2 Identity Server
   - WSO2 Integrator
   - WSO2 Micro Integrator
   - Choreo
   - WSO2 API Platform for Kubernetes

Do not change meaning or add information.
Return only the polished text.
"""

    user_template = """Polish the following {content_type}:

Original {content_type}:
{original_text}

Context (for understanding only):
{context}

Polished {content_type}:"""

    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", user_template)
    ])


# ------------------ CORE FUNCTIONS ------------------

def sync_with_context(row, llm, prompt_template):
    """
    Synchronize question and answer with context using LLM.
    """
    context = row.get("reference_contexts", "")
    question = row.get("user_input", "")
    answer = row.get("reference", "")

    prompt = prompt_template.format_messages(
        context=context[:1500],
        question=question,
        answer=answer
    )

    for _ in range(3):
        try:
            response = llm.invoke(prompt)
            text = response.content.strip()
            if "Question:" in text and "Answer:" in text:
                q = text.split("Question:")[1].split("Answer:")[0].strip()
                a = text.split("Answer:")[1].strip()
                return q, a
        except Exception as e:
            print(f"⚠️ Sync error: {e}")
            time.sleep(2)
    return question, answer


def polish_text(text, content_type, context, llm, prompt_template):
    """
    Polish question or answer using LLM.
    """
    if not text or pd.isna(text):
        return text

    prompt = prompt_template.format_messages(
        content_type=content_type,
        original_text=text,
        context=context[:800]
    )

    for _ in range(3):
        try:
            response = llm.invoke(prompt)
            polished = response.content.strip()
            if polished:
                return polished
        except Exception as e:
            print(f"⚠️ Polishing error: {e}")
            time.sleep(2)
    return text


def process_dataset(df, llm_sync, llm_polish, sync_prompt, polish_prompt):
    """
    Synchronize and polish dataset.
    """
    results = []
    print("\n🔄 Synchronizing and polishing dataset...\n")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        context = row.get("reference_contexts", "")
        # Step 1: Sync
        synced_q, synced_a = sync_with_context(row, llm_sync, sync_prompt)

        # Step 2: Polish
        polished_q = polish_text(synced_q, "question", context, llm_polish, polish_prompt)
        polished_a = polish_text(synced_a, "answer", context, llm_polish, polish_prompt)

        results.append({
            "user_input": polished_q,
            "reference": polished_a,
            "reference_contexts": context,
            "synthesizer_name": row.get("synthesizer_name", "")
        })
    return pd.DataFrame(results)


def upload_to_huggingface(df, hf_token, hf_repo_id):
    """
    Upload final dataset to Hugging Face.
    """
    new_repo_id = f"{hf_repo_id}_synced_polished"
    dataset = Dataset.from_pandas(df)

    print(f"\n🚀 Uploading to Hugging Face: {new_repo_id} ...")
    dataset.push_to_hub(new_repo_id, token=hf_token, private=False)
    print(f"✅ Uploaded: https://huggingface.co/datasets/{new_repo_id}")


# ------------------ MAIN ------------------

def main():
    try:
        hf_token = os.getenv("HF_TOKEN")
        hf_repo_id = os.getenv("HF_REPO_ID")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not all([hf_token, hf_repo_id, openai_api_key]):
            raise ValueError("Missing environment variables! (HF_TOKEN, HF_REPO_ID, OPENAI_API_KEY)")

        os.environ["OPENAI_API_KEY"] = openai_api_key

        print("=" * 80)
        print("🧠 QA DATASET SYNC + POLISH PIPELINE")
        print("=" * 80)

        print(f"\n⬇️ Downloading dataset: {hf_repo_id}")
        dataset = load_dataset(hf_repo_id, split="train")
        df = dataset.to_pandas()
        print(f"✅ Loaded {len(df)} records\n")

        # Create LLMs
        llm_sync = ChatOpenAI(model="gpt-4o", temperature=0.2)
        llm_polish = ChatOpenAI(model="gpt-4o", temperature=0.1)
        sync_prompt = create_sync_prompt()
        polish_prompt = create_polish_prompt()

        # Process dataset
        df_final = process_dataset(df, llm_sync, llm_polish, sync_prompt, polish_prompt)

        # Save locally
        output_file = "synced_polished_dataset.csv"
        df_final.to_csv(output_file, index=False, encoding="utf-8")
        print(f"\n💾 Saved locally: {output_file}")

        # Upload
        upload_to_huggingface(df_final, hf_token, hf_repo_id)

        print("\n✅ Completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
