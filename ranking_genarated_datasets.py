import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import pandas as pd
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


# ------------------ SCORING PROMPT ------------------

def create_scoring_prompt():
    """
    Create a prompt template for scoring how well an answer matches the question using the given context.
    """
    system_message = """
You are an expert evaluator for question-answer datasets.

Your job is to determine how well the provided answer can be justified or derived 
from the given reference context with respect to the question.

Scoring criteria:
- 10: The answer is fully supported by the reference_context and directly addresses the question.
- 7–9: The answer is mostly supported; only minor details are missing.
- 4–6: The answer is partially supported or somewhat related.
- 1–3: The answer is mostly unsupported or unrelated.
- 0: The answer has no connection to the context or contradicts it.

Rules:
- Focus only on the relationship between question, answer, and context.
- Do not reward stylistic quality or grammar.
- Always return ONLY a number between 0 and 10.
"""

    user_template = """
Question: {question}

Answer: {answer}

Reference Context: {context}

Relevance Score (0–10):
"""

    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", user_template)
    ])


# ------------------ HELPER FUNCTIONS ------------------

def detect_column_names(df):
    """Detects columns for question, answer, and context."""
    columns = df.columns.tolist()
    question_col = next((c for c in ['user_input', 'question', 'query', 'questions'] if c in columns), None)
    answer_col = next((c for c in ['reference', 'ground_truth', 'answer', 'response', 'ground_truths'] if c in columns), None)
    context_col = next((c for c in ['reference_contexts', 'contexts', 'context'] if c in columns), None)
    return question_col, answer_col, context_col


def safe_get_context(row, context_col):
    """Safely extract context from various formats."""
    if not context_col:
        return ""
    try:
        value = row[context_col]
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        if isinstance(value, list):
            return " ".join([str(v) for v in value if v])
        return str(value)
    except Exception:
        return ""


def score_relevance(question, answer, context, llm, prompt_template, max_retries=3):
    """
    Use the LLM to score how well the answer fits the context for the given question.
    """
    if not question or not answer:
        return 0.0

    for attempt in range(max_retries):
        try:
            prompt = prompt_template.format_messages(
                question=question,
                answer=answer,
                context=context[:1000] if context else "No context"
            )
            response = llm.invoke(prompt)
            content = response.content.strip()
            try:
                score = float(content)
                if 0 <= score <= 10:
                    return score
            except ValueError:
                pass
            print(f"⚠️ Invalid score response ({content}), retrying...")
        except Exception as e:
            print(f"⚠️ Error on attempt {attempt+1}: {e}")
        time.sleep(1)
    return 0.0


# ------------------ MAIN LOGIC ------------------

def rank_dataset(df, llm, prompt_template):
    """Score and rank dataset by context relevance."""
    question_col, answer_col, context_col = detect_column_names(df)
    print(f"\nDetected columns:")
    print(f" - Question: {question_col}")
    print(f" - Answer: {answer_col}")
    print(f" - Context: {context_col}")

    if not (question_col and answer_col and context_col):
        raise ValueError("Missing one or more required columns (question, answer, context)")

    scores = []
    with tqdm(total=len(df), desc="Scoring relevance") as pbar:
        for idx, row in df.iterrows():
            q = str(row[question_col])
            a = str(row[answer_col])
            c = safe_get_context(row, context_col)
            score = score_relevance(q, a, c, llm, prompt_template)
            scores.append(score)
            pbar.update(1)

    df["relevance_score"] = scores
    df = df.sort_values(by="relevance_score", ascending=False).reset_index(drop=True)
    print("\n✅ Scoring completed. Top 5 samples:")
    print(df.head(5)[[question_col, answer_col, "relevance_score"]])
    return df


def upload_to_huggingface(df, hf_token, hf_repo_id, suffix="_ranked"):
    """Upload ranked dataset to Hugging Face Hub."""
    try:
        repo_id = f"{hf_repo_id}{suffix}"
        print(f"\n🚀 Uploading ranked dataset to Hugging Face: {repo_id}")
        dataset = Dataset.from_pandas(df)
        dataset.push_to_hub(repo_id, token=hf_token, private=False)
        print(f"✅ Uploaded to https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"❌ Upload error: {e}")
        import traceback; traceback.print_exc()


def main():
    try:
        hf_token = os.getenv("HF_TOKEN")
        hf_repo_id = os.getenv("HF_REPO_ID")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("EVAL_MODEL_NAME", "gpt-4o")

        if not openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY")
        if not hf_repo_id:
            raise ValueError("Missing HF_REPO_ID")

        os.environ["OPENAI_API_KEY"] = openai_api_key

        print("="*80)
        print("🎯 HUGGING FACE Q&A RELEVANCE SCORER")
        print("="*80)
        print(f"\n📦 Dataset: {hf_repo_id}")
        print(f"🤖 Model: {model_name}")

        dataset = load_dataset(hf_repo_id, split="train")
        df = dataset.to_pandas()
        print(f"✅ Loaded {len(df)} samples")

        llm = ChatOpenAI(model=model_name, temperature=0)
        prompt_template = create_scoring_prompt()

        df_ranked = rank_dataset(df, llm, prompt_template)

        output_file = "ranked_qa_dataset.csv"
        df_ranked.to_csv(output_file, index=False, encoding="utf-8")
        print(f"\n💾 Saved ranked dataset to {output_file}")

        # if hf_token:
        #     upload_to_huggingface(df_ranked, hf_token, hf_repo_id)

        print("\n✨ Done! Dataset ranked by context relevance.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
