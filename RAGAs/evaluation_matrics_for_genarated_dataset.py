import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from datasets import load_dataset
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import context_precision, context_recall
from ragas import SingleTurnSample, EvaluationDataset


# -------------------------------------------------------------------
# STEP 1: Load environment variables
# -------------------------------------------------------------------
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_REPO_ID = os.getenv("HF_REPO_ID")  # Example: "yourusername/yourdataset"

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment.")
if not HF_REPO_ID:
    raise ValueError("Missing HF_REPO_ID in environment (e.g., 'HF_REPO_ID=yourusername/yourdataset').")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# -------------------------------------------------------------------
# STEP 2: Load dataset from Hugging Face
# -------------------------------------------------------------------
print("=" * 80)
print("⬇️  LOADING DATASET FROM HUGGING FACE")
print("=" * 80)
print(f"📦 Dataset ID: {HF_REPO_ID}")

dataset = load_dataset(HF_REPO_ID, split="train")
df = dataset.to_pandas()

print(f"✅ Loaded {len(df)} samples")
print(f"📋 Columns: {list(df.columns)}")

# -------------------------------------------------------------------
# STEP 3: Detect columns (question, answer, context)
# -------------------------------------------------------------------
def detect_column_names(df):
    columns = df.columns.tolist()
    question_col = next((c for c in ['question', 'user_input', 'query', 'prompt'] if c in columns), None)
    answer_col = next((c for c in ['answer', 'response', 'output', 'reference', 'ground_truth'] if c in columns), None)
    context_col = next((c for c in ['context', 'retrieved_contexts', 'reference_contexts', 'contexts'] if c in columns), None)

    print("\n🧩 Detected columns:")
    print(f"   Question column: {question_col}")
    print(f"   Answer column:   {answer_col}")
    print(f"   Context column:  {context_col}")

    if not question_col or not answer_col or not context_col:
        raise ValueError("❌ Could not detect all required columns. Please ensure dataset has question, answer, and context columns.")
    return question_col, answer_col, context_col


question_col, answer_col, context_col = detect_column_names(df)


# -------------------------------------------------------------------
# STEP 4: Convert to RAGAS format (EvaluationDataset)
# -------------------------------------------------------------------
def to_ragas_samples(df, q_col, a_col, ctx_col):
    samples = []
    for _, row in df.iterrows():
        user_input = str(row[q_col])
        response = str(row[a_col])
        reference = str(row[a_col])  # Use same as ground truth if not available

        context_value = row[ctx_col]
        if isinstance(context_value, list):
            retrieved_contexts = [str(c) for c in context_value]
        else:
            retrieved_contexts = [str(context_value)]

        samples.append(
            SingleTurnSample(
                user_input=user_input,
                retrieved_contexts=retrieved_contexts,
                response=response,
                reference=reference,
            )
        )
    return samples


print("\n🧠 Preparing data for RAGAS evaluation...")
samples = to_ragas_samples(df, question_col, answer_col, context_col)
evaluation_dataset = EvaluationDataset(samples=samples)
print(f"✅ Prepared {len(samples)} samples for evaluation")


# -------------------------------------------------------------------
# STEP 5: Initialize LLM and define metrics
# -------------------------------------------------------------------
print("\n🤖 Initializing evaluator LLM...")
evaluator_llm = llm_factory()

metrics_to_evaluate = [context_precision, context_recall]

print(f"✅ Metrics to evaluate: {[m.name for m in metrics_to_evaluate]}")


# -------------------------------------------------------------------
# STEP 6: Run evaluation
# -------------------------------------------------------------------
print("\n🚀 Running RAGAS evaluation...")
results = evaluate(evaluation_dataset, metrics=metrics_to_evaluate, llm=evaluator_llm)
print("✅ Evaluation complete!")


# -------------------------------------------------------------------
# STEP 7: Display results
# -------------------------------------------------------------------
try:
    df_results = results.to_pandas()

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 60)

    print("\n" + "=" * 80)
    print("📊 RAGAS RESULTS (Context Precision & Recall)")
    print("=" * 80)

    for i, row in df_results.iterrows():
        print(f"\n🔹 Sample {i + 1}:")
        print(f"   Question: {row['user_input']}")
        print(f"   Response: {row['response']}")
        print(f"   Context Precision: {row['context_precision']:.4f}")
        print(f"   Context Recall:    {row['context_recall']:.4f}")
        print("-" * 60)

    print("\n📈 AVERAGE SCORES:")
    print("-" * 60)
    print(f"   Context Precision: {df_results['context_precision'].mean():.4f}")
    print(f"   Context Recall:    {df_results['context_recall'].mean():.4f}")
    print("=" * 80)

    # Save to CSV
    output_csv = "ragas_context_metrics.csv"
    df_results.to_csv(output_csv, index=False)
    print(f"\n💾 Results saved to: {output_csv}")

except Exception as e:
    print(f"⚠️ Could not format results: {e}")
    print(results)
