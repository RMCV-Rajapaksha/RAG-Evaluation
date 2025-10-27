import os
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import llm_factory
from ragas import SingleTurnSample, EvaluationDataset

try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError("python-dotenv is required to load .env files. Install with `pip install python-dotenv`.")

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
    raise ValueError(
        "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or add it to a .env file."
    )

# Initialize the evaluator LLM
evaluator_llm = llm_factory()

# Sample 1: Basic factual question
sample1 = SingleTurnSample(
    user_input="What is the capital of Germany?",
    retrieved_contexts=["Berlin is the capital and largest city of Germany. It has a population of over 3.7 million people."],
    response="The capital of Germany is Berlin.",
    reference="Berlin is the capital of Germany.",  # Ground truth answer
)

# Sample 2: Author identification
sample2 = SingleTurnSample(
    user_input="Who wrote 'Pride and Prejudice'?",
    retrieved_contexts=["'Pride and Prejudice' is a romantic novel of manners written by Jane Austen, published in 1813."],
    response="'Pride and Prejudice' was written by Jane Austen.",
    reference="Jane Austen",  # Ground truth can be concise
)

# Sample 3: Scientific question with extra context
sample3 = SingleTurnSample(
    user_input="What is the main function of the mitochondria?",
    retrieved_contexts=[
        "The mitochondrion is an organelle found in the cells of most eukaryotes. The primary function of mitochondria is to generate most of the cell's supply of adenosine triphosphate (ATP), used as a source of chemical energy. The sun is the primary source of energy for Earth."
    ],
    response="The main function of mitochondria is to produce ATP, which is the cell's energy currency.",
    reference="The primary role of mitochondria is cellular respiration and generating ATP.",
)

# Sample 4: A question where the response is not fully faithful to the context
sample4 = SingleTurnSample(
    user_input="Where is the Taj Mahal located?",
    retrieved_contexts=["The Taj Mahal is an ivory-white marble mausoleum on the south bank of the Yamuna river in the Indian city of Agra."],
    response="The Taj Mahal is a famous landmark in New Delhi, India.",  # Hallucinated response
    reference="The Taj Mahal is located in Agra, India.",
)

# Create an EvaluationDataset from the SingleTurnSample instances
dataset = EvaluationDataset(samples=[sample1, sample2, sample3, sample4])

# Define the metrics to evaluate
metrics_to_evaluate = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]

def run_evaluation():
    """
    Runs the Ragas evaluation and prints the results.
    """
    print("Running Ragas evaluation...")
    # The evaluation is now synchronous
    result = evaluate(
        dataset, metrics=metrics_to_evaluate, llm=evaluator_llm
    )
    print("Evaluation complete.")
    return result

if __name__ == "__main__":
    # Running the evaluation function
    evaluation_result = run_evaluation()

    print("\n" + "="*80)
    print("RAGAS EVALUATION RESULTS")
    print("="*80)

    # Convert to DataFrame for better formatting
    try:
        import pandas as pd
        df = evaluation_result.to_pandas()
        
        # Set pandas display options for better formatting
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        
        print("\n📊 DETAILED RESULTS BY SAMPLE:")
        print("-" * 80)
        
        # Display each sample's results
        for idx, row in df.iterrows():
            print(f"\n🔍 Sample {idx + 1}:")
            print(f"   Question: {row['user_input']}")
            print(f"   Response: {row['response'][:100]}..." if len(row['response']) > 100 else f"   Response: {row['response']}")
            print(f"\n   Metrics:")
            print(f"   • Faithfulness:       {row['faithfulness']:.4f}")
            print(f"   • Answer Relevancy:   {row['answer_relevancy']:.4f}")
            print(f"   • Context Precision:  {row['context_precision']:.4f}")
            print(f"   • Context Recall:     {row['context_recall']:.4f}")
            print("-" * 80)
        
        # Calculate and display aggregate scores
        print("\n📈 AGGREGATE SCORES (Average across all samples):")
        print("-" * 80)
        print(f"   Faithfulness:       {df['faithfulness'].mean():.4f}")
        print(f"   Answer Relevancy:   {df['answer_relevancy'].mean():.4f}")
        print(f"   Context Precision:  {df['context_precision'].mean():.4f}")
        print(f"   Context Recall:     {df['context_recall'].mean():.4f}")
        print("="*80)
        
        # Save to CSV
        output_file = "ragas_evaluation_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to: {output_file}")
        
    except ImportError:
        print("\n⚠️  Install pandas (`pip install pandas`) to display formatted results.")
        print("\nRaw Evaluation Results:")
        print(evaluation_result)