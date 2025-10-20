import os
import asyncio
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import llm_factory
from ragas.testset import SingleTurnSample, EvaluationDataset



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



evaluator_llm = llm_factory()


sample1 = SingleTurnSample(
    user_input="What is the capital of Germany?",
    retrieved_contexts=["Berlin is the capital and largest city of Germany. It has a population of over 3.7 million people."],
    response="The capital of Germany is Berlin.",
    reference="Berlin is the capital of Germany.", # Ground truth answer
)



sample2 = SingleTurnSample(
    user_input="Who wrote 'Pride and Prejudice'?",
    retrieved_contexts=["'Pride and Prejudice' is a romantic novel of manners written by Jane Austen, published in 1813."],
    response="'Pride and Prejudice' was written by Jane Austen.",
    reference="Jane Austen", # Ground truth can be concise
)


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
    response="The Taj Mahal is a famous landmark in New Delhi, India.", # Hallucinated response
    reference="The Taj Mahal is located in Agra, India.",
)

# Create an EvaluationDataset from the SingleTurnSample instances
dataset = EvaluationDataset(samples=[sample1, sample2, sample3, sample4])



metrics_to_evaluate = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]



async def run_evaluation():
    """
    Runs the Ragas evaluation and prints the results.
    """
    print("Running Ragas evaluation...")
    # The evaluation is an asynchronous operation
    result = await evaluate(
        dataset, metrics=metrics_to_evaluate, llm=evaluator_llm
    )
    print("Evaluation complete.")
    return result


if __name__ == "__main__":
    # Running the async function
    evaluation_result = asyncio.run(run_evaluation())

    print("\nEvaluation Results:")
    print(evaluation_result)

    # For a more structured view, you can convert the results to a pandas DataFrame
    try:
        df = evaluation_result.to_pandas()
        print("\nEvaluation Results (DataFrame):")
        print(df.to_string())
    except ImportError:
        print("\nInstall pandas (`pip install pandas`) to display results as a DataFrame.")
