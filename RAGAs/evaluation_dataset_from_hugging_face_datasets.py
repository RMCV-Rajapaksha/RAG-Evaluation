from datasets import load_dataset
from ragas import EvaluationDataset


# Ensure that the dataset contains the necessary fields for evaluation, such as user inputs, retrieved contexts, responses, and references.
dataset = load_dataset("explodinggradients/amnesty_qa","english_v3")

# Load the dataset into a Ragas EvaluationDataset object.
eval_dataset = EvaluationDataset.from_hf_dataset(dataset["eval"])

