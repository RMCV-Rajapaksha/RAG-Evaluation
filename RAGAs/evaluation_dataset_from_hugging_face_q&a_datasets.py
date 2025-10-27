from ragas.testset.graph import Node, KnowledgeGraph
from ragas.testset.transforms.extractors import NERExtractor
from ragas.testset.transforms.relationship_builders.traditional import JaccardSimilarityBuilder
from ragas.testset.transforms import apply_transforms
from ragas import QuerySynthesizer
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
import asyncio
import os
from dataclasses import dataclass, field
from typing import List
import random
import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi
from ragas.testset import QuerySynthesizer


try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError("python-dotenv is required. Install with `pip install python-dotenv`.")


@dataclass
class EntityQuerySynthesizer(QuerySynthesizer):
    """Custom synthesizer to generate Q&A pairs based on entities in the knowledge graph."""
    
    llm: LangchainLLMWrapper = field(default=None)
    
    def __post_init__(self):
        """Initialize the LLM if not provided."""
        if self.llm is None:
            base_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            self.llm = LangchainLLMWrapper(langchain_llm=base_llm)

    async def _generate_scenarios(self, n, knowledge_graph, callbacks):
        """
        Generate scenarios by selecting nodes with entities.
        Each scenario represents a node with extracted entities.
        """
        scenarios = []
        nodes_with_entities = []
        
        # Filter nodes that have entities
        for node in knowledge_graph.nodes:
            if hasattr(node, 'properties') and 'entities' in node.properties:
                entities = node.properties.get('entities', {})
                if entities:  # Only include nodes with non-empty entities
                    nodes_with_entities.append(node)
        
        if not nodes_with_entities:
            print("⚠️  No nodes with entities found. Using all nodes.")
            nodes_with_entities = knowledge_graph.nodes
        
        # Generate n scenarios (sample with replacement if needed)
        for i in range(n):
            if nodes_with_entities:
                selected_node = random.choice(nodes_with_entities)
                scenarios.append({
                    'node': selected_node,
                    'index': i + 1,
                    'entities': selected_node.properties.get('entities', {}),
                    'content': selected_node.properties.get('page_content', '')
                })
        
        print(f"✅ Generated {len(scenarios)} scenarios from {len(nodes_with_entities)} nodes with entities")
        return scenarios

    async def _generate_sample(self, scenario, callbacks):
        """
        Transform each scenario into a Q&A sample.
        Uses LLM to generate question, extract context, and create reference answer.
        """
        node = scenario['node']
        content = scenario['content']
        entities = scenario['entities']
        
        # Extract key entities for question generation
        entity_text = self._format_entities(entities)
        
        # Generate question using LLM
        question_prompt = f"""Based on the following text and entities, generate a clear and specific question that can be answered using the text.

Text: {content}

Entities: {entity_text}

Generate only the question without any additional text or explanation."""

        try:
            question_response = await self.llm.agenerate_text(prompt=question_prompt)
            question = question_response.strip()
        except Exception as e:
            print(f"⚠️  Error generating question: {e}")
            question = f"What information is provided about {entity_text}?"
        
        # Generate reference answer using LLM
        answer_prompt = f"""Based on the following text, provide a concise and accurate answer to the question.

Text: {content}

Question: {question}

Provide only the answer without repeating the question."""

        try:
            answer_response = await self.llm.agenerate_text(prompt=answer_prompt)
            reference = answer_response.strip()
        except Exception as e:
            print(f"⚠️  Error generating answer: {e}")
            reference = content[:200] + "..."  # Fallback to truncated content
        
        # Use the content as context
        contexts = [content]
        
        return SingleTurnSample(
            user_input=question, 
            reference_contexts=contexts, 
            reference=reference
        )
    
    def _format_entities(self, entities):
        """Format entities dictionary into readable text."""
        if not entities:
            return "general topics"
        
        formatted = []
        for entity_type, entity_list in entities.items():
            if entity_list:
                formatted.append(f"{entity_type}: {', '.join(entity_list[:3])}")  # Limit to 3 entities per type
        
        return "; ".join(formatted) if formatted else "general topics"


async def generate_qa_dataset(nodes: List[Node], n_samples: int = 10):
    """
    Generate Q&A dataset from nodes using Ragas.
    
    Args:
        nodes: List of Node objects with page_content
        n_samples: Number of Q&A pairs to generate
    
    Returns:
        List of dictionaries containing questions, contexts, and answers
    """
    # Create knowledge graph
    kg = KnowledgeGraph(nodes=nodes)
    
    # Define transforms
    extractor = NERExtractor()
    rel_builder = JaccardSimilarityBuilder(
        property_name="entities", 
        key_name="MISC",
        new_property_name="entity_jaccard_similarity"
    )
    
    transforms = [extractor, rel_builder]
    
    # Apply transforms to extract entities
    print("🔍 Extracting entities from knowledge graph...")
    await apply_transforms(kg, transforms)
    
    # Instantiate EntityQuerySynthesizer
    synthesizer = EntityQuerySynthesizer()
    
    # Generate scenarios
    print(f"📝 Generating {n_samples} scenarios...")
    scenarios = await synthesizer._generate_scenarios(n_samples, kg, None)
    
    # Generate samples for each scenario
    print("🤖 Generating Q&A pairs using LLM...")
    samples = []
    for i, scenario in enumerate(scenarios, 1):
        print(f"  Generating sample {i}/{len(scenarios)}...")
        sample = await synthesizer._generate_sample(scenario, None)
        samples.append({
            'question': sample.user_input,
            'context': sample.reference_contexts[0] if sample.reference_contexts else "",
            'answer': sample.reference,
            'entities': scenario.get('entities', {})
        })
    
    print(f"✅ Generated {len(samples)} Q&A pairs successfully!")
    return samples


def upload_to_huggingface(samples: List[dict], repo_id: str, token: str, private: bool = False):
    """
    Upload Q&A dataset to Hugging Face Hub.
    
    Args:
        samples: List of Q&A dictionaries
        repo_id: Hugging Face repository ID (username/dataset-name)
        token: Hugging Face API token
        private: Whether to make the dataset private
    """
    # Convert to DataFrame
    df = pd.DataFrame(samples)
    
    # Convert entities dict to string for storage
    df['entities'] = df['entities'].apply(str)
    
    print(f"\n📊 Dataset statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"\n📄 Sample data:")
    print(df.head(2).to_string())
    
    # Create Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    
    # Upload to Hugging Face
    print(f"\n🚀 Uploading to Hugging Face: {repo_id}")
    dataset.push_to_hub(
        repo_id=repo_id,
        token=token,
        private=private
    )
    
    print(f"✅ Successfully uploaded! View at: https://huggingface.co/datasets/{repo_id}")


async def main():
    # Load environment variables
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path)
    
    # Validate API keys
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment variables")
    
    hf_repo_id = os.getenv("HF_REPO_ID", "ChamaraVishwajithRajapaksha/RAG-QA-Dataset")
    
    # Create sample nodes (replace with your actual data)
    sample_nodes = [
        Node(properties={
            "page_content": "Einstein's theory of relativity revolutionized our understanding of space and time. It introduced the concept that time is not absolute but can change depending on the observer's frame of reference."
        }), 
        Node(properties={
            "page_content": "Time dilation occurs when an object moves close to the speed of light, causing time to pass slower relative to a stationary observer. This phenomenon is a key prediction of Einstein's special theory of relativity."
        }),
        Node(properties={
            "page_content": "Quantum mechanics describes the behavior of matter and energy at the atomic and subatomic scale. It introduced concepts like wave-particle duality and the uncertainty principle."
        }),
        Node(properties={
            "page_content": "The double-slit experiment demonstrates that light and matter can display characteristics of both classically defined waves and particles. This experiment is fundamental to understanding quantum mechanics."
        }),
        Node(properties={
            "page_content": "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape. They are predicted by Einstein's general theory of relativity."
        })
    ]
    
    # Generate Q&A dataset
    n_samples = 10  # Adjust as needed
    qa_samples = await generate_qa_dataset(sample_nodes, n_samples)
    
    # Display generated samples
    print("\n" + "="*80)
    print("GENERATED Q&A SAMPLES")
    print("="*80)
    for i, sample in enumerate(qa_samples[:3], 1):  # Show first 3
        print(f"\n--- Sample {i} ---")
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer']}")
        print(f"Context: {sample['context'][:100]}...")
    
    # Upload to Hugging Face
    upload_to_huggingface(
        samples=qa_samples,
        repo_id=hf_repo_id,
        token=hf_token,
        private=False  # Set to True for private dataset
    )


if __name__ == "__main__":
    asyncio.run(main())