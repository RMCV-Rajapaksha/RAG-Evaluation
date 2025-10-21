from ragas.testset.graph import Node, KnowledgeGraph
from ragas.testset.transforms.extractors import NERExtractor
from ragas.testset.transforms.relationship_builders.traditional import JaccardSimilarityBuilder
from ragas.testset.transforms import apply_transforms
from ragas.testset.synthesizers.base_query import QuerySynthesizer
from ragas.testset.synthesizers import SingleTurnSample

import asyncio
import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError("python-dotenv is required to load .env files. Install with `pip install python-dotenv`.")


@dataclass
class EntityQuerySynthesizer(QuerySynthesizer):

    async def _generate_scenarios(self, n, knowledge_graph, callbacks):
        """
        Logic to query nodes with entity
        Logic describing how to combine nodes, styles, length, persona to form n scenarios
        """
        # TODO: Implement scenario generation logic
        scenarios = []
        return scenarios

    async def _generate_sample(self, scenario, callbacks):
        """
        Logic on how to use transform each scenario to EvalSample (Query, Context, Reference)
        You may create singleturn or multiturn sample
        """
        # TODO: Implement sample generation logic
        query = "Sample query"
        contexts = ["Sample context"]
        reference = "Sample reference"
        
        return SingleTurnSample(
            user_input=query, 
            reference_contexts=contexts, 
            reference=reference
        )




async def main():
    # Load environment variables
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path)

    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
        raise ValueError(
            "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or add it to a .env file."
        )

    # Create sample nodes
    sample_nodes = [
        Node(
            properties={
                "page_content": "Einstein's theory of relativity revolutionized our understanding of space and time. It introduced the concept that time is not absolute but can change depending on the observer's frame of reference."
            }
        ), 
        Node(
            properties={
                "page_content": "Time dilation occurs when an object moves close to the speed of light, causing time to pass slower relative to a stationary observer. This phenomenon is a key prediction of Einstein's special theory of relativity."
            }
        )
    ]

    # Create knowledge graph
    kg = KnowledgeGraph(nodes=sample_nodes)

    # Define transforms
    extractor = NERExtractor()
    rel_builder = JaccardSimilarityBuilder(
        property_name="entities", 
        key_name="MISC",  # Changed from PER to MISC to match extracted entities
        new_property_name="entity_jaccard_similarity"
    )
    
    transforms = [extractor, rel_builder]

    # Apply all transforms to the knowledge graph
    await apply_transforms(kg, transforms)

    # Instantiate EntityQuerySynthesizer
    synthesizer = EntityQuerySynthesizer()

    # Call _generate_scenarios
    n = 5  # Number of scenarios to generate
    callbacks = None  # Replace with actual callbacks if needed
    scenarios = await synthesizer._generate_scenarios(n, kg, callbacks)

    print("\n-----------------------------Generated Scenarios-----------------------------")
    for i, scenario in enumerate(scenarios):
        print(f"Scenario {i+1}: {scenario}")

if __name__ == "__main__":
    asyncio.run(main())