# RAG-Evaluation

A comprehensive toolkit for evaluating Retrieval-Augmented Generation (RAG) systems using the Ragas framework. This project provides tools for generating test datasets, evaluating RAG system performance, and analyzing results.

## Features

- **Test Dataset Generation**: Create diverse question-answer datasets from your vector store
- **RAG System Evaluation**: Evaluate your RAG system using multiple metrics:
  - Faithfulness
  - Answer Relevancy
  - Context Precision
  - Context Recall
- **Hugging Face Integration**: Easily upload and share your evaluation datasets
- **Knowledge Graph Generation**: Create and enrich knowledge graphs from your document chunks
- **Multiple Query Types Support**:
  - Single-hop Specific queries (50%)
  - Multi-hop Abstract queries (25%)
  - Multi-hop Specific queries (25%)

## Prerequisites

- Python 3.13 or higher
- PostgreSQL database with vector store capabilities
- OpenAI API key
- Hugging Face account and API token

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RMCV-Rajapaksha/RAG-Evaluation.git
cd RAG-Evaluation
```

2. Create a Python virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
uv install
```

## Configuration

Create a `.env` file in the project root with the following variables:

```env
CONNECTION_STRING=your_postgres_connection_string
DB_TABLE_NAME=your_vector_store_table
OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_huggingface_token
HF_REPO_ID=your_huggingface_repo_id
MODEL_NAME=gpt-4o-mini  # or your preferred model
TESTSET_SIZE=10  # number of test cases to generate
```

## Usage

### 1. Generate Test Dataset

To create a new test dataset from your vector store:

```bash
python create_question_and_answer_dataset.py
```

This will:
- Load chunks from your vector store
- Create and enrich a knowledge graph
- Generate diverse test questions
- Save the dataset locally and upload to Hugging Face

### 2. Evaluate RAG System

To evaluate your RAG system using the generated test dataset:

```bash
python evaluation_matrics.py
```

This will:
- Run evaluation using multiple metrics
- Generate detailed results for each sample
- Calculate aggregate scores
- Save results to CSV

## Project Structure

```
├── create_question_and_answer_dataset.py  # Test dataset generation
├── evaluation_matrics.py                  # RAG system evaluation
├── main.py                               # Project entry point
├── pyproject.toml                        # Project dependencies
├── notebook/                             # Jupyter notebooks
│   └── giskard.ipynb                    # Interactive examples
└── README.md                            # Project documentation
```

## Dependencies

- dotenv: Environment variable management
- jupyter: Notebook support
- llama-index-core: RAG functionality
- llama-index-vector-stores-postgres: Vector store integration
- psycopg2-binary: PostgreSQL database adapter
- ragas: RAG evaluation framework
- rapidfuzz: Fuzzy string matching
- sqlalchemy: Database ORM

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Contact

Created by [@RMCV-Rajapaksha](https://github.com/RMCV-Rajapaksha)
