![Queryome Banner](extras/Queryome_banner.png)

**Queryome: Orchestrating Retrieval, Reasoning, and Synthesis across Biomedical Literature**

Queryome is a multi-agent deep research system for biomedical literature. It combines semantic vector search, lexical keyword search, and LLM-powered reasoning to provide comprehensive answers to complex biomedical research questions.
This repository contains the command-line version of Queryome; a desktop app is available at https://www.queryome.app/.

## Features

- **Hybrid Search**: Combines FAISS vector search with BM25 keyword search for optimal retrieval
- **Multi-Agent Architecture**: Uses specialized agents (PI Agent, SubAgent Team, Synthesizer) for planning, execution, and synthesis
- **Multiple Search Indices**: Searches across title/abstract, author keywords, and MeSH terms
- **Comprehensive Reports**: Generates well-structured research reports with proper citations

## Installation

### 1. Set up Environment Variables

Export your OpenAI API key:

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

You can add this to your `~/.bashrc` or `~/.zshrc` for persistence.

### 2. Download Indices

Download the pre-built search indices:

```bash
wget https://kiharalab.org/queryome/indices.tar.gz
tar -xzf indices.tar.gz
```

This will create an `indices/` directory containing:
- `vector_db/` - FAISS vector index and SQLite database
  - `faiss.index` - FAISS vector index
  - `articles.db` - SQLite database with article metadata
- `bm25_title_abstract/` - BM25 index for title and abstract search
- `bm25_author_keywords/` - BM25 index for author keywords search
- `bm25_mesh_terms/` - BM25 index for MeSH terms search

### 3. Create Conda Environment

Create and activate a new conda environment:

```bash
conda create -n queryome python=3.10
conda activate queryome
```

Install the required dependencies:

```bash
pip install openai numpy torch faiss-cpu bm25s PyStemmer sentence-transformers numba
```

## Usage

### Command Line Interface

**Interactive Mode:**
```bash
python queryome_cli.py
```

**Single Query:**
```bash
python queryome_cli.py --query "What are the latest treatments for Type 2 diabetes?"
```

**With Custom Log Directory:**
```bash
python queryome_cli.py --query "Efficacy of metformin in elderly patients" --log-dir ./my_logs
```

### Python API

**Single Query:**
```python
from queryome import Queryome

queryome = Queryome()
result = queryome.research("What are the latest treatments for Type 2 diabetes?")
print(result)
```

**Multiple Queries (Batch Processing):**
```python
from queryome import Queryome, batch_research

# Using the Queryome class
queryome = Queryome()
queries = [
    "Efficacy of metformin in elderly patients",
    "Side effects of insulin therapy",
    "Latest diabetes management guidelines"
]
results = queryome.research_multiple(queries)

for r in results:
    print(f"Query: {r['query']}")
    print(f"Result: {r['result']}")
    print("---")

# Or use the convenience function
results = batch_research(queries)
```

### Configuration Options

When initializing Queryome, you can customize:

```python
queryome = Queryome(
    log_dir="./custom_logs",           # Custom log directory
    enable_search_engine=True,          # Enable/disable search engine
    openai_api_key="your-key",          # Provide API key programmatically
    embedding_device="cuda:0"           # GPU device for embeddings
)
```

## Benchmark Results
All Queryome's benchmark data as seen at the paper can be downloaded from https://kiharalab.org/queryome/benchmark_data.tar.gz

## Authors

**Pranav Punuru**<sup>1</sup>, **Nabil Ibtehaz**<sup>2</sup>, **Swagarika Giri**<sup>2</sup>, **Harsha Srirangam**<sup>2</sup>, **Emilia Tugolukova**<sup>1</sup>, and **Daisuke Kihara**<sup>1,2,*</sup>

<sup>1</sup> Department of Biological Sciences, Purdue University, West Lafayette, IN 47906, USA
<sup>2</sup> Department of Computer Science, Purdue University, West Lafayette, IN 47906, USA

**\* Corresponding Author**
Email: dkihara@purdue.edu

## Citation

If you use Queryome in your research, please cite:

```bibtex
@article{punuru2025queryome,
  title={Queryome: Orchestrating Retrieval, Reasoning, and Synthesis across Biomedical Literature},
  author={Punuru, Pranav and Ibtehaz, Nabil and Giri, Swagarika and Srirangam, Harsha and Tugolukova, Emilia and Kihara, Daisuke},
  year={2025}
}
```

## License

GPL v3. (If you are interested in a different license, for example, for commercial use, please contact us (dkihara@purdue.edu).)

## Contact

For questions or support, please contact:
- Email: dkihara@purdue.edu
- Lab Website: https://kiharalab.org

For technical problems or questions, please reach to Pranav Punuru (ppunuru@purdue.edu).
