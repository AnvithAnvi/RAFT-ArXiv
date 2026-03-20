# RAFT ArXiv: Retrieval-Augmented Fine-Tuning for Scientific Q&A

A complete implementation of RAFT (Retrieval-Augmented Fine-Tuning) for building a question-answering system over ArXiv papers. This project demonstrates the full pipeline from data collection to model deployment.

## 🎯 Overview

This project implements RAFT (Retrieval-Augmented Fine-Tuning) to create a specialized Q&A system for machine learning and AI research papers from ArXiv. The system combines:

- **Retrieval**: Semantic search over paper abstracts using ChromaDB and sentence transformers
- **Augmentation**: Context-aware answer generation using retrieved documents
- **Fine-Tuning**: QLoRA fine-tuning of Phi-2 model on synthetic RAFT dataset
- **API**: FastAPI server with web interface for model comparison

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ArXiv Papers  │───▶│  RAFT Dataset   │───▶│  Fine-tuned     │
│   (Raw Data)    │    │  (Synthetic Q&A)│    │  Phi-2 Model    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Text Chunks   │    │   Base RAG      │    │   RAFT Model    │
│   (ChromaDB)    │    │   (Llama 3)     │    │   (Phi-2)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
                                            ┌─────────────────┐
                                            │   FastAPI        │
                                            │   Web Interface  │
                                            └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Ollama (for dataset generation)
- 16GB+ RAM recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/raft-arxiv.git
   cd raft-arxiv
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Ollama and pull Llama 3**
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull llama3
   ```

### Data Pipeline

Run the complete pipeline in sequence:

1. **Fetch and chunk papers**
   ```bash
   python src/fetch_papers.py
   ```

2. **Generate RAFT dataset**
   ```bash
   python src/build_raft_dataset.py
   ```

3. **Fine-tune Phi-2 model**
   ```bash
   python src/finetune_qlora.py
   ```

4. **Build baseline RAG**
   ```bash
   python src/build_baseline_rag.py
   ```

5. **Evaluate models**
   ```bash
   python eval/evaluate.py
   ```

### API Server

Start the FastAPI server:

```bash
python api/serve.py
```

Or using Docker:

```bash
docker build -t raft-arxiv .
docker run -p 8000:8000 raft-arxiv
```

Visit `http://localhost:8000` for the web interface.

## 📁 Project Structure

```
raft-arxiv/
├── api/                          # FastAPI server
│   ├── serve.py                 # Main API endpoints
│   └── static/                  # Web interface
│       └── index.html
├── src/                         # Core pipeline scripts
│   ├── fetch_papers.py          # ArXiv data collection
│   ├── build_raft_dataset.py    # Synthetic dataset generation
│   ├── finetune_qlora.py        # Model fine-tuning
│   └── build_baseline_rag.py    # RAG system setup
├── eval/                        # Evaluation framework
│   ├── evaluate.py              # Model comparison
│   └── llm_judge.py             # Automated evaluation
├── data/                        # Generated data (gitignored)
│   ├── raw/                     # Raw paper metadata
│   ├── processed/               # Text chunks
│   ├── raft_dataset/            # Training data
│   ├── raft_model/              # Fine-tuned model
│   └── chroma_db/               # Vector database
├── Dockerfile                   # Container configuration
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🔧 Configuration

Key parameters can be adjusted in the source files:

### Data Collection (`src/fetch_papers.py`)
- `SEARCH_QUERIES`: ArXiv search topics
- `PAPERS_PER_QUERY`: Papers to fetch per topic
- `CHUNK_SIZE`: Text chunk size for indexing

### Dataset Generation (`src/build_raft_dataset.py`)
- `NUM_EXAMPLES`: Size of training dataset
- `NUM_DISTRACTORS`: Number of distracting documents per example
- `OLLAMA_MODEL`: LLM for question generation

### Fine-tuning (`src/finetune_qlora.py`)
- `MODEL_NAME`: Base model (currently Phi-2)
- `LORA_R`: LoRA rank (lower = faster training)
- `MAX_LENGTH`: Maximum sequence length
- `NUM_EPOCHS`: Training epochs

### RAG System (`src/build_baseline_rag.py`)
- `EMBED_MODEL`: Sentence transformer model
- `TOP_K`: Documents to retrieve

## 🎯 API Endpoints

### Health Check
```http
GET /health
```

### Base RAG (Llama 3)
```http
POST /rag/base
Content-Type: application/json

{
  "question": "What is retrieval augmented generation?",
  "top_k": 4
}
```

### RAFT Model (Fine-tuned Phi-2)
```http
POST /rag/raft
Content-Type: application/json

{
  "question": "What is retrieval augmented generation?",
  "top_k": 4
}
```

## 📊 Evaluation Results

The evaluation framework compares:
- **Base RAG**: Llama 3 with retrieved context
- **RAFT**: Fine-tuned Phi-2 with RAFT training

Metrics include:
- Answer accuracy
- Citation quality
- Response length
- Retrieval relevance

Run evaluation:
```bash
python eval/evaluate.py
```

## 🐳 Docker Deployment

Build and run with Docker:

```bash
# Build image
docker build -t raft-arxiv .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  raft-arxiv
```

## 🔍 Key Technologies

- **Data Collection**: ArXiv API, text chunking
- **Vector Search**: ChromaDB, Sentence Transformers
- **LLMs**: Phi-2 (Microsoft), Llama 3 (Meta)
- **Fine-tuning**: QLoRA, PEFT
- **API**: FastAPI, Pydantic
- **Frontend**: Vanilla JavaScript, HTML/CSS

## 📈 Performance Optimizations

- **QLoRA**: 4-bit quantization for memory efficiency
- **Short sequences**: MAX_LENGTH=256 for faster training
- **Gradient accumulation**: Reduced batch size requirements
- **MPS acceleration**: Apple Silicon GPU support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- [RAFT Paper](https://arxiv.org/abs/2403.10131) by Zhang et al.
- Microsoft Phi-2 model
- Meta Llama 3 via Ollama
- ArXiv API for paper access

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the evaluation results in `eval/results/`
- Review the API logs for debugging

---

**Note**: This implementation is for educational and research purposes. Model outputs should be verified for accuracy in production use cases.