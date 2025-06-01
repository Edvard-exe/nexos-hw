# Smart LLM Router Project

## Overview

This project implements a smart gateway/router system that intelligently routes user queries to appropriate Large Language Models (LLMs) based on prompt characteristics, token count, complexity, and creativity requirements. The system aims to optimize cost, latency, and accuracy by selecting the most suitable model for each specific query.

## Project Structure

```
nexos.ai/
├── SmartRouter.ipynb              # Main research notebook with experiments and findings
├── router_api/
│   ├── router_api.py             # FastAPI-based router implementation
│   └── test_api.py               # Comprehensive test suite for the API
├── model_deployment/
│   └── test-bert/                # Fine-tuned DistilBERT model deployment
│       ├── app.py                # Gradio interface for the fine-tuned model
│       ├── model.safetensors     # Fine-tuned DistilBERT model weights
│       ├── config.json           # Model configuration
│       ├── tokenizer.json        # Tokenizer configuration
│       └── README.md             # Deployment documentation
├── Finetuned DistilBERT/         # Fine-tuned model artifacts
├── requirements.txt              # Project dependencies
└── Train DataFrame.csv           # Training dataset for model classification
```

## Key Findings from Research

### Main Experimental Results

The project explored two primary approaches for intelligent LLM routing:

#### 1. LLM-Based Router (GPT-4.1-nano)
- **Accuracy**: 90% on MMLU benchmark
- **Latency**: 0.231s average response time
- **Cost Effectiveness**: Extremely high due to GPT-4.1-nano usage ($0.2/1M tokens)
- **Model Selection**: Routes between gpt-4.1-nano, gpt-4.1, and o4-mini based on query complexity

#### 2. Fine-tuned DistilBERT Router
- **Accuracy**: 90% on MMLU benchmark (matching LLM-based router)
- **Training Data**: 8,142 classified queries with domain and complexity labels
- **Model Size**: Lightweight DistilBERT (66M parameters)
- **Deployment**: Available at https://huggingface.co/spaces/edvard-exe/test-bert

### Performance Comparison

| Metric | LLM Router | DistilBERT Router | Simple GPT-4.1 |
|--------|------------|-------------------|-----------------|
| Accuracy | 90% | 90% | 80% |
| Avg Latency | 0.231s | Variable | 0.689s |
| Cost | Very Low | Minimal | Medium |
| Model Usage | Balanced | 80% o4-mini preference | Fixed |

### Key Insights

1. **Training Data Alignment**: The router's performance heavily depends on training data matching target use cases. MMLU's academic nature caused the DistilBERT model to prefer o4-mini (80% usage) despite training data showing 60% gpt-4.1-nano preference.

2. **Cost-Performance Trade-off**: The LLM-based router achieved optimal balance of accuracy, latency, and cost effectiveness.

3. **Domain Mismatch Impact**: Academic evaluation datasets may not reflect real-world query distributions, affecting routing decisions.

## Router API Documentation

### Core Features

The FastAPI-based router provides intelligent query routing with the following capabilities:

- **Multi-modal Support**: Handles text queries with optional file attachments (images, PDFs)
- **Structured Responses**: Uses Pydantic models for type-safe request/response handling
- **Cost Optimization**: Routes queries based on complexity, creativity requirements, and token count
- **File Processing**: Supports image analysis and PDF document processing

### Available Models

The router selects from three OpenAI models:

- **gpt-4.1-nano**: Ideal for simple tasks, minimal reasoning, cost-effective ($0.2/1M tokens)
- **gpt-4.1**: Balanced for moderate complexity, coding, basic file interactions ($2.00/1M tokens)
- **o4-mini**: Complex reasoning, deep analysis, comprehensive file processing ($1.10/1M tokens)

### API Endpoints

#### 1. Basic Query Routing
```http
POST /route
Content-Type: application/json

{
    "prompt": "Your query here",
    "temperature": 0.7,
    "openai_api_key": "optional_key"
}
```

#### 2. File-based Query Routing
```http
POST /route/with-file
Content-Type: multipart/form-data

prompt: "Analyze this document"
temperature: 0.7
file: [uploaded_file]
openai_api_key: "optional_key"
```

#### 3. Health Check
```http
GET /health
```

#### 4. Available Models
```http
GET /models
```

#### 5. Root Information
```http
GET /
```

### Response Format

```json
{
    "response": "Generated response from selected model",
    "selected_model": "gpt-4.1-nano",
    "file_processed": true,
    "file_processing_error": null
}
```

### Authentication

API keys can be provided via:
- Request body (`openai_api_key` field)
- HTTP header (`X-OpenAI-API-Key`)
- Environment variable (`OPENAI_API_KEY`)

## Installation and Setup

### Prerequisites

- Python 3.8+
- OpenAI API key
- Hugging Face account and token (for dataset access and model deployment)
- uv package manager (recommended for faster dependency management)

### Installation

1. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository:
```bash
git clone <repository_url>
cd nexos.ai
```

3. Install dependencies using uv:
```bash
uv pip install -r requirements.txt
```

4. Set up environment variables:
```bash
export OPENAI_API_KEY="your_openai_api_key"
export HUGGINGFACE_TOKEN="your_huggingface_token"
```

Note: The Hugging Face token is required to access the ChatBot Arena dataset used in the research notebook. You can obtain a token from your [Hugging Face account settings](https://huggingface.co/settings/tokens).

### Running the Router API

```bash
cd router_api
uv run uvicorn router_api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

### Testing

Run the comprehensive test suite:

```bash
cd router_api
uv run python -m pytest test_api.py -v
```

Test coverage includes:
- Router initialization and configuration
- File type detection and processing
- API key handling
- Endpoint functionality
- Async operations
- Error handling

## Model Deployment

### Fine-tuned DistilBERT Model

The fine-tuned DistilBERT model is deployed on Hugging Face Spaces:
- **URL**: https://huggingface.co/spaces/edvard-exe/test-bert
- **Interface**: Gradio-based web interface
- **Input**: Text queries
- **Output**: Recommended model, confidence score, probability distribution

### Local Deployment

To run the fine-tuned model locally:

```bash
cd model_deployment/test-bert
python app.py
```

### Immediate Optimizations

1. **Latency Reduction**
   - Implement response caching for common queries
   - Optimize model loading and inference pipeline
   - Add request batching for multiple queries

2. **Hierarchical Classification**
   - First classify domain (technical, creative, analytical)
   - Then assess complexity within domain
   - Finally route to appropriate model

3. **Enhanced Training Data**
   - Collect domain-specific query datasets
   - Balance training data across model categories
   - Include real-world query distributions

### Advanced Features

1. **Multi-Model Ensemble**
   - Implement RouterLLM techniques for probability-based routing
   - Support hybrid approaches combining multiple models
   - Add confidence-based fallback mechanisms

2. **Adaptive Learning**
   - Implement feedback loops for routing decisions
   - Track model performance across different query types
   - Automatically adjust routing thresholds

3. **Cost Optimization**
   - Dynamic pricing consideration in routing decisions
   - Budget-aware model selection
   - Usage analytics and cost tracking

## Research Methodology

The project followed a systematic approach:

1. **Initial LLM-based Router**: Implemented using GPT-4.1-nano for routing decisions
2. **Data Collection**: Generated 8,142 labeled queries using ChatBot Arena dataset
3. **Model Fine-tuning**: Trained DistilBERT on classification task
4. **Evaluation**: Tested both approaches on MMLU benchmark
5. **Deployment**: Created production-ready API and model deployment

## License

This project is licensed under the MIT License. See LICENSE file for details.