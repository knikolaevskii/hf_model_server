# Hugging Face Model Server

A high-performance CUDA-optimized server for running Hugging Face language models with OpenAI-compatible API endpoints.

## Features

- 🚀 **CUDA optimized** for GPU inference
- 🔄 **OpenAI-compatible APIs** (`/v1/completions` and `/v1/chat/completions`)
- 🎯 **Dynamic model loading** - switch models without restarting
- ⚡ **Flexible generation parameters** - only sends provided parameters to the model
- 🧠 **Memory efficient** with automatic GPU cache management
- 📊 **Real-time monitoring** with health checks and GPU memory tracking

## Quick Start

### Prerequisites
```bash
pip install torch transformers fastapi uvicorn accelerate
```

### Basic Usage

**Start the server:**
```bash
python3 hf_server.py --model microsoft/DialoGPT-medium --port 7860
```

**Test the API:**
```bash
curl -X POST "http://localhost:7860/v1/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Generate SQL to select all users",
       "temperature": 0,
       "max_tokens": 100,
       "model": "test"
     }'
```

## API Endpoints

### `/v1/completions`
OpenAI-style completions endpoint for direct prompt input.

**Request:**
```json
{
  "prompt": "Your prompt here",
  "temperature": 0.7,
  "max_tokens": 256,
  "do_sample": false,
  "num_beams": 4,
  "stop": [";", "```"]
}
```

### `/v1/chat/completions`
OpenAI-style chat completions endpoint.

**Request:**
```json
{
  "model": "your-model",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 256
}
```

### `/health`
Health check endpoint with system status and GPU memory info.

### `/load_model`
Dynamically load a new model without restarting the server.

## Command Line Options

```bash
python3 hf_server.py [OPTIONS]

Options:
  --model TEXT       Model name to load (required)
  --host TEXT        Host to bind server [default: 0.0.0.0]
  --port INTEGER     Port to bind server [default: 8000]
  --mode CHOICE      Mode: server/test/interactive [default: server]
```

## Operation Modes

### Server Mode (Default)
```bash
python3 hf_server.py --model defog/llama-3-sqlcoder-8b --mode server

python3 hf_server.py --model premai-io/prem-1B-SQL --mode server
```
Starts the API server on the specified port.

### Test Mode
```bash
python3 hf_server.py --model your-model --mode test
```
Runs predefined test prompts and exits.

### Interactive Mode
```bash
python3 hf_server.py --model your-model --mode interactive
```
Interactive chat session for manual testing.

## Supported Generation Parameters

- `temperature` - Sampling temperature (0.0 = deterministic)
- `max_tokens` - Maximum tokens to generate
- `do_sample` - Enable/disable sampling
- `num_beams` - Number of beams for beam search
- `top_p` - Nucleus sampling parameter
- `repetition_penalty` - Repetition penalty
- `length_penalty` - Length penalty for beam search
- `early_stopping` - Early stopping for beam search
- `stop` - Stop sequences
- `no_repeat_ngram_size` - N-gram repetition prevention

## Recommended Models

- **SQL Generation**: `defog/llama-3-sqlcoder-8b`
- **General Chat**: `microsoft/DialoGPT-medium`
- **Code Generation**: `microsoft/codegen-350M-mono`
- **Large Models**: `EleutherAI/gpt-neo-1.3B`

## Usage with Text2SQLGeneratorAPI

```python
from premsql.generators.base import Text2SQLGeneratorAPI

generator = Text2SQLGeneratorAPI(
    model_name="your-model-name",
    experiment_name="test",
    type="api",
    api_base_url="http://your-server:7860/v1"
)

result = generator.generate({
    "prompt": "Generate SQL to select all users"
})
```

## Performance Tips

- Use `do_sample=False` and `num_beams=4` for optimal SQL generation
- Set `temperature=0` for deterministic outputs
- Monitor GPU memory with `/health` endpoint
- Use appropriate model sizes for your GPU memory

## GPU Requirements

- **Small models** (< 1B params): 4GB+ VRAM
- **Medium models** (1-3B params): 8GB+ VRAM  
- **Large models** (7B+ params): 16GB+ VRAM

## Error Handling

The server includes comprehensive error handling and logging. Check server logs for debugging information.

## License

This project is designed for educational and research purposes. Ensure compliance with individual model licenses from Hugging Face.