#!/usr/bin/env python3
"""
Hugging Face Model Server for RunPod with CUDA optimization
Compatible with Text2SQLGeneratorAPI and OpenAI-style API endpoints
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import psutil
import gc
import subprocess
import os
import json
import argparse
from datetime import datetime
from typing import Optional, Dict, Any

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("Installing FastAPI and dependencies...")
    os.system("pip install fastapi uvicorn python-multipart")
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn

# Global variables to store the loaded model and tokenizer
model = None
tokenizer = None
device = None
model_info = {}

# Request models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict]  # Change from list[ChatMessage] to list[dict] for flexibility
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[list[str]] = None
    stream: Optional[bool] = False
    # Additional generation parameters
    do_sample: Optional[bool] = None
    num_beams: Optional[int] = None
    repetition_penalty: Optional[float] = None
    length_penalty: Optional[float] = None
    early_stopping: Optional[bool] = None
    no_repeat_ngram_size: Optional[int] = None

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[list[str]] = None
    stream: Optional[bool] = False
    # Additional generation parameters
    do_sample: Optional[bool] = None
    num_beams: Optional[int] = None
    repetition_penalty: Optional[float] = None
    length_penalty: Optional[float] = None
    early_stopping: Optional[bool] = None
    no_repeat_ngram_size: Optional[int] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Dict[str, Any]]
    usage: Dict[str, int]

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[Dict[str, Any]]
    usage: Dict[str, int]

def get_gpu_memory():
    """Get GPU memory usage"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        gpu_info = []
        for line in lines:
            used, total = map(int, line.split(', '))
            gpu_info.append((used, total))
        return gpu_info
    except:
        return []

def load_model(model_name: str, force_reload: bool = False):
    """Load a Hugging Face model with CUDA optimization"""
    global model, tokenizer, device, model_info
    
    if model is not None and not force_reload and model_info.get("name") == model_name:
        print(f"Model {model_name} already loaded.")
        return True
    
    # Clear previous model
    if model is not None:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"🚀 Loading model: {model_name}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🔥 Using CUDA GPU: {torch.cuda.get_device_name()}")
        
        # GPU memory info
        gpu_memory = get_gpu_memory()
        if gpu_memory:
            for i, (used, total) in enumerate(gpu_memory):
                print(f"📊 GPU {i} Memory: {used}MB / {total}MB ({used/total*100:.1f}% used)")
    else:
        device = torch.device("cpu")
        print("⚠️  CUDA not available, falling back to CPU")
    
    try:
        # Load tokenizer
        print("📦 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Add pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("🤖 Loading model...")
        start_time = time.time()
        
        # Load model with optimizations
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_cache=True
        )
        
        # Move to device if using CPU
        if device.type == "cpu":
            model = model.to(device)
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        
        # Model info
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            "name": model_name,
            "total_params": num_params,
            "trainable_params": trainable_params,
            "load_time": load_time,
            "device": str(device)
        }
        
        print(f"📊 Total parameters: {num_params:,} ({num_params/1e6:.1f}M)")
        print(f"🎯 Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
        
        # GPU memory after loading
        if device.type == "cuda":
            print(f"🔥 GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"📈 GPU memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        model = None
        tokenizer = None
        return False

def generate_text(prompt: str, **kwargs) -> str:
    """Generate text using the loaded model with only provided parameters"""
    global model, tokenizer, device
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="No model loaded")
    
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Clear cache before generation
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Start with base generation parameters (only required ones)
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "num_return_sequences": 1
        }
        
        # Only add parameters that were explicitly provided
        provided_params = {}
        
        # Handle max_tokens specifically
        if 'max_tokens' in kwargs and kwargs['max_tokens'] is not None:
            generation_kwargs["max_new_tokens"] = kwargs['max_tokens']
            provided_params["max_new_tokens"] = kwargs['max_tokens']
        
        # Add generation parameters only if they were provided
        # Handle do_sample and temperature logic properly
        if 'do_sample' in kwargs and kwargs['do_sample'] is not None:
            generation_kwargs['do_sample'] = kwargs['do_sample']
            provided_params['do_sample'] = kwargs['do_sample']
            
            # Only add temperature if we're sampling
            if kwargs['do_sample'] and 'temperature' in kwargs and kwargs['temperature'] is not None:
                generation_kwargs['temperature'] = kwargs['temperature']
                provided_params['temperature'] = kwargs['temperature']
        elif 'temperature' in kwargs and kwargs['temperature'] is not None and kwargs['temperature'] > 0:
            # If temperature is provided but do_sample isn't, enable sampling
            generation_kwargs['do_sample'] = True
            generation_kwargs['temperature'] = kwargs['temperature']
            provided_params['do_sample'] = True
            provided_params['temperature'] = kwargs['temperature']
        
        # Add other parameters
        param_mapping = {
            'num_beams': 'num_beams',
            'top_p': 'top_p',
            'repetition_penalty': 'repetition_penalty',
            'length_penalty': 'length_penalty',
            'early_stopping': 'early_stopping',
            'no_repeat_ngram_size': 'no_repeat_ngram_size'
        }
        
        for param_name, generation_param in param_mapping.items():
            if param_name in kwargs and kwargs[param_name] is not None:
                generation_kwargs[generation_param] = kwargs[param_name]
                provided_params[generation_param] = kwargs[param_name]
        
        # Only add top_p if we're sampling
        if 'top_p' in generation_kwargs and not generation_kwargs.get('do_sample', False):
            del generation_kwargs['top_p']
            del provided_params['top_p']
        
        # Handle stop sequences if provided
        if 'stop' in kwargs and kwargs['stop'] is not None and kwargs['stop']:
            stop_sequences = kwargs['stop']
            stop_token_ids = []
            for stop_seq in stop_sequences:
                tokens = tokenizer.encode(stop_seq, add_special_tokens=False)
                stop_token_ids.extend(tokens)
            if stop_token_ids:
                generation_kwargs["eos_token_id"] = stop_token_ids
                provided_params["stop_sequences"] = stop_sequences
        
        print(f"🎯 Using only provided parameters: {provided_params}")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(**generation_kwargs)
        
        # Decode only the new tokens
        generated_tokens = outputs[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text.strip()
        
    except Exception as e:
        print(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# FastAPI app
app = FastAPI(title="Hugging Face Model Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hugging Face Model Server", "model_loaded": model is not None}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_info": model_info if model is not None else None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory": get_gpu_memory()
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible completions endpoint (for direct prompt input)"""
    
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="No model loaded")
        
        if not request.prompt:
            raise HTTPException(status_code=400, detail="No prompt provided")
        
        # Use the prompt directly
        prompt = request.prompt
        
        # Generate response - only pass provided parameters
        generation_params = {}
        
        # Only include parameters that were actually provided in the request
        if request.temperature is not None:
            generation_params['temperature'] = request.temperature
        if request.max_tokens is not None:
            generation_params['max_tokens'] = request.max_tokens
        if request.top_p is not None:
            generation_params['top_p'] = request.top_p
        if request.do_sample is not None:
            generation_params['do_sample'] = request.do_sample
        if request.num_beams is not None:
            generation_params['num_beams'] = request.num_beams
        if request.repetition_penalty is not None:
            generation_params['repetition_penalty'] = request.repetition_penalty
        if request.length_penalty is not None:
            generation_params['length_penalty'] = request.length_penalty
        if request.early_stopping is not None:
            generation_params['early_stopping'] = request.early_stopping
        if request.no_repeat_ngram_size is not None:
            generation_params['no_repeat_ngram_size'] = request.no_repeat_ngram_size
        if request.stop is not None:
            generation_params['stop'] = request.stop
        
        generated_text = generate_text(prompt=prompt, **generation_params)
        
        # Create response
        response = CompletionResponse(
            id=f"cmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "text": generated_text,
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": len(prompt.split()),  # Approximate
                "completion_tokens": len(generated_text.split()),  # Approximate
                "total_tokens": len(prompt.split()) + len(generated_text.split())
            }
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Completion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def chat_completions(request: CompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="No model loaded")
        
        # Extract the prompt from messages
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        # Combine messages into a single prompt
        prompt_parts = []
        for msg in request.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt = "\n".join(prompt_parts)
        if not prompt.endswith("\nAssistant:"):
            prompt += "\nAssistant:"
        
        # Generate response - only pass provided parameters
        generation_params = {}
        
        # Only include parameters that were actually provided in the request
        if request.temperature is not None:
            generation_params['temperature'] = request.temperature
        if request.max_tokens is not None:
            generation_params['max_tokens'] = request.max_tokens
        if request.top_p is not None:
            generation_params['top_p'] = request.top_p
        if request.do_sample is not None:
            generation_params['do_sample'] = request.do_sample
        if request.num_beams is not None:
            generation_params['num_beams'] = request.num_beams
        if request.repetition_penalty is not None:
            generation_params['repetition_penalty'] = request.repetition_penalty
        if request.length_penalty is not None:
            generation_params['length_penalty'] = request.length_penalty
        if request.early_stopping is not None:
            generation_params['early_stopping'] = request.early_stopping
        if request.no_repeat_ngram_size is not None:
            generation_params['no_repeat_ngram_size'] = request.no_repeat_ngram_size
        if request.stop is not None:
            generation_params['stop'] = request.stop
        
        generated_text = generate_text(prompt=prompt, **generation_params)
        
        # Create response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": len(prompt.split()),  # Approximate
                "completion_tokens": len(generated_text.split()),  # Approximate
                "total_tokens": len(prompt.split()) + len(generated_text.split())
            }
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Chat completion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """OpenAI-compatible completions endpoint (for direct prompt input)"""
    
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="No model loaded")
        
        if not request.prompt:
            raise HTTPException(status_code=400, detail="No prompt provided")
        
        # Use the prompt directly
        prompt = request.prompt
        
        # Generate response - only pass provided parameters
        generation_params = {}
        
        # Only include parameters that were actually provided in the request
        if request.temperature is not None:
            generation_params['temperature'] = request.temperature
        if request.max_tokens is not None:
            generation_params['max_tokens'] = request.max_tokens
        if request.top_p is not None:
            generation_params['top_p'] = request.top_p
        if request.do_sample is not None:
            generation_params['do_sample'] = request.do_sample
        if request.num_beams is not None:
            generation_params['num_beams'] = request.num_beams
        if request.repetition_penalty is not None:
            generation_params['repetition_penalty'] = request.repetition_penalty
        if request.length_penalty is not None:
            generation_params['length_penalty'] = request.length_penalty
        if request.early_stopping is not None:
            generation_params['early_stopping'] = request.early_stopping
        if request.no_repeat_ngram_size is not None:
            generation_params['no_repeat_ngram_size'] = request.no_repeat_ngram_size
        if request.stop is not None:
            generation_params['stop'] = request.stop
        
        generated_text = generate_text(prompt=prompt, **generation_params)
        
        # Create response
        response = CompletionResponse(
            id=f"cmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "text": generated_text,
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": len(prompt.split()),  # Approximate
                "completion_tokens": len(generated_text.split()),  # Approximate
                "total_tokens": len(prompt.split()) + len(generated_text.split())
            }
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Completion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load_model")
async def load_model_endpoint(model_name: str, force_reload: bool = False):
    """Load a new model"""
    success = load_model(model_name, force_reload)
    if success:
        return {"message": f"Model {model_name} loaded successfully", "model_info": model_info}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to load model {model_name}")

@app.get("/models")
async def list_models():
    """List available models (currently loaded model)"""
    return {
        "object": "list",
        "data": [
            {
                "id": model_info.get("name", "unknown"),
                "object": "model",
                "created": int(time.time()),
                "owned_by": "huggingface"
            }
        ] if model is not None else []
    }

def test_model_interactive():
    """Interactive model testing"""
    if model is None:
        print("❌ No model loaded!")
        return
    
    print("\n🧪 Interactive Model Testing")
    print("Type 'quit' to exit, 'clear' to clear GPU cache")
    print("-" * 50)
    
    while True:
        try:
            prompt = input("\nEnter prompt: ").strip()
            
            if prompt.lower() == 'quit':
                break
            elif prompt.lower() == 'clear':
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("🧹 GPU cache cleared")
                continue
            elif not prompt:
                continue
            
            start_time = time.time()
            response = generate_text(prompt=prompt, max_tokens=100)
            generation_time = time.time() - start_time
            
            print(f"\n🤖 Response: {response}")
            print(f"⏱️  Generation time: {generation_time:.2f}s")
            
            if torch.cuda.is_available():
                print(f"🔥 GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Hugging Face Model Server")
    parser.add_argument("--model", type=str, required=True, 
                       help="Model name to load (required)")
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                       help="Host to bind the server")
    parser.add_argument("--port", type=int, default=7979, 
                       help="Port to bind the server")
    parser.add_argument("--mode", type=str, choices=["server", "test", "interactive"], 
                       default="server", help="Mode to run in")
    
    args = parser.parse_args()
    
    print("🤗 Hugging Face Model Server")
    print("=" * 60)
    
    # Load the model
    print(f"Loading model: {args.model}")
    if not load_model(args.model):
        print("❌ Failed to load model. Exiting.")
        return
    
    if args.mode == "server":
        print(f"\n🚀 Starting server on {args.host}:{args.port}")
        print(f"📖 API Documentation: http://{args.host}:{args.port}/docs")
        print(f"🔗 Health Check: http://{args.host}:{args.port}/health")
        print(f"💬 Chat Endpoint: http://{args.host}:{args.port}/v1/chat/completions")
        print(f"📝 Completions Endpoint: http://{args.host}:{args.port}/v1/completions")
        print("\nPress Ctrl+C to stop the server")
        
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        
    elif args.mode == "test":
        # Run some basic tests
        test_prompts = [
            "Hello, how are you?",
            "What is Python?",
            "Generate SQL to select all users from a table called 'users':"
        ]
        
        print("\n🧪 Running test prompts...")
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i} ---")
            print(f"Prompt: {prompt}")
            
            start_time = time.time()
            response = generate_text(prompt=prompt, max_tokens=100)
            generation_time = time.time() - start_time
            
            print(f"Response: {response}")
            print(f"Time: {generation_time:.2f}s")
        
    elif args.mode == "interactive":
        test_model_interactive()

if __name__ == "__main__":
    main()