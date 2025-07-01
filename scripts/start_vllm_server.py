#!/usr/bin/env python3
"""Start vLLM server for local model inference."""

import argparse
import subprocess
import sys
import time
import requests
import logging

logger = logging.getLogger(__name__)


def check_server_health(base_url: str, max_retries: int = 120) -> bool:
    """Check if vLLM server is healthy."""
    health_url = f"{base_url}/health"
    
    for i in range(max_retries):
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                logger.info("vLLM server is healthy")
                return True
        except requests.exceptions.RequestException:
            pass
        
        logger.info(f"Waiting for server to start... ({i+1}/{max_retries})")
        time.sleep(2)
    
    return False


def start_vllm_server(
    model: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096,
    **kwargs
):
    """Start vLLM server with specified configuration."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", host,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
    ]
    
    # Add additional arguments
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    logger.info(f"Starting vLLM server with command: {' '.join(cmd)}")
    
    try:
        # Start the server
        process = subprocess.Popen(cmd)
        
        # Wait for server to be ready
        base_url = f"http://{host}:{port}"
        if check_server_health(base_url):
            logger.info(f"vLLM server started successfully at {base_url}")
            logger.info(f"Model: {model}")
            logger.info("Press Ctrl+C to stop the server")
            
            # Keep the server running
            process.wait()
        else:
            logger.error("Failed to start vLLM server")
            process.terminate()
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\nShutting down vLLM server...")
        process.terminate()
        process.wait()
        logger.info("Server stopped")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Start vLLM server for local model inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with a specific model
  python scripts/start_vllm_server.py --model microsoft/Phi-3-mini-4k-instruct
  
  # Start with custom port
  python scripts/start_vllm_server.py --model mistralai/Mistral-7B-Instruct-v0.2 --port 8080
  
  # Start with tensor parallel
  python scripts/start_vllm_server.py --model meta-llama/Llama-2-13b-chat-hf --tensor-parallel-size 2
        """
    )
    
    parser.add_argument(
        "--model",
        required=True,
        help="Model name or path (HuggingFace model ID or local path)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0 to 1.0)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "half", "float16", "bfloat16", "float32"],
        help="Data type for model weights"
    )
    parser.add_argument(
        "--quantization",
        choices=["awq", "gptq", "squeezellm"],
        help="Quantization method"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start server
    start_vllm_server(
        model=args.model,
        host=args.host,
        port=args.port,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        quantization=args.quantization
    )


if __name__ == "__main__":
    main()