"""Simple async client for OpenAI-compatible API servers."""

import asyncio
import aiohttp
import json
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        config = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        if self.stop:
            config["stop"] = self.stop
        return config


async def generate(
    session: aiohttp.ClientSession,
    prompt: str,
    config: GenerationConfig,
    base_url: str = "http://localhost:8000",
    model_name: Optional[str] = None,
    debug: bool = False
) -> str:
    """Send a single generation request using OpenAI-compatible completions endpoint.
    
    The server will automatically batch this with other concurrent requests.
    """
    data = {
        "prompt": prompt,
        **config.to_dict()
    }
    
    if model_name:
        data["model"] = model_name
    
    if debug:
        logger.debug(f"Request prompt: {prompt[:200]}...")
        logger.debug(f"Request config: {json.dumps(config.to_dict())}")
        
    try:
        async with session.post(
            f"{base_url}/v1/completions",
            json=data
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"HTTP {response.status}: {error_text}")
                
            result = await response.json()
            
            # Check if we got a valid response structure
            if not result.get("choices") or not result["choices"]:
                raise Exception(f"Invalid response structure: {result}")
                
            text = result["choices"][0].get("text", "")
            
            # Check for empty text
            if not text or not text.strip():
                logger.warning(f"Empty text in response. Full response: {json.dumps(result, indent=2)}")
                finish_reason = result["choices"][0].get("finish_reason", "unknown")
                logger.warning(f"Finish reason: {finish_reason}")
                if debug:
                    logger.warning(f"Request prompt: {prompt[:500]}...")
                
            if debug:
                logger.debug(f"Response text: '{text[:200]}...'")
                
            return text
            
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise


async def generate_many(
    prompts: List[str],
    config: GenerationConfig,
    base_url: str = "http://localhost:8000",
    model_name: Optional[str] = None,
    max_concurrent: int = 50,
    debug: bool = False
) -> List[str]:
    """Generate responses for multiple prompts concurrently.
    
    Just sends all requests concurrently and lets vLLM handle the batching.
    """
    connector = aiohttp.TCPConnector(limit=max_concurrent)
    timeout = aiohttp.ClientTimeout(total=300)  # Increase to 5 minutes
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Create all tasks
        tasks = [
            generate(session, prompt, config, base_url, model_name, debug)
            for prompt in prompts
        ]
        
        # Run them all concurrently - vLLM will batch them automatically
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results
        outputs = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to generate for prompt {i}: {result}", exc_info=result)
                outputs.append(None)  # Return None to indicate failure
            else:
                outputs.append(result)
                
        return outputs