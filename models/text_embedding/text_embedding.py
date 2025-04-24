import time
from typing import Optional
import requests
import json
import tiktoken

from dify_plugin import TextEmbeddingModel
from dify_plugin.entities.model import EmbeddingInputType
from dify_plugin.entities.model.text_embedding import (
    EmbeddingUsage,
    TextEmbeddingResult,
)
from dify_plugin.errors.model import CredentialsValidateFailedError, InvokeError


class HuggingfaceEmbeddingModelsTextEmbeddingModel(TextEmbeddingModel):
    """
    Model class for huggingface_embedding_models text embedding model.
    Uses a dedicated embedding service via HTTP API.
    """

    def _invoke(
        self,
        model: str,
        credentials: dict,
        texts: list[str],
        user: Optional[str] = None,
        input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
    ) -> TextEmbeddingResult:
        """
        Invoke the embedding model via HTTP API.
        
        Args:
            model: Model name to use
            credentials: Model credentials containing API endpoint
            texts: List of texts to embed
            user: Optional user identifier
            input_type: Type of input (document or query)
            
        Returns:
            TextEmbeddingResult with embeddings
            
        Raises:
            InvokeError: If the API call fails
        """
        api_endpoint = credentials.get("api_endpoint", "http://host.docker.internal:8001/embedding")
        
        request_data = {
            "model_name": model,
            "texts": texts
        }
        
        try:
            response = requests.post(
                api_endpoint,
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            response.raise_for_status()
            result = response.json()
            embeddings = result.get("embeddings", [])
            
            tokens = sum(self.get_num_tokens(model, credentials, texts))
            usage = self._calc_response_usage(
                model=model,
                credentials=credentials,
                tokens=tokens,
            )
            
            return TextEmbeddingResult(
                embeddings=embeddings,
                model=model,
                usage=usage
            )
            
        except requests.exceptions.RequestException as e:
            raise InvokeError(f"Error calling embedding service API: {str(e)}")
        except json.JSONDecodeError:
            raise InvokeError(f"Error parsing API response: {response.text}")
        except Exception as e:
            raise InvokeError(f"Unexpected error during embedding: {str(e)}")

    def get_num_tokens(
        self, model: str, credentials: dict, texts: list[str]
    ) -> list[int]:
        """
        Get number of tokens for given texts using tiktoken.
        
        Args:
            model: Model name
            credentials: Model credentials
            texts: Texts to count tokens for
            
        Returns:
            List of token counts
        """
        try:
            encoding_name = "cl100k_base"
            encoding = tiktoken.get_encoding(encoding_name)
            return [len(encoding.encode(text)) for text in texts]
        except Exception:
            return [len(text.split()) for text in texts]

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials by checking API availability.
        
        Args:
            model: Model name
            credentials: Model credentials containing API endpoint
            
        Raises:
            CredentialsValidateFailedError: If credentials validation fails
        """
        base_url = credentials.get("api_endpoint", "http://localhost:8001/embedding")
        health_endpoint = base_url.rsplit('/', 1)[0] + "/health"
        
        try:
            response = requests.get(
                health_endpoint,
                timeout=10
            )
            
            if response.status_code != 200:
                raise CredentialsValidateFailedError(f"API health check failed: {response.status_code}")
                
            result = response.json()
            if result.get("status") != "ok":
                raise CredentialsValidateFailedError(f"API reports unhealthy status: {result}")
                
        except requests.exceptions.RequestException as e:
            raise CredentialsValidateFailedError(f"Could not connect to embedding service: {str(e)}")

    def _calc_response_usage(
        self, model: str, credentials: dict, tokens: int
    ) -> EmbeddingUsage:
        """
        Calculate the embedding usage, include input tokens and request times.
        
        Args:
            model: Model name
            credentials: Model credentials
            tokens: Number of tokens processed
            
        Returns:
            EmbeddingUsage with usage information
        """
        return EmbeddingUsage(
            tokens=tokens,
            total_tokens=tokens,
            unit_price=0.0,
            price_unit="0.0",
            total_price=0.0,
            currency="USD",
            latency=time.perf_counter() - self.started_at,
        )

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """
        Map exception types to InvokeError types.
        
        Returns:
            Dictionary mapping InvokeError types to lists of exception types
        """
        return {
            InvokeError: [RuntimeError, ValueError, OSError, requests.exceptions.RequestException],
            CredentialsValidateFailedError: [CredentialsValidateFailedError],
        }