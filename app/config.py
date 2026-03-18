# app/config.py

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # Watsonx / IBM Cloud
    watsonx_url: str = os.getenv(
        "WATSONX_URL", "https://us-south.ml.cloud.ibm.com"
    )
    watsonx_api_key: str = os.getenv("WATSONX_API_KEY")
    watsonx_project_id: str = os.getenv("WATSONX_PROJECT_ID")

    # IBM Granite models
    granite_model_id: str = os.getenv(
        "GRANITE_MODEL_ID", "ibm/granite-3-8b-instruct"
    )
    embedding_model_id: str = os.getenv(
        "EMBEDDING_MODEL_ID",
        "ibm/granite-embedding-278m-multilingual",
    )

    # Project paths
    base_dir: Path = Path(__file__).resolve().parents[1]
    raw_docs_dir: Path = base_dir / "data" / "documents"
    vector_store_dir: Path = base_dir / "data" / "vector_store"


settings = Settings()


def get_llm():
    """
    Returns an IBM Granite chat model via Watsonx.
    Used for grounded generation in the RAG pipeline.
    """
    from langchain_ibm import ChatWatsonx

    if not settings.watsonx_api_key or not settings.watsonx_project_id:
        raise RuntimeError(
            "Missing WATSONX_API_KEY or WATSONX_PROJECT_ID environment variables."
        )

    params = {
        "decoding_method": "greedy",
        "temperature": 0.0,      # deterministic, grounded answers
        "max_new_tokens": 1024,
        "min_new_tokens": 5,
    }

    return ChatWatsonx(
        model_id=settings.granite_model_id,
        url=settings.watsonx_url,
        project_id=settings.watsonx_project_id,
        api_key=settings.watsonx_api_key,
        params=params,
    )


def get_embeddings():
    """
    Returns IBM Granite embedding model via Watsonx.
    Used for vector indexing and similarity search.
    """
    from langchain_ibm import WatsonxEmbeddings
    from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

    embed_params = {
        # Keep embeddings aligned with 300–500 token chunking
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 512,
    }

    return WatsonxEmbeddings(
        model_id=settings.embedding_model_id,
        url=settings.watsonx_url,
        project_id=settings.watsonx_project_id,
        api_key=settings.watsonx_api_key,
        params=embed_params,
    )
