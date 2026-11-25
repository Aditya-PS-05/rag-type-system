import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default_factory=lambda: os.getenv(
        "DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag"))
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434"))
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "nomic-embed-text"))
    embed_dim: int = field(default_factory=lambda: int(os.getenv("EMBED_DIM", "768")))
    chat_model: str = field(default_factory=lambda: os.getenv("CHAT_MODEL", "qwen3:4b-instruct"))
    team: str = field(default_factory=lambda: os.getenv("RAG_TEAM", "default"))
