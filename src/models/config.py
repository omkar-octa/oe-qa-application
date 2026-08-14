from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    claude_api_key: str
    claude_model: str = "claude-sonnet-5"
    claude_max_tokens: int = 4096

    # Vision work is split off from claude_model because neither job needs the
    # reasoning claude_model is paying for; both are bounded by how much of the
    # render the model can actually resolve. Both therefore want the
    # high-resolution vision tier (2576px on the long edge, matching
    # PageRenderer): a model below that tier downsamples to ~1568px.
    #
    # For page transcription that downsample loses strokes on 8-9pt body text.
    # For figures it was measured on Fig. 5 of the Calvillo fixture, a
    # six-panel UK map: claude-haiku-4-5 read the values but could not attach
    # them to regions at all ("values assigned to regions based on their
    # position on the map") and dropped the signs the legend encodes by colour,
    # which is the difference between a number being in the index and the
    # question being answerable. Kept as separate settings because figures are
    # where a cost-sensitive run would trade that accuracy away.
    claude_vision_model: str = "claude-sonnet-5"
    claude_figure_model: str = "claude-sonnet-5"
    index_path: Path = Path("data/index.json")
    top_k: int = 5

    # Off by default: PostgresSearchIndex.search() otherwise fetches and sends
    # one neighbouring chunk each side of every match, on every search, whether
    # or not the match already stood on its own -- real, unconditional token
    # cost with no per-chunk check for whether it's needed (see docs/search.md,
    # "This is unconditional, and it costs tokens even when the match alone was
    # enough"). Flip this in .env, or pass attach_context=True to
    # PostgresSearchIndex directly, to turn it back on.
    attach_search_context: bool = False

    # Dense embeddings via OpenAI's API rather than a locally-run model: on
    # this machine's CPU, BGE-M3 measured at ~180 chars/second, making a bulk
    # embed of even this small corpus a ~60 minute job (see
    # docs/architecture.md). A remote API call has no local model-load cost
    # to amortise, which was the entire reason the embedding daemon existed;
    # there is no daemon any more, just this.
    openai_api_key: str | None = None
    embedding_model: str = "text-embedding-3-large"
    # text-embedding-3-large's own maximum is 3072; OpenAI's v3 models are
    # trained so a shorter requested dimension still outperforms the older
    # ada-002 at full length, so this trades a little quality for a smaller
    # vector column and faster HNSW build/search.
    embedding_dimensions: int = 512

    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_user: str = "postgres"
    pg_password: str = "postgres"
    pg_database: str = "embeddings"


settings = Settings()
