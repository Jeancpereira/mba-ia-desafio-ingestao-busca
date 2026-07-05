import os

from dotenv import load_dotenv

load_dotenv()

PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"


def get_provider():
    provider = os.getenv("LLM_PROVIDER", PROVIDER_OPENAI).strip().lower()
    if provider not in (PROVIDER_OPENAI, PROVIDER_GEMINI):
        raise ValueError(
            f"LLM_PROVIDER inválido: '{provider}'. Use 'openai' ou 'gemini'."
        )
    return provider


def require_env(name):
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Variável de ambiente obrigatória não definida: {name}")
    return value


def get_embeddings():
    provider = get_provider()
    if provider == PROVIDER_OPENAI:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=require_env("OPENAI_API_KEY"),
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001"),
        google_api_key=require_env("GOOGLE_API_KEY"),
    )


LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))


def get_llm():
    provider = get_provider()
    if provider == PROVIDER_OPENAI:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano"),
            api_key=require_env("OPENAI_API_KEY"),
            timeout=LLM_TIMEOUT_SECONDS,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite"),
        google_api_key=require_env("GOOGLE_API_KEY"),
        temperature=0,
        timeout=LLM_TIMEOUT_SECONDS,
    )


def get_vector_store():
    from langchain_postgres import PGVector

    return PGVector(
        embeddings=get_embeddings(),
        collection_name=require_env("PG_VECTOR_COLLECTION_NAME"),
        connection=require_env("DATABASE_URL"),
        use_jsonb=True,
    )
