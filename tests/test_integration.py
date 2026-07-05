"""Testes de integração — exigem docker compose up e API key válida no .env."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

load_dotenv()

_provider = os.getenv("LLM_PROVIDER", "openai")
_key_var = "OPENAI_API_KEY" if _provider == "openai" else "GOOGLE_API_KEY"

requires_api = pytest.mark.skipif(
    not os.getenv(_key_var),
    reason=f"{_key_var} não definida — teste de integração ignorado",
)


@requires_api
class TestIntegracao:
    def test_ingestao_gravou_embeddings(self):
        import sqlalchemy

        engine = sqlalchemy.create_engine(os.getenv("DATABASE_URL"))
        with engine.connect() as conn:
            count = conn.execute(
                sqlalchemy.text("SELECT count(*) FROM langchain_pg_embedding")
            ).scalar()
        assert count > 0, "Nenhum embedding no banco — rode python src/ingest.py"

    def test_busca_retorna_ate_10_resultados_com_score(self):
        from providers import get_vector_store

        store = get_vector_store()
        results = store.similarity_search_with_score(
            "Qual o faturamento da empresa Alfa Energia S.A.?", k=10
        )
        assert 0 < len(results) <= 10
        for doc, score in results:
            assert doc.page_content
            assert isinstance(score, float)

    def test_pergunta_dentro_do_contexto(self):
        from search import search_prompt

        answer = search_prompt("Qual o faturamento da empresa Alfa Energia S.A.?")
        assert answer is not None
        assert "722.875.391,46" in answer

    def test_pergunta_fora_do_contexto(self):
        from search import search_prompt

        answer = search_prompt("Qual é a capital da França?")
        assert answer is not None
        assert "Não tenho informações necessárias" in answer
