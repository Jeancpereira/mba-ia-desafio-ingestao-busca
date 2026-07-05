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
    def test_ingestao_gravou_embeddings_na_collection_configurada(self):
        import sqlalchemy

        collection = os.getenv("PG_VECTOR_COLLECTION_NAME")
        engine = sqlalchemy.create_engine(os.getenv("DATABASE_URL"))
        with engine.connect() as conn:
            count = conn.execute(
                sqlalchemy.text(
                    "SELECT count(*) FROM langchain_pg_embedding e "
                    "JOIN langchain_pg_collection c ON c.uuid = e.collection_id "
                    "WHERE c.name = :collection"
                ),
                {"collection": collection},
            ).scalar()
        assert count > 0, "Nenhum embedding na collection — rode python src/ingest.py"

    def test_busca_retorna_ate_10_resultados_com_score(self):
        from providers import get_vector_store

        store = get_vector_store()
        results = store.similarity_search_with_score("faturamento", k=10)
        assert 0 < len(results) <= 10
        for doc, score in results:
            assert doc.page_content
            assert isinstance(score, float)

    def test_pergunta_dentro_do_contexto_nao_retorna_recusa(self):
        from search import NO_CONTEXT_ANSWER, build_chain

        chain = build_chain()
        assert chain is not None

        # Pergunta genérica sobre "faturamento": deve casar com algum chunk do
        # PDF ingerido e gerar uma resposta baseada no CONTEXTO — sem acoplar
        # o teste a um valor literal de um PDF de exemplo específico.
        answer = chain.ask("Qual o faturamento mencionado no documento?")
        assert answer != NO_CONTEXT_ANSWER
        assert answer.strip()

    def test_pergunta_fora_do_contexto_retorna_recusa_padrao(self):
        from search import NO_CONTEXT_ANSWER, build_chain

        chain = build_chain()
        assert chain is not None

        answer = chain.ask("Qual é a capital da França?")
        assert NO_CONTEXT_ANSWER in answer

    def test_reingestao_nao_duplica_chunks(self):
        import sqlalchemy

        from ingest import ingest_pdf

        collection = os.getenv("PG_VECTOR_COLLECTION_NAME")
        engine = sqlalchemy.create_engine(os.getenv("DATABASE_URL"))

        def count_embeddings():
            with engine.connect() as conn:
                return conn.execute(
                    sqlalchemy.text(
                        "SELECT count(*) FROM langchain_pg_embedding e "
                        "JOIN langchain_pg_collection c ON c.uuid = e.collection_id "
                        "WHERE c.name = :collection"
                    ),
                    {"collection": collection},
                ).scalar()

        before = count_embeddings()
        ingest_pdf()
        after = count_embeddings()
        assert after == before, "Reingestão do mesmo PDF duplicou chunks na collection"
