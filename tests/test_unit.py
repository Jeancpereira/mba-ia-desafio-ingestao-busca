import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_text_splitters import RecursiveCharacterTextSplitter

import ingest
import providers
import search


class TestSplit:
    def test_chunk_size_and_overlap_configurados(self):
        assert ingest.CHUNK_SIZE == 1000
        assert ingest.CHUNK_OVERLAP == 150

    def test_chunks_respeitam_tamanho_maximo(self):
        text = "palavra " * 2000
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=ingest.CHUNK_SIZE,
            chunk_overlap=ingest.CHUNK_OVERLAP,
        )
        chunks = splitter.split_text(text)
        assert len(chunks) > 1
        assert all(len(c) <= 1000 for c in chunks)

    def test_chunks_tem_overlap(self):
        text = "palavra " * 2000
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=ingest.CHUNK_SIZE,
            chunk_overlap=ingest.CHUNK_OVERLAP,
        )
        chunks = splitter.split_text(text)
        # Início do chunk seguinte deve reaparecer no fim do chunk anterior (overlap)
        inicio_segundo = chunks[1][:50]
        assert inicio_segundo in chunks[0]


class TestPrompt:
    def test_template_contem_secoes_obrigatorias(self):
        t = search.PROMPT_TEMPLATE
        assert "CONTEXTO:" in t
        assert "REGRAS:" in t
        assert "PERGUNTA DO USUÁRIO:" in t
        assert "Não tenho informações necessárias para responder sua pergunta." in t
        assert "{contexto}" in t
        assert "{pergunta}" in t

    def test_top_k_igual_10(self):
        assert search.TOP_K == 10

    def test_build_context_concatena_page_content(self):
        class Doc:
            def __init__(self, content):
                self.page_content = content

        results = [(Doc("trecho A"), 0.1), (Doc("trecho B"), 0.2)]
        contexto = search.build_context(results)
        assert "trecho A" in contexto
        assert "trecho B" in contexto


class TestSearchChain:
    def test_ask_sem_resultados_nao_chama_llm(self):
        class StoreVazio:
            def similarity_search_with_score(self, question, k):
                return []

        class LLMNuncaChamado:
            def invoke(self, message):
                raise AssertionError("LLM não deveria ser chamado sem contexto")

        chain = search.SearchChain(StoreVazio(), LLMNuncaChamado())
        assert chain.ask("qualquer pergunta") == search.NO_CONTEXT_ANSWER

    def test_ask_com_resultados_chama_llm_com_contexto(self):
        class Doc:
            def __init__(self, content):
                self.page_content = content

        class StoreComResultado:
            def similarity_search_with_score(self, question, k):
                return [(Doc("fato relevante"), 0.9)]

        class LLMEcoa:
            def invoke(self, message):
                class Resposta:
                    content = "ok"

                assert "fato relevante" in message.to_string()
                return Resposta()

        chain = search.SearchChain(StoreComResultado(), LLMEcoa())
        assert chain.ask("pergunta") == "ok"


class TestIngestDedup:
    def test_chunk_id_e_deterministico(self):
        class Chunk:
            def __init__(self, content, page):
                self.page_content = content
                self.metadata = {"page": page}

        c1 = Chunk("mesmo conteúdo", 3)
        c2 = Chunk("mesmo conteúdo", 3)
        c3 = Chunk("conteúdo diferente", 3)
        assert ingest._chunk_id("doc.pdf", c1) == ingest._chunk_id("doc.pdf", c2)
        assert ingest._chunk_id("doc.pdf", c1) != ingest._chunk_id("doc.pdf", c3)

    def test_pdf_path_ausente_gera_erro_claro(self, monkeypatch):
        monkeypatch.delenv("PDF_PATH", raising=False)
        with pytest.raises(ValueError, match="PDF_PATH"):
            ingest.ingest_pdf()


class TestProviders:
    def test_provider_invalido_gera_erro(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "banana")
        with pytest.raises(ValueError, match="LLM_PROVIDER inválido"):
            providers.get_provider()

    def test_provider_default_openai(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        assert providers.get_provider() == "openai"

    def test_provider_gemini(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        assert providers.get_provider() == "gemini"

    def test_embeddings_sem_api_key_gera_erro(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            providers.get_embeddings()

    def test_llm_tem_timeout_configurado(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        llm = providers.get_llm()
        assert llm.request_timeout == providers.LLM_TIMEOUT_SECONDS

    def test_llm_openai_classe_correta(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from langchain_openai import ChatOpenAI

        assert isinstance(providers.get_llm(), ChatOpenAI)

    def test_llm_gemini_classe_correta(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        from langchain_google_genai import ChatGoogleGenerativeAI

        assert isinstance(providers.get_llm(), ChatGoogleGenerativeAI)

    def test_embeddings_openai_modelo_correto(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        emb = providers.get_embeddings()
        assert emb.model == "text-embedding-3-small"
