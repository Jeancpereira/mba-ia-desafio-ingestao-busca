from langchain_core.prompts import PromptTemplate

from providers import get_llm, get_vector_store

TOP_K = 10
NO_CONTEXT_ANSWER = "Não tenho informações necessárias para responder sua pergunta."

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def build_context(results):
    return "\n\n".join(doc.page_content for doc, _score in results)


class SearchChain:
    """Recursos inicializados uma única vez e reutilizados a cada pergunta."""

    def __init__(self, store, llm):
        self.store = store
        self.llm = llm
        self.prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["contexto", "pergunta"],
        )

    def ask(self, question):
        results = self.store.similarity_search_with_score(question, k=TOP_K)
        if not results:
            return NO_CONTEXT_ANSWER

        contexto = build_context(results)
        message = self.prompt.invoke({"contexto": contexto, "pergunta": question})
        return self.llm.invoke(message).content


def build_chain():
    try:
        store = get_vector_store()
        llm = get_llm()
    except ValueError as error:
        # ValueError vem de _require_env (providers.py) — mensagem já é segura,
        # cita apenas o nome da variável ausente, nunca um valor/segredo.
        print(f"Erro de inicialização: {error}")
        return None
    except Exception as error:
        # Outras exceções (conexão recusada, etc.) podem embutir a DSN completa
        # com credenciais — nunca imprimir a mensagem bruta.
        print(f"Erro de inicialização ({type(error).__name__}). Verifique DATABASE_URL e as credenciais.")
        return None

    return SearchChain(store, llm)
