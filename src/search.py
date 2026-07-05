from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from providers import get_llm, get_vector_store

TOP_K = 10

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


def search_prompt(question=None):
    try:
        store = get_vector_store()
        llm = get_llm()
    except Exception as error:
        print(f"Erro de inicialização: {error}")
        return None

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["contexto", "pergunta"],
    )

    def retrieve(pergunta):
        results = store.similarity_search_with_score(pergunta, k=TOP_K)
        return {"contexto": build_context(results), "pergunta": pergunta}

    chain = RunnableLambda(retrieve) | prompt | llm

    if question is not None:
        return chain.invoke(question).content

    return chain
