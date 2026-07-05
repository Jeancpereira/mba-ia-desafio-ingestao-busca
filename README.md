# Desafio MBA Engenharia de Software com IA - Full Cycle

## Ingestão e Busca Semântica com LangChain e Postgres

Software que ingere um PDF em um banco PostgreSQL com extensão pgVector e permite fazer perguntas via CLI, respondidas exclusivamente com base no conteúdo do PDF.

## Tecnologias

- Python 3.13 + LangChain
- PostgreSQL 17 + pgVector (via Docker)
- Embeddings/LLM: OpenAI **ou** Google Gemini (configurável)

## Pré-requisitos

- Docker e Docker Compose
- Python 3.13+
- API Key da OpenAI ou do Google Gemini

## Configuração

1. Crie e ative o ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Crie o arquivo `.env` a partir do template:

```bash
cp .env.example .env
```

4. Edite o `.env` e preencha:

- `LLM_PROVIDER`: `openai` ou `gemini`
- `OPENAI_API_KEY` (se provider openai) ou `GOOGLE_API_KEY` (se provider gemini)
- Demais variáveis já vêm com valores padrão funcionais:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=documents
PDF_PATH=document.pdf
```

## Execução

1. Suba o banco de dados:

```bash
docker compose up -d
```

2. Execute a ingestão do PDF:

```bash
python src/ingest.py
```

3. Rode o chat:

```bash
python src/chat.py
```

Exemplo de uso:

```
Faça sua pergunta (digite 'sair' para encerrar):

PERGUNTA: Qual o faturamento da empresa Alfa Energia S.A.?
RESPOSTA: O faturamento foi de R$ 722.875.391,46.

PERGUNTA: Qual é a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## Testes

```bash
pip install pytest
pytest tests/ -v
```

- `tests/test_unit.py`: testes unitários (split, prompt, providers) — não exigem rede.
- `tests/test_integration.py`: testes de integração e E2E — exigem banco no ar, ingestão executada e API key válida no `.env` (são ignorados automaticamente sem a key).

## Estrutura

```
├── docker-compose.yml      # PostgreSQL + pgVector
├── requirements.txt        # Dependências
├── .env.example            # Template de configuração
├── src/
│   ├── ingest.py           # Ingestão do PDF (chunks de 1000, overlap 150)
│   ├── search.py           # Busca vetorial (k=10) + prompt + LLM
│   ├── chat.py             # CLI de perguntas
│   └── providers.py        # Factory OpenAI/Gemini (embeddings, LLM, vector store)
├── tests/                  # Suite pytest
├── document.pdf            # PDF para ingestão
└── README.md
```
