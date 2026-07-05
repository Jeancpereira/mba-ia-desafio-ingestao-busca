from search import build_chain

EXIT_COMMANDS = {"sair", "exit", "quit"}


def main():
    chain = build_chain()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Faça sua pergunta (digite 'sair' para encerrar):\n")

    while True:
        try:
            question = input("PERGUNTA: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando o chat.")
            break

        if not question:
            continue

        if question.lower() in EXIT_COMMANDS:
            print("Encerrando o chat.")
            break

        try:
            answer = chain.ask(question)
        except KeyboardInterrupt:
            print("\nEncerrando o chat.")
            break
        except Exception as error:
            print(f"Erro ao processar a pergunta ({type(error).__name__}): {error}\n")
            continue

        print(f"RESPOSTA: {answer}\n")


if __name__ == "__main__":
    main()
