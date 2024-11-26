from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import pandas as pd

# Carregar variáveis de ambiente
load_dotenv()

# Carregar a planilha com os dados
def load_contacts(file_path):
    return pd.read_excel(file_path)

# Função para buscar o número com base no nome do colaborador
def find_user_number(dataframe, collaborator_name):
    user_row = dataframe[dataframe['Colaboradores'].str.contains(collaborator_name, case=False, na=False)]
    if not user_row.empty:
        return user_row.iloc[0]['N° corporativo'], user_row.iloc[0]['Operadora']
    else:
        return None, None

# Configurar o modelo principal da IA (conversa geral)
conversation_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Configurar o modelo de extração de nomes
name_extraction_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=20,  # Limite de tokens para garantir respostas curtas
    timeout=None,
    max_retries=2,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Função para identificar nomes de pessoas usando o LLM
def extract_name_with_llm(user_input):
    prompt = f"Extraia e forneça somente o nome da pessoa mencionado no texto: '{user_input}'"
    response = name_extraction_llm.invoke(prompt)
    name = response.content.strip()
    print(f"Nome identificado pelo modelo: {name}")
    return name

# Função principal
def main():
    # Input do usuário
    user_input = input("Digite sua solicitação: ")

    # Extrair o nome do colaborador usando a IA de extração de nomes
    collaborator_name = extract_name_with_llm(user_input)

    if collaborator_name:
        print(f"Identificado nome do colaborador: {collaborator_name}")
        # Carregar a planilha e buscar o número do colaborador
        contacts_df = load_contacts("data.xlsx")
        user_phone, user_operator = find_user_number(contacts_df, collaborator_name)
        
        if user_phone:
            # Criar a resposta com o número encontrado
            response_context = (f"O número corporativo de {collaborator_name} é: {user_phone} "
                                f"({user_operator}).")
        else:
            # Caso o colaborador não seja encontrado
            response_context = f"Colaborador {collaborator_name} não encontrado na lista."
    else:
        # Caso o nome não seja identificado ou a solicitação não seja clara
        response_context = "Não consegui identificar o nome do colaborador na sua solicitação."

    # Enviar o contexto de volta à IA principal para uma resposta mais elaborada
    response = conversation_llm.invoke(response_context)
    print(f"Resposta da IA: {response.content}")

# Executar o fluxo principal
if __name__ == "__main__":
    main()
