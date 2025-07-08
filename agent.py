import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    contas: pd.DataFrame
    demonstracao_texto: str

loader = PyPDFLoader('demonstracao_financeira.pdf')
docs = loader.load()
demo_financeira_texto = "\n".join([doc.page_content for doc in docs])

planilha_contas = pd.read_excel('contas_contabeis.xlsx')

llm = ChatOpenAI(model="gpt-3.5-turbo")

prompt = ChatPromptTemplate.from_template("""
Extraia o valor e o nome exato da conta associada à conta contábil informada abaixo:

Texto da Demonstração Financeira:
{demo_text}

Conta a buscar: {conta}

Retorne no formato: nome_da_conta_no_pdf:valor ou "Não encontrado:0" caso não exista.
""")

parser = StrOutputParser()
chain = prompt | llm | parser

prompt_contas_pdf = ChatPromptTemplate.from_template("""
Liste todas as contas do Ativo Circulante presentes no texto da Demonstração Financeira abaixo:

Texto da Demonstração Financeira:
{demo_text}

Retorne as contas separadas por ponto e vírgula (;).
""")

chain_contas_pdf = prompt_contas_pdf | llm | parser

async def planilhador_ativo_circulante(state: AgentState):
    contas_df = state['contas']
    demo_text = state['demonstracao_texto']
    resultados = []
    contas_extraidas = set()
    for conta in contas_df['Conta Contábil']:
        resposta = await chain.ainvoke({
            "demo_text": demo_text,
            "conta": conta
        })
        nome_conta_pdf, valor = resposta.split(":")
        valor_float = float(valor.replace(",", "").replace(".", ""))
        if nome_conta_pdf in contas_extraidas and nome_conta_pdf != "Não encontrado":
            raise ValueError(f"Erro: A conta '{nome_conta_pdf}' do PDF foi preenchida mais de uma vez.")
        resultados.append(valor_float)
        if nome_conta_pdf != "Não encontrado":
            contas_extraidas.add(nome_conta_pdf)
    contas_df['Valor Extraído'] = resultados
    total_contas = contas_df['Valor Extraído'].sum()
    total_ativo_circulante_resp = await chain.ainvoke({
        "demo_text": demo_text,
        "conta": "Total do Ativo Circulante"
    })
    _, total_ativo_circulante = total_ativo_circulante_resp.split(":")
    total_ativo_circulante = float(total_ativo_circulante.replace(",", "").replace(".", ""))
    if total_contas != total_ativo_circulante:
        diferenca = total_ativo_circulante - total_contas
        contas_df.loc[contas_df['Conta Contábil'] == 'Outros ativos', 'Valor Extraído'] += diferenca
    contas_pdf_resposta = await chain_contas_pdf.ainvoke({"demo_text": demo_text})
    contas_pdf_set = set(conta.strip() for conta in contas_pdf_resposta.split(";"))
    if not contas_pdf_set.issubset(contas_extraidas):
        faltantes = contas_pdf_set - contas_extraidas
        raise ValueError(f"Erro: As seguintes contas do PDF não foram planilhadas: {', '.join(faltantes)}")
    return {'contas': contas_df, 'demonstracao_texto': demo_text}

workflow = StateGraph(AgentState)
workflow.add_node("planilhador_ativo_circulante", planilhador_ativo_circulante)
workflow.set_entry_point("planilhador_ativo_circulante")
workflow.add_edge("planilhador_ativo_circulante", END)
app = workflow.compile()

import asyncio

state = {
    "contas": planilha_contas,
    "demonstracao_texto": demo_financeira_texto
}

resultado_final = asyncio.run(app.ainvoke(state))
resultado_final['contas'].to_excel('resultado_ativo_circulante.xlsx', index=False




from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from typing import TypedDict, Optional

# Estado da conversa
class WorkflowState(TypedDict):
    empresa: str
    ponto_de_atencao: Optional[str]
    analise_final: Optional[str]

# LLM base
llm = ChatOpenAI(model="gpt-4", temperature=0.2)

# ------------------- AGENTE ANALISTA 1 -------------------
def agente_analista_1(state: WorkflowState) -> WorkflowState:
    empresa = state["empresa"]
    
    prompt = f"""
Você é um analista financeiro. Avalie o seguinte texto sobre uma empresa e diga se há algum ponto de atenção.
Se houver, explique em uma frase. Caso contrário, diga "Nenhum ponto de atenção encontrado".

Texto: "{empresa}"
"""
    resposta = llm.invoke(prompt).content.strip()
    
    if "nenhum ponto" in resposta.lower():
        return {"empresa": empresa, "ponto_de_atencao": None}
    else:
        return {"empresa": empresa, "ponto_de_atencao": resposta}

# ------------------- SUPERVISOR -------------------
def supervisor(state: WorkflowState) -> str:
    if state.get("ponto_de_atencao"):
        return "chamar_analista_2"
    return END

# ------------------- AGENTE ANALISTA 2 -------------------
def agente_analista_2(state: WorkflowState) -> WorkflowState:
    ponto = state["ponto_de_atencao"]
    
    prompt = f"""
Você é um analista sênior de riscos. O seguinte ponto de atenção foi identificado:
"{ponto}"

Faça uma análise mais aprofundada desse ponto de atenção, considerando possíveis causas e consequências para a empresa.
"""
    resposta = llm.invoke(prompt).content.strip()
    return {"analise_final": resposta}

# ------------------- CONSTRUÇÃO DO GRAFO -------------------
graph = StateGraph(WorkflowState)

graph.add_node("analista_1", RunnableLambda(agente_analista_1))
graph.add_node("supervisor", RunnableLambda(supervisor))
graph.add_node("analista_2", RunnableLambda(agente_analista_2))

graph.set_entry_point("analista_1")
graph.add_edge("analista_1", "supervisor")
graph.add_conditional_edges("supervisor", {
    "chamar_analista_2": "analista_2",
    END: END
})
graph.add_edge("analista_2", END)

workflow = graph.compile()