# Instale as bibliotecas necessárias:
# pip install langchain openai pypdf faiss-cpu sentence-transformers pillow torchvision

import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from PIL import Image
import torch
from torchvision import models, transforms
import numpy as np

# Configurar o modelo
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4-turbo")

torch.manual_seed(42)
class ImageProjector(nn.Module):
    def __init__(self, input_dim=1000, output_dim=384):
        super(ImageProjector, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projection(x)

projector = ImageProjector()


# Carregar PDF e criar embeddings de texto
def load_pdf_and_create_db(path):
    loader = PyPDFLoader(path)
    docs = loader.load_and_split()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    return docs, db

# Extrair features da imagem usando ResNet
def extract_image_features(image_path):
    model = models.resnet50(pretrained=True)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path)
    image_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        features = model(image_tensor)

    return features.numpy().flatten()

# Buscar documentos similares usando texto e imagem
def multimodal_query(docs, db, query_text, image_path):
    text_embedding = db.embedding_function.embed_query(query_text)
    image_embedding = extract_image_features(image_path)

    combined_embedding = (text_embedding + image_embedding[:len(text_embedding)]) / 2

    similar_docs = db.similarity_search_by_vector(combined_embedding)
    return similar_docs

# Gerar resposta multimodal
def multimodal_rag_answer(docs, db, query_text, image_path):
    similar_docs = multimodal_query(docs, db, query_text, image_path)

    context = " ".join([doc.page_content for doc in similar_docs])

    messages = [
        HumanMessage(content=f"Contexto extraído: {context}\n\nPergunta: {query_text}")
    ]

    response = llm.invoke(messages)
    return response.content

# Exemplo de uso
if __name__ == "__main__":
    pdf_path = "seu_documento.pdf"
    query_image_path = "imagem_tabela.png"
    query_text = "Descreva o conteúdo desta tabela."

    docs, db = load_pdf_and_create_db(pdf_path)

    resposta = multimodal_rag_answer(docs, db, query_text, query_image_path)
    print("Resposta gerada:", resposta)

