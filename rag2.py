# Instale as bibliotecas necessárias:
# pip install langchain openai pypdf pillow torchvision sentence-transformers pdf2image faiss-cpu

import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from pdf2image import convert_from_path
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms
import numpy as np

# Configurar o modelo de linguagem
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4-turbo")

# Projeção linear para ajustar dimensão dos embeddings de imagem
torch.manual_seed(42)
class ImageProjector(nn.Module):
    def __init__(self, input_dim=1000, output_dim=384):
        super(ImageProjector, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projection(x)

projector = ImageProjector()

# Extrair embeddings da imagem usando ResNet
def extract_image_features(image):
    model = models.resnet50(pretrained=True)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = image.convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        features = model(image_tensor)

    # Projeção linear para ajustar dimensão
    projected_features = projector(features).squeeze()
    return projected_features.numpy()

# Converter PDF em imagens
def pdf_to_images(pdf_path):
    return convert_from_path(pdf_path)

# Buscar a página mais similar usando embeddings de imagem
def find_most_similar_page(pdf_images, query_image_path):
    query_image = Image.open(query_image_path)
    query_embedding = extract_image_features(query_image)

    similarities = []
    for page_image in pdf_images:
        page_embedding = extract_image_features(page_image)
        similarity = np.dot(query_embedding, page_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(page_embedding))
        similarities.append(similarity)

    most_similar_index = np.argmax(similarities)
    return most_similar_index, pdf_images[most_similar_index]

# Gerar resposta baseada na página mais similar
def multimodal_rag_answer(pdf_images, query_image_path, query_text):
    page_index, similar_page_image = find_most_similar_page(pdf_images, query_image_path)

    # Aqui poderia usar OCR para extrair texto da imagem se quiser combinar com texto
    # Mas neste exemplo, apenas informamos a página encontrada

    messages = [
        HumanMessage(content=f"Encontrei a página {page_index + 1} mais parecida com a imagem fornecida. Descreva o conteúdo desta página com base na pergunta: {query_text}.")
    ]

    response = llm.invoke(messages)
    return response.content

# Exemplo de uso
if __name__ == "__main__":
    pdf_path = "seu_documento.pdf"
    query_image_path = "imagem_tabela.png"
    query_text = "Descreva o conteúdo desta tabela."

    pdf_images = pdf_to_images(pdf_path)

    resposta = multimodal_rag_answer(pdf_images, query_image_path, query_text)
    print("Resposta gerada:", resposta)
