from docling import PDFDoc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_tables_text(pdf_path):
    doc = PDFDoc(pdf_path)
    tables_text = []
    for page in doc.pages:
        tables = page.tables
        for table in tables:
            # Concatena os textos das células para formar um texto único da tabela
            table_text = '\n'.join(['\t'.join(row) for row in table])
            tables_text.append(table_text)
    return tables_text

def compare_tables(base_tables, comparison_tables):
    # Vetoriza o texto das tabelas para comparação de similaridade
    vectorizer = TfidfVectorizer()
    base_vectors = vectorizer.fit_transform(base_tables)
    comp_vectors = vectorizer.transform(comparison_tables)
    
    # Calcula a similaridade entre cada tabela base e cada tabela de comparação
    similarities = cosine_similarity(base_vectors, comp_vectors)
    
    # Para cada tabela base, encontra a tabela mais similar nos PDFs de comparação
    best_matches = similarities.argmax(axis=1)
    best_scores = similarities.max(axis=1)
    return best_matches, best_scores

# 1. Extração do PDF multipágina
multi_page_pdf = "multi_page.pdf"
base_tables = extract_tables_text(multi_page_pdf)

# 2. Extração dos PDFs de uma página
single_page_pdfs = ["page1.pdf", "page2.pdf", "page3.pdf"]  # Exemplo
comparison_tables = []
pdf_index = []

for idx, pdf in enumerate(single_page_pdfs):
    tables = extract_tables_text(pdf)
    comparison_tables.extend(tables)
    pdf_index.extend([idx] * len(tables))  # Marca de qual PDF veio cada tabela

# 3. Comparação
best_matches, best_scores = compare_tables(base_tables, comparison_tables)

# Resultado final
for i, (match_idx, score) in enumerate(zip(best_matches, best_scores)):
    matched_pdf = single_page_pdfs[pdf_index[match_idx]]
    print(f"Tabela {i} do PDF multipágina corresponde melhor com tabela {match_idx} do PDF '{matched_pdf}' (similaridade: {score:.2f})")