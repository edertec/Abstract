import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stopwords
import string

# Carregar modelo Spacy
nlp = spacy.load("en_core_web_sm")

# Função para remover stopwords, pontuação e termos irrelevantes
def clean_text(text):
    doc = nlp(text.lower())
    stopwords = nlp.Defaults.stop_words | sklearn_stopwords  # Usar stopwords do Spacy + sklearn
    custom_stopwords = {'study', 'datum', 'apply', 'use', 'technique', 'result', 'findings', 'based', '%', 'abstract', 'problem'}  # Stopwords customizadas
    stopwords |= custom_stopwords
    
    tokens = [token.lemma_ for token in doc if token.text not in string.punctuation and 
              token.text not in stopwords and not token.is_digit and len(token.text) > 2]
    return tokens

# Função para contar as palavras mais frequentes
def get_top_keywords(column_data, top_n=10):
    all_tokens = []
    for text in column_data.dropna():
        all_tokens.extend(clean_text(text))
    return Counter(all_tokens).most_common(top_n)

# Função para gerar os gráficos de palavras-chave
def plot_top_keywords(df, top_n=10):
    # Palavras mais frequentes em Methods
    top_methods = get_top_keywords(df['Methods_Techniques'], top_n)
    
    # Palavras mais frequentes em Results
    top_results = get_top_keywords(df['Results'], top_n)
    
    # Criar o gráfico
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico de Methods
    methods_keywords, methods_counts = zip(*top_methods)
    axs[0].barh(methods_keywords, methods_counts, color='skyblue')
    axs[0].set_title('Top 10 Keywords in Methods')
    axs[0].set_xlabel('Frequency')
    axs[0].invert_yaxis()  # Inverter o eixo y para mostrar o mais frequente no topo
    
    # Gráfico de Results
    results_keywords, results_counts = zip(*top_results)
    axs[1].barh(results_keywords, results_counts, color='salmon')
    axs[1].set_title('Top 10 Keywords in Results')
    axs[1].set_xlabel('Frequency')
    axs[1].invert_yaxis()
    
    plt.tight_layout()
    plt.show()

# Função principal para carregar e executar o código
def main():
    # Carregar o CSV (ajuste o caminho do arquivo se necessário)
    df = pd.read_csv('sintese_estudos.csv')
    
    # Gerar os gráficos para os top 10 métodos e resultados
    plot_top_keywords(df, top_n=10)

if __name__ == "__main__":
    main()