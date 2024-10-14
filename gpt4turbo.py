import openai
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import config

# Configurar sua chave API da OpenAI
openai.api_key = config.OPENAI_KEY

# Função para enviar um prompt à API do OpenAI para analisar e correlacionar problemas, métodos e resultados
def analyze_problems_methods_results(texts):
    prompt = f"""As an expert in natural language processing, analyze the following scientific abstracts. 
    Your task is to:
    1. Identify the key problems being addressed in the studies.
    2. List the methods or techniques used to address those problems.
    3. Summarize the main results or findings.
    Please create a clear mapping that shows which problems are addressed by which methods and what results are reported.
    Here are the texts: \n\n{texts}"""

    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3000,
        temperature=0.7
    )

    # Correção para acessar corretamente o conteúdo da resposta
    summary_text = response.choices[0].message.content
    return summary_text

# Função para processar os dados com lotes menores
def process_with_chatgpt(df, batch_size=5):
    abstracts = df['abstract'].dropna().tolist()  # Coleta todos os abstracts
    results = []

    # Dividir os abstracts em lotes
    for i in range(0, len(abstracts), batch_size):
        batch_texts = "\n".join(abstracts[i:i + batch_size])
        try:
            analysis = analyze_problems_methods_results(batch_texts)
            results.append(analysis)
        except openai.error.RateLimitError as e:
            print(f"Rate limit exceeded for batch {i // batch_size + 1}, skipping...")
            continue  # Ignorar erros de limite de taxa e continuar com o próximo lote

    return "\n".join(results)

# Função para exibir a análise de forma organizada
def display_analysis(analysis_text):
    print("Analysis of Problems, Methods, and Results:")
    print(analysis_text)

# Função para gerar gráficos com base nos termos mais comuns
def plot_keywords(counter_data, title):
    keywords, counts = zip(*counter_data.most_common(10))
    plt.barh(keywords, counts, color='skyblue')
    plt.title(title)
    plt.xlabel('Frequency')
    plt.gca().invert_yaxis()
    plt.show()

# Função para gerar contagens de termos em cada seção
def keyword_count_by_section(analysis_text, section):
    lines = analysis_text.split('\n')
    section_found = False
    section_text = []

    for line in lines:
        if section.lower() in line.lower():
            section_found = True
        elif section_found and line.strip() == "":
            break  # Parar quando a seção seguinte começar
        elif section_found:
            section_text.append(line)
    
    # Fazer contagem dos termos
    section_words = " ".join(section_text).split()
    return Counter(section_words)

def main():
    # Carregar os dados
    df = pd.read_csv('sintese_estudos.csv')

    # Processar os abstracts com ChatGPT para obter a análise de problemas, métodos e resultados
    analysis_text = process_with_chatgpt(df)

    # Exibir a análise completa (para revisão)
    display_analysis(analysis_text)

    # Contagem de palavras-chave por seção
    problem_counter = keyword_count_by_section(analysis_text, 'Problems')
    method_counter = keyword_count_by_section(analysis_text, 'Methods')
    results_counter = keyword_count_by_section(analysis_text, 'Results')

    # Gerar gráficos das palavras-chave mais comuns
    plot_keywords(problem_counter, 'Top 10 Keywords in Problems')
    plot_keywords(method_counter, 'Top 10 Keywords in Methods')
    plot_keywords(results_counter, 'Top 10 Keywords in Results')

if __name__ == "__main__":
    main()