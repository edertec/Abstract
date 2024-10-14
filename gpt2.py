from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import csv

# Usar GPT-2 padrão com precisão de 16 bits para otimizar a memória
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Definir explicitamente o token de preenchimento (pad_token)
tokenizer.pad_token = tokenizer.eos_token

# Move the model to the CPU, usando precisão de 16 bits
device = "cpu"
model = model.to(device)

# Function to extract structured summary from the abstract
def get_structured_summary(abstract_text):
    # Prompt detalhado para extrair todos os elementos
    prompt = (
        f"Summarize the following abstract into the following sections:\n"
        f"1. Objectives: What was the main purpose or goal of this study?\n"
        f"2. Problem: What problem or challenge was addressed in this study?\n"
        f"3. Data: What data was used in the study?\n"
        f"4. Methods_Techniques: What methods or techniques were applied?\n"
        f"5. Results: What were the main findings or results of the study?\n\n"
        f"Abstract: {abstract_text}"
    )
    
    # Tokenize input com atenção ao preenchimento e truncamento
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    # Gerar resumo usando um número maior de tokens para evitar cortes abruptos
    summary_ids = model.generate(
        inputs["input_ids"], 
        attention_mask=inputs["attention_mask"], 
        max_new_tokens=150,  # Aumentar tokens para capturar tudo
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# Função para salvar os dados estruturados no CSV
def append_to_csv(data, output_filename="structured_summaries.csv"):
    with open(output_filename, mode='a', newline='') as file:  # 'a' para modo de append
        writer = csv.writer(file)
        writer.writerow([data.get('Objectives', 'N/A'), data.get('Problem', 'N/A'), data.get('Data', 'N/A'),
                         data.get('Methods_Techniques', 'N/A'), data.get('Results', 'N/A')])

# Função para processar o resumo estruturado
def parse_structured_summary(summary_text):
    # Inicializa um dicionário vazio
    parsed_data = {
        "Objectives": "N/A",
        "Problem": "N/A",
        "Data": "N/A",
        "Methods_Techniques": "N/A",
        "Results": "N/A"
    }

    # Quebra o texto do resumo em linhas e processa cada seção
    lines = summary_text.split('\n')

    current_section = None
    for line in lines:
        # Verifica se a linha começa com algum rótulo de seção
        if line.startswith("Objectives:"):
            current_section = "Objectives"
            parsed_data["Objectives"] = line.replace("Objectives:", "").strip()
        elif line.startswith("Problem:"):
            current_section = "Problem"
            parsed_data["Problem"] = line.replace("Problem:", "").strip()
        elif line.startswith("Data:"):
            current_section = "Data"
            parsed_data["Data"] = line.replace("Data:", "").strip()
        elif line.startswith("Methods_Techniques:"):
            current_section = "Methods_Techniques"
            parsed_data["Methods_Techniques"] = line.replace("Methods_Techniques:", "").strip()
        elif line.startswith("Results:"):
            current_section = "Results"
            parsed_data["Results"] = line.replace("Results:", "").strip()

    return parsed_data

# Função para processar os abstracts
def process_abstracts(abstracts, output_filename="structured_summaries.csv"):
    for abstract in abstracts:
        print(f"Processing abstract: {abstract[:60]}...")  # Mostrar parte curta do abstract para tracking
        summary = get_structured_summary(abstract)
        
        # Verificar se o resumo foi gerado corretamente e processar o texto
        if summary:
            parsed_data = parse_structured_summary(summary)
            append_to_csv(parsed_data, output_filename)  # Salvar imediatamente após processamento
        else:
            print("Skipping abstract due to error.")

# Carregar abstracts de um arquivo
def load_abstracts_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()  # Assume que cada linha é um novo abstract

# Função principal para executar o processo
def main():
    # Carregar seus abstracts (ajuste o caminho do arquivo conforme necessário)
    abstracts = load_abstracts_from_file('abstracts_list.txt')

    # Inicializar o arquivo CSV com cabeçalhos
    with open('structured_summaries.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Objectives", "Problem", "Data", "Methods_Techniques", "Results"])

    # Processar os abstracts e salvar no CSV
    process_abstracts(abstracts)

if __name__ == "__main__":
    main()