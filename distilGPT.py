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

# Function to extract the problem the study aims to solve
def get_problem_summary(abstract_text):
    # Prompt mais direto para extrair apenas o problema
    prompt = (
        f"Identify and describe the main problem that this study aims to solve in one concise sentence:\n\nAbstract: {abstract_text}"
    )
    
    # Tokenize input com atenção ao preenchimento e truncamento
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    # Gerar resumo usando um número maior de tokens para evitar cortes abruptos
    summary_ids = model.generate(
        inputs["input_ids"], 
        attention_mask=inputs["attention_mask"], 
        max_new_tokens=70,  # Aumentar número de tokens para evitar truncamento
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# Função para salvar o problema no arquivo CSV
def append_problem_to_csv(problem, output_filename="problem_summaries.csv"):
    with open(output_filename, mode='a', newline='') as file:  # 'a' para modo de append
        writer = csv.writer(file)
        writer.writerow([problem])

# Função para processar abstracts e extrair o problema
def process_abstracts(abstracts, output_filename="problem_summaries.csv"):
    for abstract in abstracts:
        print(f"Processing abstract: {abstract[:60]}...")  # Mostrar parte curta do abstract para tracking
        problem = get_problem_summary(abstract)
        
        # Verificar se o problema foi extraído e salvar no CSV
        if problem:
            append_problem_to_csv(problem, output_filename)  # Salvar imediatamente após processamento
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
    with open('problem_summaries.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Problem"])

    # Processar os abstracts e salvar no CSV
    process_abstracts(abstracts)

if __name__ == "__main__":
    main()