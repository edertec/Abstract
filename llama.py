from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import csv

# Use the LLaMA-2 int8 model for better performance on lower hardware
model_name = "meta-llama/Llama-2-7b-hf-int8"  # Use the int8 version of the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move the model to the GPU if available, or use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Function to extract the problem the study aims to solve
def get_problem_summary(abstract_text):
    # Prompt asking only for the problem
    prompt = f"What problem does the study aim to solve?\n\nAbstract: {abstract_text}"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate the summary using reduced max_new_tokens and simplified settings
    summary_ids = model.generate(inputs["input_ids"], max_new_tokens=50, num_beams=1)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    print(f"Extracted Problem:\n{summary}\n")  # Log summary for debugging
    return summary

# Function to append the problem to a CSV file
def append_problem_to_csv(problem, output_filename="problem_summaries.csv"):
    with open(output_filename, mode='a', newline='') as file:  # 'a' for append mode
        writer = csv.writer(file)
        writer.writerow([problem])

# Function to process abstracts and extract the problem
def process_abstracts(abstracts, output_filename="problem_summaries.csv"):
    for abstract in abstracts:
        print(f"Processing abstract: {abstract[:60]}...")  # Show part of the abstract for tracking
        problem = get_problem_summary(abstract)
        
        # Check if problem is valid and append to CSV
        if problem:
            append_problem_to_csv(problem, output_filename)  # Save immediately after processing
        else:
            print("Skipping abstract due to error.")

# Load abstracts from file
def load_abstracts_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()  # Assumes each line is a new abstract

# Main function to execute the process
def main():
    # Load your abstracts (adjust file path as needed)
    abstracts = load_abstracts_from_file('abstracts_list.txt')

    # Initialize the CSV file with headers
    with open('problem_summaries.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Problem"])

    # Process the abstracts and append to CSV
    process_abstracts(abstracts)

if __name__ == "__main__":
    main()