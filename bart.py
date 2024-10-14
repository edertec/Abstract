from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import csv

# Load the BART model and tokenizer from Hugging Face
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Move the model to the GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Function to summarize the abstract and extract elements using BART
def get_bart_summary(abstract_text):
    # Modify the prompt to request specific elements
    prompt = (
        f"Extract the following details from the abstract:\n\n"
        f"1. Objectives: What was the main goal of this study?\n"
        f"2. Problem: What problem or challenge was addressed?\n"
        f"3. Data: What data was used in this study?\n"
        f"4. Methods_Techniques: What methods or techniques were applied?\n"
        f"5. Results: What were the main findings or results?\n\n"
        f"Abstract: {abstract_text}"
    )
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate the summary
    summary_ids = model.generate(inputs["input_ids"], max_length=300, min_length=50, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    print(f"Generated Summary:\n{summary}\n")  # Log summary for debugging
    return summary

# Function to parse the generated summary into structured sections
def parse_structured_summary(summary_text):
    # Initialize an empty dictionary
    parsed_data = {
        "Objectives": "N/A",
        "Problem": "N/A",
        "Data": "N/A",
        "Methods_Techniques": "N/A",
        "Results": "N/A"
    }

    # Split the summary into lines and process each section
    lines = summary_text.split('\n')

    current_section = None
    for line in lines:
        # Identify and extract each section based on its label
        if "Objectives" in line:
            parsed_data["Objectives"] = line.split("Objectives:", 1)[1].strip() if "Objectives:" in line else "N/A"
        elif "Problem" in line:
            parsed_data["Problem"] = line.split("Problem:", 1)[1].strip() if "Problem:" in line else "N/A"
        elif "Data" in line:
            parsed_data["Data"] = line.split("Data:", 1)[1].strip() if "Data:" in line else "N/A"
        elif "Methods_Techniques" in line:
            parsed_data["Methods_Techniques"] = line.split("Methods_Techniques:", 1)[1].strip() if "Methods_Techniques:" in line else "N/A"
        elif "Results" in line:
            parsed_data["Results"] = line.split("Results:", 1)[1].strip() if "Results:" in line else "N/A"

    return parsed_data

# Function to append the output to a CSV file
def append_to_csv(data, output_filename="structured_summaries.csv"):
    with open(output_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([data['Objectives'], data['Problem'], data['Data'], data['Methods_Techniques'], data['Results']])

# Function to process abstracts and extract the needed elements
def process_abstracts(abstracts, output_filename="structured_summaries.csv"):
    for abstract in abstracts:
        print(f"Processing abstract: {abstract[:60]}...")  # Show part of the abstract for tracking
        summary = get_bart_summary(abstract)
        
        # Check if summary is valid
        if summary:
            parsed_data = parse_structured_summary(summary)
            append_to_csv(parsed_data, output_filename)  # Save immediately after processing
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
    with open('structured_summaries.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Objectives", "Problem", "Data", "Methods_Techniques", "Results"])

    # Process the abstracts and append to CSV
    process_abstracts(abstracts)

if __name__ == "__main__":
    main()