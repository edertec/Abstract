import csv
from transformers import T5ForConditionalGeneration, T5Tokenizer, BartForConditionalGeneration, BartTokenizer

# Choose whether to use T5 or BART model (uncomment one)
MODEL_NAME = 't5-small'  # For T5
# MODEL_NAME = 'facebook/bart-large-cnn'  # For BART

# Load the pre-trained model and tokenizer
if 't5' in MODEL_NAME:
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
else:
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)

# Function to summarize the abstract
def get_structured_summary(abstract_text):
    # Add 'summarize:' prefix for T5, not necessary for BART
    input_text = f"summarize: {abstract_text}" if 't5' in MODEL_NAME else abstract_text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate the summary
    summary_ids = model.generate(input_ids, max_length=300, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Print and return the generated summary
    print(f"Generated Summary:\n{summary}\n")  # Log summary for debugging
    return summary

# Function to append the output to a CSV file
def append_to_csv(data, output_filename="structured_summaries.csv"):
    with open(output_filename, mode='a', newline='') as file:  # 'a' for append mode
        writer = csv.writer(file)
        writer.writerow([data.get('Objectives', 'N/A'), data.get('Problem', 'N/A'), data.get('Data', 'N/A'),
                         data.get('Methods_Techniques', 'N/A'), data.get('Results', 'N/A')])

# Function to parse the summary
def parse_structured_summary(summary_text):
    # Initialize an empty dictionary
    parsed_data = {
        "Objectives": "N/A",
        "Problem": "N/A",
        "Data": "N/A",
        "Methods_Techniques": "N/A",
        "Results": "N/A"
    }

    # Check if summary is empty
    if not summary_text or summary_text.strip() == "":
        print("Empty summary generated, skipping.")
        return parsed_data

    # Check if summary contains meaningful text
    lines = summary_text.split('\n')
    if len(lines) == 0:
        print("No lines found in summary, skipping.")
        return parsed_data

    # Try to extract known sections
    for line in lines:
        if "Objectives" in line:
            parsed_data["Objectives"] = line.split(":", 1)[1].strip() if ":" in line else "N/A"
        elif "Problem" in line:
            parsed_data["Problem"] = line.split(":", 1)[1].strip() if ":" in line else "N/A"
        elif "Data" in line:
            parsed_data["Data"] = line.split(":", 1)[1].strip() if ":" in line else "N/A"
        elif "Methods_Techniques" in line:
            parsed_data["Methods_Techniques"] = line.split(":", 1)[1].strip() if ":" in line else "N/A"
        elif "Results" in line:
            parsed_data["Results"] = line.split(":", 1)[1].strip() if ":" in line else "N/A"
    
    return parsed_data

# Function to process abstracts
def process_abstracts(abstracts, output_filename="structured_summaries.csv"):
    for abstract in abstracts:
        print(f"Processing abstract: {abstract[:60]}...")  # Show part of the abstract for tracking
        summary = get_structured_summary(abstract)
        
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