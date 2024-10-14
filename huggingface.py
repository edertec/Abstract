import requests
import csv
import config  # Import your config.py file

# Your Hugging Face API token
API_TOKEN = config.APY_HF

# Set up headers for the API request
headers = {
    "Authorization": f"Bearer {API_TOKEN}"
}

# Function to call the Hugging Face API for structured summarization
def get_structured_summary(abstract_text):
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    
    # More explicit prompt asking for specific sections in detail
    prompt = (
        f"Please summarize the following abstract into the following sections clearly and concisely:\n\n"
        f"1. Objectives: What was the main purpose or goal of this study?\n"
        f"2. Problem: What problem or challenge was addressed in this study?\n"
        f"3. Data: What data was used in the study? Mention data sources and types of data.\n"
        f"4. Methods_Techniques: What methods or techniques were applied in the study?\n"
        f"5. Results: What were the main findings or results of the study?\n\n"
        f"Abstract: {abstract_text}"
    )
    
    payload = {
        "inputs": prompt,
        "parameters": {"min_length": 50, "max_length": 300}  # Adjust summary length accordingly
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        summary = response.json()[0]['summary_text']
        return summary
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

# Function to append the output to a CSV file immediately after each abstract
def append_to_csv(data, output_filename="structured_summaries.csv"):
    with open(output_filename, mode='a', newline='') as file:  # 'a' for append mode
        writer = csv.writer(file)
        writer.writerow([data.get('Objectives', 'N/A'), data.get('Problem', 'N/A'), data.get('Data', 'N/A'),
                         data.get('Methods_Techniques', 'N/A'), data.get('Results', 'N/A')])

# Function to parse the structured summary into a dictionary
def parse_structured_summary(summary_text):
    # Initialize an empty dictionary
    parsed_data = {
        "Objectives": "N/A",
        "Problem": "N/A",
        "Data": "N/A",
        "Methods_Techniques": "N/A",
        "Results": "N/A"
    }

    # Split the summary text into lines and process each section
    lines = summary_text.split('\n')

    current_section = None
    for line in lines:
        # Check if the line starts with any section label
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

# Function to process abstracts
def process_abstracts(abstracts, output_filename="structured_summaries.csv"):
    for abstract in abstracts:
        print(f"Processing abstract: {abstract[:60]}...")  # Show part of the abstract for tracking
        summary = get_structured_summary(abstract)
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