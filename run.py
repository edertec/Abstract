import openai
import csv
import time
import config

# Set up the OpenAI API key
openai.api_key = config.OPENAI_KEY

# Function to call the OpenAI API using GPT-4-turbo
def get_abstract_summary(abstract_text):
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",  # Use GPT-4-turbo for cost savings
            messages=[
                {"role": "system", "content": "You are an expert summarizer."},
                {"role": "user", "content": f"Summarize the following abstract: {abstract_text}"}
            ],
            # max_tokens=300  # Adjust based on the size of your abstracts
        )
        # Correct way to access the content from the response
        summary_text = response['choices'][0]['message']['content']

        # Log the response to understand its structure
        print("Raw API response:")
        print(summary_text)
        
        return summary_text
    except Exception as e:
        print(f"Error processing abstract: {e}")
        return None  # Return None if there's an error

# Function to parse the content into a dictionary
def parse_analysis(analysis_text):
    # Log the raw response to see what's returned
    print("Raw API response for analysis:")
    print(analysis_text)

    sections = ['Objectives', 'Problem', 'Data', 'Methods and Techniques', 'Results']
    parsed_data = {}

    # Split the text into lines
    lines = analysis_text.split('\n')
    
    current_section = None
    for line in lines:
        for section in sections:
            if line.startswith(section):
                current_section = section
                parsed_data[section] = line.replace(f"{section}:", "").strip()
                break
        else:
            if current_section:
                parsed_data[current_section] += " " + line.strip()
    
    # Log parsed data to verify if it's correct
    print("Parsed data:")
    print(parsed_data)
    
    return parsed_data

# Function to append the output to a CSV file immediately after each abstract
def append_to_csv(data, output_filename="processed_abstracts.csv"):
    with open(output_filename, mode='a', newline='') as file:  # 'a' for append mode
        writer = csv.writer(file)
        writer.writerow([data.get('Objectives', 'N/A'), data.get('Problem', 'N/A'), data.get('Data', 'N/A'),
                         data.get('Methods and Techniques', 'N/A'), data.get('Results', 'N/A')])

# Function to process abstracts
def process_abstracts(abstracts, output_filename="processed_abstracts.csv"):
    for abstract in abstracts:
        print(f"Processing abstract: {abstract[:60]}...")  # Show part of the abstract for tracking
        summary = get_abstract_summary(abstract)
        if summary:
            parsed_data = parse_analysis(summary)
            append_to_csv(parsed_data, output_filename)  # Save immediately after processing
            time.sleep(1)  # Add a short delay to prevent overwhelming the API
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
    with open('processed_abstracts.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Objectives", "Problem", "Data", "Methods_Techniques", "Results"])

    # Process the abstracts and append to CSV
    process_abstracts(abstracts)

if __name__ == "__main__":
    main()