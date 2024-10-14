import openai
import asyncio
import config

# Set up the OpenAI API key
openai.api_key = config.OPENAI_KEY

# Use an async function for the API call
async def get_abstract_analysis(abstract_text):
    response = await openai.chat.completions.create(  # note 'acreate' for async
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": "You are an expert summarizer."},
            {"role": "user", "content": f"Extract the following fields from this abstract:\n\nAbstract: {abstract_text}\n\n1. Objectives\n2. Problem\n3. Data\n4. Methods and Techniques\n5. Results"}
        ]
    )
    return response['choices'][0]['message']['content']

# Run the async function in the main event loop
if __name__ == "__main__":
    abstract_text = "The workplace influences the safety, health, and productivity of workers..."
    result = asyncio.run(get_abstract_analysis(abstract_text))
    print(result)