import openai
import asyncio

# Set up the OpenAI API key
openai.api_key = 'sk-teUzuN70Nh0-zJZry6GWUA2v3y1zU6VQwCKqQOsL5aT3BlbkFJQ4zfosPYAfYhp3N4EZk0OSGJsT3PCw7viVhI3vj8gA'

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