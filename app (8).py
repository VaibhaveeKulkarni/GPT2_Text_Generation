
from transformers import pipeline

# Initialize the text generation pipeline
generator = pipeline("text-generation", model="gpt2")

def generate_text(prompt, max_length=100, num_return_sequences=1):
    ""Generates text using the pre-trained GPT-2 model."""
    print(f"
Generating text for prompt: '{prompt}'")
    result = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=num_return_sequences
    )
    return result[0]["generated_text"]

if __name__ == "__main__":
    # Example usage
    sample_prompt = "The future of Artificial Intelligence is"
    generated = generate_text(sample_prompt)
    print("
--- Generated Text ---")
    print(generated)
