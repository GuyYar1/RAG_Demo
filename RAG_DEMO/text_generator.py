import re

def generate_text_in_chunks(input_text, model, tokenizer, max_length=512):
    """
    Generate text in chunks using a pre-trained language model.

    Args:
    input_text (str): The input text to be processed.
    model: The pre-trained language model.
    tokenizer: The tokenizer corresponding to the model.
    max_length (int): The maximum length of each chunk.

    Returns:
    str: The generated text.
    """
    chunks = [input_text[i:i + max_length] for i in range(0, len(input_text), max_length)]
    generated_text = ""

    try:
        for chunk in chunks:
            # If an error happens in this loop (either here or in model.generate)
            print(f"Chunk length: {len(chunk)}")
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=max_length)

            # If tokenizer or model.generate fails, this will throw an exception
            outputs = model.generate(
                **inputs,
                max_length=max_length + 50,
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id
            )

            # Only this much text would be added up to the point of failure
            generated_text += tokenizer.decode(outputs[0], skip_special_tokens=True)

    # If an exception occurs anywhere in the try block, this will catch it
    except Exception as e:
        print(f"Error during tokenization or generation: {e}")

    # Return the generated text (which might be incomplete due to the exception)
    return generated_text


# Rerank Function to extract requested items from the response
def rerank_response1(query, response_text):

    # Extract the number of items from the query (e.g., "5" in "5 crime movies")
    number_of_items = int(re.search(r'\d+', query).group())  # Find the first number in the query
    genre = re.search(r'(\w+)\s+movies', query, re.IGNORECASE).group(1)  # Extract genre (e.g., "crime")

    # Split the response into a list of lines for easier processing
    lines = response_text.splitlines()

    # Filter the lines that mention the genre (e.g., "crime")
    filtered_lines = [line for line in lines if genre.lower() in line.lower()]

    # Extract only the number of items requested
    reranked_lines = filtered_lines[:number_of_items]

    # Join the reranked lines into a formatted string and return
    return "\n".join(reranked_lines)

def rerank_response(items, keyword):
    """
    Filter and rerank items based on a keyword (e.g., genre).

    Args:
    items (list of str): List of items (e.g., movie titles or descriptions).
    keyword (str): Keyword to filter items by.

    Returns:
    list of str: Filtered and reranked list of items.
    """
    # Define a regular expression pattern to split the text into sentences


    print("rerank_response")

    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'

    # Split the text into sentences
    sentences = re.split(pattern, items)

    # Filter sentences containing the keyword
    filtered_sentences = [sentence for sentence in sentences if keyword.lower() in sentence.lower()]

    print(f"Reranking find {len(filtered_sentences)} sentences")
    for sentence in filtered_sentences:
        print(f"- {sentence}")
    # You can add further ranking logic here if needed
    # For simplicity, we'll just return the filtered sentences
    return filtered_sentences


def extract_sentences_with_keyword(documents, keyword):
    filtered_sentences = []

    # Check the type of documents
    print(f"Type of documents before function call: {type(documents)}")

    # Ensure documents is a list and keyword is a string
    if not isinstance(documents, list):
        raise TypeError("documents should be a list")
    if not isinstance(keyword, str):
        raise TypeError("keyword should be a string")

    # Iterate through the documents list
    for doc_id, text in enumerate(documents):
        # Split the text into sentences
        sentences = re.split(r'\.\s*|\n', text)  # Split by periods or newlines
        for sentence in sentences:
            # Check if the keyword is in the sentence
            if keyword.lower() in sentence.lower():
                filtered_sentences.append({
                    'doc_id': doc_id,
                    'sentence': sentence.strip()
                })

    print(f"Found {len(filtered_sentences)} sentences containing the keyword '{keyword}'")
    return filtered_sentences


