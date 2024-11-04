# If you run Chroma in Docker, specify the URL
#client = Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))
#client.delete_collection(name="faq_answers")
# If you want to connect to a running server
# client = Client("http://localhost:8000")

from chromadb import PersistentClient

from sentence_transformers import SentenceTransformer
import numpy as np
import json

client = PersistentClient(path="/home/starias/Smart-FAQ-Assistant/data/chroma_db/")

collection = client.get_or_create_collection(name="faq_answers", metadata={"hnsw:space": "cosine"})



# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_data(data_path="/home/starias/Smart-FAQ-Assistant/data/faq_data.json"):
    # Load data from the JSON file
    with open(data_path, 'r') as f:
        corpus = json.load(f)

    # Insert data into Chroma DB
    for idx, item in enumerate(corpus):
        embedding = model.encode(item['question'], convert_to_tensor=True).tolist()  # Convert to list for storage
        # Generate a unique ID for each document
        collection.add(
            documents=[item['question']], 
            embeddings=[embedding], 
            metadatas=[{"answer": item['answer']}], 
            ids=[f"doc_{idx}"]
        )

def get_answer_from_db(user_question, similarity_threshold=0.7):  # Adjust threshold as needed
    not_found = "Sorry, I couldn't find an answer to your question."
    original_question = None
    similarity = None
    # Encode the user question to create an embedding
    user_embedding = model.encode(user_question, convert_to_tensor=True).tolist()

    # Query the collection for the most similar question
    results = collection.query(user_embedding, n_results=5)  # Retrieve top 5 matches for better thresholding

    # Print the results for debugging
    print("Results:", results)

    # Check if there are any results
    if results and results['documents']:
        # Access the cosine distances directly
        cosine_distances = np.array(results['distances'][0])  # Retrieve distances as they are
        print("Cosine Distances:", cosine_distances)

        # Calculate cosine similarities from distances
        cosine_similarities = 1 - cosine_distances  # Convert distances to similarities
        print("Cosine Similarities:", cosine_similarities)

        best_similarity = cosine_similarities[0]  # Highest similarity indicates the best match
        print("Best Similarity:", best_similarity)

        # Filter based on the similarity threshold
        if best_similarity < similarity_threshold:  # Adjusted comparison for similarity
            answer = not_found
            
            return (answer, None, None)
        best_answer = results['metadatas'][0][0]['answer']
        
        answer = best_answer
        original_question = results['documents'][0][0]
        similarity = float(round(best_similarity,2))
        
    else:
        answer = not_found
        
    return (answer, original_question, similarity)


if __name__ == "__main__":
    load_data()

    
    user_input = "Fuck you"
    user_input = "Python uninstall"
    user_input = "What is Airflow?"
    answer = get_answer_from_db(user_input)
    print("Answer:", answer)

