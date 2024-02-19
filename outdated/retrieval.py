import os
from sentence_transformers import SentenceTransformer
import pinecone

os.system('cls' if os.name == 'nt' else 'clear')
#Initialize infrastructure:

print("Initializing infrastructure for information retrieval...")

embedding_model_name = 'intfloat/e5-large-v2'
embedding_model = SentenceTransformer(embedding_model_name)

pinecone.init(api_key="0fde16e6-29bf-4ed6-846b-06cbbb93739e", environment="gcp-starter")
index = pinecone.Index("paragraph-chunks")

print("Done! Welcome to the pipeline, write 'quit' to exit.")

def main():
    while(True):
        user_input = input("Please enter your question: ")
        if(user_input == "quit"):
            break;
        print("Answer:",pipeline(user_input))
    
def pipeline(user_input):
    
    embedded_query = embed_query(user_input)

    top_k = return_document(embedded_query)

    context = get_data_from(top_k)
    
    return context

def embed_query(user_input):
    return embedding_model.encode("query: " + user_input)

def get_data_from(top_k):
    context = ""
    for idx, match in enumerate(top_k["matches"]):
        context = context + "Chunk " + str(idx) + ": " + match["metadata"]["text"]
    return context

def return_document(query_vector):
  query_response = index.query(
      query_vector.tolist(),
      top_k=1,
      include_metadata=True
      )
  return query_response

main()
