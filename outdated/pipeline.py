import os
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForQuestionAnswering,AutoTokenizer, pipeline
import pinecone
from langchain_openai import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
import os

chat = ChatOpenAI(
    openai_api_key= "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",
    model='gpt-3.5-turbo'
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.system('cls' if os.name == 'nt' else 'clear')
#Initialize infrastructure:
print("Initializing infrastructure...")

chat = ChatOpenAI(
    openai_api_key= "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",
    model='gpt-3.5-turbo'
)

embedding_model_name = 'intfloat/e5-large-v2'
QA_model_name = "deepset/roberta-base-squad2"
QA_tokenizer_name = "deepset/roberta-base-squad2" #https://huggingface.co/deepset/roberta-base-squad2

embedding_model = SentenceTransformer(embedding_model_name)
QA_model = AutoModelForQuestionAnswering.from_pretrained(QA_model_name)
QA_tokenizer = AutoTokenizer.from_pretrained(QA_tokenizer_name)

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
    
    messages = [
    SystemMessage(content="You are a friendly assistant that will answer questions"),
    ]
    
    augmented_prompt = f"""Try to to answer the question with the Chunks. If thats not possible say so and answer it with your own knowledge
    Contexts:
    {context}
    Query: {user_input}"""
    
    prompt = HumanMessage(
        content=augmented_prompt
    )
    
    messages.append(prompt)
    res = chat(messages)
    
    return res

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
