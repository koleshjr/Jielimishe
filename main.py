import os
from langchain.vectorstores import Qdrant
from langchain.embeddings.cohere import CohereEmbeddings
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from qdrant_client import QdrantClient
load_dotenv()

openai_api_key = os.getenv("OPEN_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")






query = "je, hati hii inazungumzia nini?"

client = QdrantClient(url=qdrant_url, prefer_grpc=True, api_key=qdrant_api_key)

embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
qdrant = Qdrant(client=client, collection_name="texts", embedding_function=embeddings.embed_query)
search_results = qdrant.similarity_search(query, k=2)
chain = load_qa_chain(OpenAI(openai_api_key=openai_api_key,temperature=0.2), chain_type="stuff")
results = chain({"input_documents": search_results, "question": query}, return_only_outputs=True)

print (results)