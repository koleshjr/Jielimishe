import os
from langchain.vectorstores import Qdrant
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from dotenv import load_dotenv
from consts import llm_model_cohere,llm_model_openai
load_dotenv()

openai_api_key = os.getenv("OPEN_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")

#loading pdf files
loader = DirectoryLoader(
    "documents/",
    glob = "**/*.pdf",
    loader_cls =PyPDFLoader,
)

documents = loader.load()

embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
qdrant = Qdrant.from_documents(documents, embeddings, url=qdrant_url, collection_name='texts', prefer_grpc=True, api_key=qdrant_api_key)


# pinecone.init(
#     api_key=pinecone_api_key , # find at app.pinecone.io
#     environment="asia-southeast1-gcp",  # next to api key in console
# )
# index_name = 'zindi'

# vectordb= Pinecone.from_documents(texts, embeddings, index_name=index_name)
