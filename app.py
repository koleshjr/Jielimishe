import os
import streamlit as st
from langchain.vectorstores import Qdrant
from langchain.embeddings.cohere import CohereEmbeddings
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from qdrant_client import QdrantClient

load_dotenv()

openai_api_key = os.getenv("OPEN_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")

prompt_template = """
Perform the following actions:
1. Answer the question based on the text provided .
2. Translate the answer in the language used by the user.
3. If the text doesn't contain the answer, reply that the answer is not available.

Text: {context}
Question: {question}
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
def run_qa_chain(qdrant, query):
    search_results = qdrant.similarity_search(query, k=2)
    chain = load_qa_chain(OpenAI(openai_api_key=openai_api_key,temperature=0.2), chain_type="stuff", prompt = PROMPT)
    results = chain({"input_documents": search_results, "question": query}, return_only_outputs=True)
    answer = results['output_text']
    return answer

client = QdrantClient(url=qdrant_url, prefer_grpc=True, api_key=qdrant_api_key)
qdrant = Qdrant(client=client, collection_name="texts",embedding_function=embeddings.embed_query)

st.title("Jielimishe Kuhusu Katiba")

query = st.text_input("Uliza Swali:")
if query:
    results = run_qa_chain(qdrant, query)
    st.write(results)
