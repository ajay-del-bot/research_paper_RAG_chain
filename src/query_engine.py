from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
import os

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY2')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
index_name = os.environ.get('INDEX_NAME')

pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)

vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentences maximum and keep the answer concise.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

def get_answer(question: str):
    # Retrieve relevant document chunks
    docs = retriever.invoke(question)
    
    # Print or collect references
    references = []
    print("\n=== Retrieved Context ===")
    for i, doc in enumerate(docs):
        # If your doc object has metadata, include it here
        ref_info = {
            "content": doc.page_content,
            "metadata": doc.metadata if hasattr(doc, "metadata") else {}
        }
        references.append(ref_info)
        print(f"\n--- Doc {i+1} ---\n{doc.page_content}\n")
        if hasattr(doc, "metadata"):
            print(f"Metadata: {doc.metadata}")

    # Generate answer
    response = rag_chain.invoke({"input": question})
    answer = response["answer"]

    # Return both answer and references
    return {
        "answer": answer,
        "references": references
    }
