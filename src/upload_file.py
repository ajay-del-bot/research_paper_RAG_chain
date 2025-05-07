from collections import defaultdict
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_pinecone import PineconeVectorStore
import re
from nltk.corpus import stopwords
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
index_name = os.environ.get('INDEX_NAME')

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)


# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Create Pinecone index (run once or check before creating)
def create_index():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Check and create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    print(f"Index '{index_name}' not found. Creating it now...")
    create_index()

def load_pdf_file(file_path: str):
    loader = PyPDFLoader(file_path)
    document = loader.load()
    return document

def load_pdfs_from_directory(directory_path: str):
    loader = DirectoryLoader(directory_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"Page \d+|Header Text|Footer Text", "", text)
    stop_words = set(stopwords.words("english"))
    pattern = r'\b(' + '|'.join(map(re.escape, stop_words)) + r')\b'
    result = re.sub(pattern, '', text)
    text = re.sub(r"\s+", " ", result).strip()
    return text

def group_docs_by_path(documents):
    grouped_docs = defaultdict(list)
    final_grouped_docs = defaultdict(list)
    for doc in documents:
        source = doc.metadata['source']
        page = {
            'title': doc.metadata.get('title'),
            'total_pages': doc.metadata.get('total_pages'),
            'page': doc.metadata.get('page'),
            'page_label': doc.metadata.get('page_label'),
            'page_content': doc.page_content
        }
        grouped_docs[source].append(page)

    for source, pages in grouped_docs.items():
        if not pages:
            continue
        page_content = ''.join([p['page_content'] for p in pages])
        cleaned = clean_text(page_content)
        final_grouped_docs[source] = {
            'title': pages[0]['title'],
            'total_pages': pages[0]['total_pages'],
            'page_content': page_content,
            'cleaned_page_content': cleaned
        }

    return final_grouped_docs

def split_by_section(text):
    pattern = r"\n\d{0,2}\.?\s*(abstract|introduction|toolkit overview|toolkit usage|related work|experiments|methodology|results?|conclusion|references)\s*\n"
    matches = list(re.finditer(pattern, text, re.IGNORECASE))

    sections = []
    for i in range(len(matches)):
        title = matches[i].group().strip()
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((title, text[start:end].strip()))
    return sections

def chunk_section(section_text, section_title, doc_title, source, chunk_size=1000, chunk_overlap=40):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(section_text)
    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk,
            metadata={
                'section': section_title,
                'title': doc_title,
                'source': source
            }
        )
        documents.append(doc)
    return documents

def chunk_all_sections(doc_map):
    for source, doc in doc_map.items():
        sections = split_by_section(doc['page_content'])
        all_chunks = []
        for title, content in sections:
            section_title = title.strip().lower().replace("\n", " ").strip()
            section_chunks = chunk_section(content, section_title, doc['title'], source)
            all_chunks.extend(section_chunks)
        doc_map[source]['chunks'] = all_chunks
        doc_map[source]['sections'] = sections
    return doc_map

def push_to_vector_db(document_map):
    for doc in document_map.values():
        if 'chunks' not in doc:
            continue
        PineconeVectorStore.from_documents(
            documents=doc['chunks'],
            index_name=index_name,
            embedding=embeddings
        )
