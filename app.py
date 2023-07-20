import streamlit as st 
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import os

# Setup enviorn variable api key
# os.environ['OPENAI_API_KEY'] = apikey

# Get the api key
load_dotenv()
os.environ.get("OPENAI_API_KEY")

# Customize the layout
st.set_page_config(page_title="DOCAI", page_icon=" ", layout="wide", )     
st.markdown(f"""
            <style>
            .stApp {{background-image: url("https://images.unsplash.com/photo-1509537257950-20f875b03669?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1469&q=80"); 
                     background-attachment: fixed;
                     background-size: cover}}
         </style>
         """, unsafe_allow_html=True)

# function for writing uploaded file in temp
def write_text_file(content, file_path):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        return True
    except Exception as e:
        print(f"Error occurred while writing the file: {e}")
        return False

# set prompt template
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Answer:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# initialize hte LLM & Embeddings

llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()
# llm = LlamaCpp(model_path="./model/llama-7b.ggmlv3.q4_0.bin")
# embeddings = LlamaCppEmbeddings(model_path="model/llama-7b.ggmlv3.q4_0.bin")
llm_chain = LLMChain(llm=llm, prompt=prompt)

st.title("  Document Conversation  ")
uploaded_file = st.file_uploader("Upload an article", type="pdf")

if uploaded_file is not None:
    # content = uploaded_file.read().decode('utf-8')
    # content = uploaded_file.getvalue()
    print(uploaded_file.name)
    file_path = "temp"
    with open(uploaded_file.name, 'wb') as f:
        f.write(uploaded_file.getvalue())
    # st.write(content)
    # file_path = "temp.pdf"
    # write_text_file(content, file_path)   
 
    loader = PyPDFLoader(uploaded_file.name)
    docs = loader.load()    
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    texts = text_splitter.split_documents(docs)
    db = Chroma.from_documents(texts, embeddings)    
    st.success("File Loaded Successfully!!")
 
    # Query through LLM    
    question = st.text_input("Ask something from the file", placeholder="Find something similar to: ....this.... in the text?", disabled=not uploaded_file,)    
    if question:
        similar_doc = db.similarity_search(question, k=5)
        context = similar_doc[0].page_content
        query_llm = LLMChain(llm=llm, prompt=prompt)
        response = query_llm.run({"context": context, "question": question})        
        st.write(response)
        
        with st.expander('Document Similarity Search'):
            # Find the relevant pages
            search = db.similarity_search_with_score(question, k=5) 
            # Write out the first 
            st.write(context) 