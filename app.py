import os
import openai
import streamlit as st
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from dotenv import load_dotenv
load_dotenv()

def load_vectorstore(base_path: str, embedding_function):
    print(">>> Loading vectorstore.")
    # Create a list of file paths for the results files you want to load.
    embd_paths = [os.path.join(base_path, f"results{i}.txt") for i in range(500, 2000, 500)]
    
    # Load the first index to initialize the final index.
    final_db = FAISS.load_local(embd_paths[0], embedding_function, allow_dangerous_deserialization=True)
    
    # Loop through the remaining paths and merge each into the final index.
    for path in embd_paths[1:]:
        new_db = FAISS.load_local(path, embedding_function, allow_dangerous_deserialization=True)
        final_db.merge_from(new_db)
    
    # Save the final merged index locally.
    final_db.save_local("final_faiss_index")
    return final_db

def load_npy_files(directory):
    """Load all .npy files from the specified directory into a list of arrays."""
    all_arrays = []
    # List all files in the given directory
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            print("Reading ",filename)
            file_path = os.path.join(directory, filename)
            # Load the array from .npy file
            array = np.load(file_path)
            # Append the numpy array to the list
            all_arrays.append(array.tolist())  # Convert array to list before appending if you need pure Python lists
    return all_arrays

def invoke_chain(chain, question):
    print("Invoking the RAG chain...")
    try:
        response = chain.stream(question)
        print("Chain invocation completed.")
        print("="*30)
        return response
    except Exception as e:
        print(f"Error during chain invocation: {e}")
        return "Error processing your request."

def retrieve_top_documents(query_embedding, index, top_k=3):
    """Retrieve top K similar documents from FAISS index."""
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return indices.flatten().tolist(), distances.flatten().tolist()

def setup_language_model_chain(vectorstore: Chroma):
    print(">>>Setting up LLM chain...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    template = """
                Answer the question comprehensively based on the following context:
                {context}

                Question: {question}


                give well-structured answer.
                """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    print(">>>Chain setup completed.")
    print("="*30)
    return rag_chain

def validate_openai_api_key(api_key: str) -> bool:
    """
    Validates the OpenAI API key by attempting to list the available models.

    Args:
    api_key (str): The OpenAI API key to validate.

    Returns:
    bool: True if the API key is valid, False otherwise.
    """
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.APIConnectionError:
        return False
    else:
        return True
    
with st.sidebar:
    st.title('Document Query Interface')
    api_key = st.sidebar.text_input('Enter API key:', type='password', key='api_key')
    # Check if the API key is valid
    api_key_valid = validate_openai_api_key(api_key)
        
    if api_key_valid:
        st.sidebar.success('API key validated!', icon='✅')
        os.environ['OPENAI_API_KEY'] = api_key
        embd = OpenAIEmbeddings()
        model = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()],temperature=0, model="gpt-3.5-turbo")
    else:
        st.sidebar.error('Invalid API key!', icon='⚠️')

    st.markdown('---')

# Initialize session state variables if they don't exist
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Main chat interface logic
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if api_key_valid:
    # Capture the user's input
    
    if prompt := st.chat_input("Enter your question:", disabled= not(api_key_valid)):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Initialize the RAG chain if it doesn't exist

    if "vectorstore" not in st.session_state:
        embedding_function = SentenceTransformerEmbeddings(model_name= "all-MiniLM-L6-v2", )
        index_path = "C:\\Users\\saura\\OneDrive\\Desktop\\data\\output\\upload"
        
        st.session_state.vectorstore = load_vectorstore(index_path, embedding_function)
    
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = setup_language_model_chain(st.session_state.vectorstore)
    
    # Generate a new response if the last message is not from the assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = invoke_chain(st.session_state.rag_chain, prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
            
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
