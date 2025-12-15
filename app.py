import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()


os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")




def process_uploaded_files(uploaded_files,embeddings):

    documents=[]
    for uploaded_file in uploaded_files:
        temp_pdf=f"./temp.pdf"
        with open(temp_pdf,"wb") as file:
            file.write(uploaded_file.getvalue())
            #filename=uploaded_file.name

        loader=PyPDFLoader(temp_pdf)
        docs=loader.load()
        documents.extend(docs)

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=200)
    splits=text_splitter.split_documents(documents)
    vector_store=FAISS.from_documents(documents=splits,embedding=embeddings)
    # retriever=vector_store.as_retriever()

    return vector_store
    
def build_rag_chian(llm, retriever):

    contextualize_q_systemprompt=(
        " Use the chat history to answer the latest user question" \
        "which might related to context in the chat history," \
        "design a standalone question which can be understood without the " \
        "chat history. Do NOT answer the question, " \
        "just redesign it if needed and otherwise return is as it is."
        )

    contextualize_q_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_systemprompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]

        )
    
    history_aware_retriver=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

    system_prompt=(
        "You are an helpful assistant for Q and A tasks" \
        "use the following pieces of retrieved context to answer " \
        "the question. if you don't know the answer, say that you" \
        "don't know. answer the question in a clear and concise manner." \
        "\n\n" \
        "{context}"
        )
    
    qa_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )

    question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
    rag_chain=create_retrieval_chain(history_aware_retriver,question_answer_chain)
    return rag_chain




def get_session_history(session:str)->BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session]=ChatMessageHistory()
    return st.session_state.store[session]

def validate_api_key(api_key: str):
    
    try:
        llm = ChatGroq(api_key=api_key, model="openai/gpt-oss-20b")
        response = llm.invoke("Hello")
        return True
    except Exception as e:
        print("API key validation failed:", e)
        return False

st.title("AI-Powered PDF Q&A Assistant (RAG + Groq + FAISS + Streamlit)")

api_key = st.text_input("Enter your Groq API key", type="password")

uploaded_files = st.file_uploader("Choose PDF Files", type="pdf", accept_multiple_files=True)

session_id = st.text_input("Session ID", value="default_session")

user_input = st.text_input("Ask a question:")

if "store" not in st.session_state:
    st.session_state.store = {}

if "api_key" not in st.session_state:
    st.session_state.api_key = None

if "api_validated" not in st.session_state:
    st.session_state.api_validated = False


if api_key and not st.session_state.api_validated:
    with st.spinner("Validating API key..."):
        if validate_api_key(api_key):
            st.session_state.api_key = api_key
            st.session_state.api_validated = True
            st.success("API key validated successfully")
        else:
            st.error("Invalid API key")
            st.stop()

if not st.session_state.api_validated:
    st.warning("Please enter a valid Groq API key to continue")
    st.stop()



if uploaded_files:

    st.sidebar.subheader("Uploaded Files:")
    for f in uploaded_files:
        st.sidebar.write(f"- {f.name}")
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = process_uploaded_files(uploaded_files, embeddings)

    retriever = st.session_state.vector_store.as_retriever()

    llm = ChatGroq(api_key=st.session_state.api_key, model="openai/gpt-oss-20b")

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = build_rag_chian(llm, retriever)

    conversational_rag_chain = RunnableWithMessageHistory(
        st.session_state.rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )

        st.write("Assistant:", response["answer"])
        with st.expander(" Chat History"):
            for msg in session_history.messages:
                st.markdown(f"{msg.type.title()}:{msg.content}")


else:
    st.warning(" Please upload files to continue.")







