import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
import os
from datetime import datetime

#Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


#Extract text from PDFs
def get_pdf_text(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    if not text.strip():
        raise ValueError("No text extracted. Your PDF might be image-based or empty.")
    return text


#Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    if not chunks:
        raise ValueError("No chunks created from text.")
    return chunks


#Build FAISS vectorstore and save
def build_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


#Build ConversationalRetrievalChain
@st.cache_resource
def get_conversational_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    prompt_template = """
    You are a helpful assistant answering questions based on the following context.
    If the answer is clearly stated in the context, include it.
    If it is not explicitly stated, but you can infer or explain it based on your knowledge, please do.
    If you truly cannot find or infer an answer, say "Answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever()

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=False
    )
    return chain


#Streamlit app
def main():
    st.set_page_config(page_title="PDF Gemini Chatbot", layout="wide")
    st.title("PDF Chatbot with Gemini")

    if "chat_history" not in st.session_state:
        # each entry is a dict: {"sender": "user"/"bot", "message": str, "time": timestamp}
        st.session_state.chat_history = []

    with st.sidebar:
        st.header("Upload PDFs")
        pdf_files = st.file_uploader("Choose PDF file(s)", type="pdf", accept_multiple_files=True)
        if st.button("Process PDFs"):
            if not pdf_files:
                st.error("Please upload at least one PDF file.")
            else:
                try:
                    text = get_pdf_text(pdf_files)
                    chunks = get_text_chunks(text)
                    build_vector_store(chunks)
                    st.success("PDFs processed and knowledge base built.")
                    st.session_state.chain = get_conversational_chain()
                except Exception as e:
                    st.error(f"Error: {e}")

    if "chain" in st.session_state:
        st.markdown("---")
        st.subheader("Chat")

        with st.form("chat_form", clear_on_submit=True):
            user_question = st.text_input("Type your question:", placeholder="Ask about the PDFs...")
            submitted = st.form_submit_button("Send")

        if submitted and user_question:
            now = datetime.now().strftime("%H:%M:%S")
            st.session_state.chat_history.append({
                "sender": "user",
                "message": user_question,
                "time": now
            })

            with st.spinner("Thinking..."):
                result = st.session_state.chain.invoke({
                    "question": user_question,
                    "chat_history": [(msg["message"], msg["message"]) for msg in st.session_state.chat_history if msg["sender"] in ["user", "bot"]]
                })
                answer = result["answer"].strip()
                now = datetime.now().strftime("%H:%M:%S")
                st.session_state.chat_history.append({
                    "sender": "bot",
                    "message": answer,
                    "time": now
                })

        if st.session_state.chat_history:
            st.markdown("### Conversation")
            for entry in reversed(st.session_state.chat_history):
                sender = entry["sender"]
                msg = entry["message"]
                ts = entry["time"]

                if sender == "user":
                    st.markdown(f"""
                    <div style='text-align: right; background-color: #DCF8C6; padding: 10px; border-radius: 10px; margin-bottom: 5px;'>
                        <strong> You [{ts}]:</strong><br>{msg}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='text-align: left; background-color: #F1F0F0; padding: 10px; border-radius: 10px; margin-bottom: 5px;'>
                        <strong> Bot [{ts}]:</strong><br>{msg}
                    </div>
                    """, unsafe_allow_html=True)

        if st.button("🗑 Clear Conversation"):
            st.session_state.chat_history = []


if __name__ == "__main__":
    main()
