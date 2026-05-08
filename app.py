import streamlit as st
import torch
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# 1. Setting the web application
st.set_page_config(page_title="MFU Academy AI", page_icon="🎓")
st.title("🎓 MFU Academy AI Assistant")
st.write("ระบบผู้ช่วยตอบคำถามคอร์สเรียนออนไลน์ด้วยเทคนิค RAG Pipeline")

# แสดง Group No. และรายชื่อสมาชิกตามข้อกำหนด BDA
st.markdown("### Group No: BDA_Project2_10")
st.sidebar.header("รายชื่อสมาชิกกลุ่ม")
st.sidebar.markdown("""
- 6631501148 Kanphong Nasuriwong
- 6631501158 Nitiwat Chatturong
- 6631501168 Worada Suyawa
- 6631501169 Supison Kingjuntrasin
""")

# 2. Setting cache for the model and RAG pipeline
@st.cache_resource
def load_rag_model():
    # ตรวจสอบและโหลดไฟล์ข้อมูล
    if not os.path.exists("รายละเอียดคอร์สออนไลน์.txt"):
        st.error("⚠️ ไม่พบไฟล์ 'รายละเอียดคอร์สออนไลน์.txt' กรุณาอัปโหลดขึ้น GitHub ด้วยครับ")
        return None

    loader = TextLoader("รายละเอียดคอร์สออนไลน์.txt", encoding="utf-8")
    docs = loader.load()

    # หั่นข้อมูลและสร้าง Vector Store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # โหลดโมเดล OpenThaiGPT
    model_id = "openthaigpt/openthaigpt-1.0.0-7b-chat"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=bnb_config)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, temperature=0.3)
    llm = HuggingFacePipeline(pipeline=pipe)

    # กำหนด Prompt
    system_prompt = (
        "คุณคือผู้ช่วยอัจฉริยะของ MFU Academy ตอบคำถามอย่างสุภาพโดยใช้ข้อมูลที่ให้มาเท่านั้น "
        "หากไม่มีข้อมูลให้ตอบว่า 'ขออภัยครับ ไม่พบข้อมูลในระบบ'\n\nContext:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)

# Load the model
rag_chain = load_rag_model()

# 3. Chat Interface Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. User Input & Prediction
if user_query := st.chat_input("สอบถามรายละเอียดคอร์สเรียน..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 5. Show the result
    if rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("🔄 กำลังค้นหาข้อมูล..."):
                response = rag_chain.invoke({"input": user_query})
                answer = response['answer']
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
