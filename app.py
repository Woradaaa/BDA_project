import streamlit as st
import torch
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- 1. แสดงรายละเอียดกลุ่มและสมาชิกตามโจทย์ BDA Project Part 2 ---
st.set_page_config(page_title="MFU Academy AI Assistant", page_icon="🎓")

# แสดงชื่อกลุ่มในรูปแบบที่อาจารย์กำหนด [cite: 391]
st.title("🎓 MFU Academy AI Assistant")
st.markdown("### Group No: BDA_Project2_10") 

# แสดงรายชื่อสมาชิกกลุ่มในส่วน Description [cite: 392]
st.markdown("""
**Description:**
ระบบ AI ผู้ช่วยสำหรับตอบคำถามคอร์สเรียนออนไลน์ของ MFU Academy พัฒนาด้วยเทคนิค RAG Pipeline 
และใช้งานผ่าน Streamlit ตามข้อกำหนดของรายวิชา Business Data Analytics

**รายชื่อสมาชิกกลุ่ม:**
- 6631501148 Kanphong Nasuriwong 
- 6631501158 Nitiwat Chatturong
- 6631501168 Worada Suyawa
- 6631501169 Supison Kingjuntrasin
""")
st.divider()

# --- 2. การตั้งค่าระบบ RAG Pipeline ---
@st.cache_resource
def initialize_rag():
    # โหลดข้อมูลจากรายละเอียดคอร์ส [cite: 1-350] และคำถาม Q&A [cite: 353-376]
    loader_txt = TextLoader("รายละเอียดคอร์สออนไลน์.txt", encoding="utf-8")
    loader_pdf = PyPDFLoader("MFU Academy Q&A.pdf")
    
    docs = loader_txt.load() + loader_pdf.load()

    # แบ่งข้อความเป็นส่วนย่อย (Chunking) เพื่อการค้นหาที่แม่นยำ
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # สร้าง Vector Store ด้วย FAISS และ Embedding ของ HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # ตั้งค่าโมเดลภาษา OpenThaiGPT พร้อมการทำ 4-bit Quantization
    model_id = "openthaigpt/openthaigpt-1.0.0-7b-chat"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=bnb_config)

    text_pipeline = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, 
        max_new_tokens=256, temperature=0.3, do_sample=True, repetition_penalty=1.1
    )
    llm = HuggingFacePipeline(pipeline=text_pipeline)

    # สร้าง System Prompt เพื่อควบคุมพฤติกรรมของ AI
    system_prompt = (
        "You are an expert advisor for MFU Academy. "
        "Answer the user's questions clearly and politely based ONLY on the provided context. "
        "If you don't know the answer, say that you don't have that information. "
        "Always answer in Thai."
        "\n\nContext:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

# โหลดระบบ RAG (จะทำเพียงครั้งเดียวด้วย st.cache_resource)
with st.spinner("กำลังเริ่มระบบ AI สำหรับ MFU Academy..."):
    rag_chain = initialize_rag()

# --- 3. ส่วนการทำงานของ Chat UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("สอบถามเกี่ยวกับคอร์สเรียนหรือการรับ Certificate..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาข้อมูล..."):
            response = rag_chain.invoke({"input": user_input})
            answer = response['answer']
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
