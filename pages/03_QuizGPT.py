import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler


st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

llm = ChatOpenAI(
    temperature=0.5,
    model = "gpt-4-turbo",
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler()
    ]
)

@st.cache_data(show_spinner="Loading file...") 
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=400,
        chunk_overlap=20,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

with st.sidebar:
    docs = None
    choice=st.selectbox(
        "choose",
        (
            "File",
            "Wikipedia"
        )
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload file",
            type=["pdf","txt","docx"],
            )
        if file:
            docs = split_file(file)
            st.write(docs)  
    else:
        topic = st.text_input("Search wikipedia...")
        if topic:
            retriever = WikipediaRetriever(top_k_results=1)
            with st.status("Searching Wikipedia..."):
                docs = retriever.get_relevant_documents(topic)
                st.write(docs)

if not docs:
    st.markdown("""
    퀴즈에 온것을 환영해요
    """)
else:


    prompt = ChatPromptTemplate.from_messages([
        ("system","""
        You are a helpful assistant that is role playing as a teacher,
        아래 내용에 따라 유저의 텍스트에 대한 지식을 판단하는 문제 10개를 만드세요
        각각의 질문들은 모두 4개의 선택지가 있어야 하고 그 중 3개는 오답 1개는 정답이어야 합니다.
        그리고 우리는 AI가 (O)문자를 통해 정답을 표시하도록 합니다.

        질문에대한 예시들이야

        질문: 바다의 색은 무슨 색이니?
        답변: 빨강, 노랑, 녹색, 파랑(O)   
         
        질문: 일본의 수도는 어디이니?
        답변: 파리, 도쿄(O), 오사카, 베이징
         
        질문: 쥴리어스 시저는 누구이니?
        답변: 로마의 황제(O), 코미디언, 경찰, 요리사  
         
        질문: 영화 아바타1편은 언재 개봉했니?
        답변: 1632, 1989, 2023, 2009(O)  
         
        이제 니 차례야!
         
        Context:{context}
        """)
        ]
    )
    chain = {
        "context" : format_docs
    } | prompt | llm

    start = st.button("퀴즈생성")

    if start:
        chain.invoke(docs)

    # st.write(docs)
    