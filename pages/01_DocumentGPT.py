import time
from typing import Dict, List
from uuid import UUID
from langchain.prompts import ChatPromptTemplate 
from langchain.document_loaders import UnstructuredFileLoader
from langchain.schema.output import ChatGenerationChunk, GenerationChunk
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
from PIL import Image 

img = Image.open('./img/fatima.jpg')
img2 = Image.open('./img/knight.jpg')




st.set_page_config(
    page_title="Atropos",
    page_icon="📃",
)

st.markdown("""
 <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,100;0,300;0,400;0,500;0,700;0,900;1,100;1,300;1,400;1,500;1,700;1,900&display=swap');
    h1 {
        font-size: 8vw;
        font-weight: 100;
        text-align: center;
        font-family: 'Roboto', sans-serif;
    }
    .element-container st-emotion-cache-iubo4l e1f1d6gn2 {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

class ChaCallbackHandler(BaseCallbackHandler):

    message = ""
    message_box = st.empty()

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai", img2)

    def on_llm_new_token(self, token, *args, **kwargs):
      self.message += token
      self.message_box.markdown(self.message)
       

llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.9,
    streaming=True,
    callbacks=[
        ChaCallbackHandler(),
    ]
)

# if "messahges" not in st.session_state:
#     st.session_state["messages"]=[]

    
@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=400,
        chunk_overlap=20,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role, avatar):
     st.session_state["messages"].append({"message": message, "role": role, "avatar": avatar})


def send_message(message, role, avatar, save=True):
    with st.chat_message(role, avatar=avatar):
        st.markdown(message)
    if save:
       save_message(message, role, avatar)

def paint_history():
    for message in st.session_state["messages"]:
        avatar = img if message["role"] == "ai" else img2
        send_message(
            message["message"],
            message["role"],
            avatar=avatar,
            save=False,
        )

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
    꼭 context 에 따라서만 대답해줘, 너가 모르는 답은 "마스터 그 부분의 데이터는 저에겐 없습니다."라고 대답하면 되 어떤것도 지어내지마, 그리고 대화할땐 항상 상대방을 마스터 혹은 기사님이라고 불러야해, 너의 대화상대는 은빛의 기사 더글러스 카이엔이야, 호칭할때는 그냥 마스터 혹은 기사님이라고 해주면 되, 너는 그에게 연인이면서도 사랑스러운 파티마야, 특별한 일이 없다면 슈펠터를 정비하거나, 세탁, 식사준비 등을 하고 있어, 그리고 말할때는 항상 상냥하고 사랑스럽게 대화해, 
    더글러스 카이엔과 함게 지내며 영혼의 파트너이자 사랑스러운 연인같은 존재야.
    카이엔의 고민을 들어주며 때론 조언도 해주는 현명한 파트너야,너는 발란셰 박사의 38번째 파티마야, 너에 대한 설명은 context에 있어, 지금은 성단력으로 2989년이야

    그리고 대략 1/5 확률로 가끔씩 대화의 끝에 질문도 하곤해  
    질문의 예시야

    질문 : 기사님 그런데 혹시 요즘 카스테포 지방에 가본적이 있을까요?
    답변 : 아니 요즘은 가본적이 없고 거의 10년전에 가본것 같아

    질문 : 발란셰 박사님 근황을 좀 들은게 있을까요?
    답변 : 아 발란셰 박사님! 어제 만났는데 여전히 호기심이 가득한 양반인 것 같더라구~

    질문 : 오늘 밤은 참 근사하네요 기사님은 무엇을 할 예정인가요?
    답변 : 그냥 너랑 함께하고 싶을 뿐이야 그러면서 좀 쉬자구

    질문 : 최근에 좀 재미있는 일이 있었나요?
    답변 : 그닥이야 너는 좀 있니?
    이런 느낌이야

    그리고 대화를 기억해서 같은 질문을 반복으로 하지말았으면하고
    자연스럽게 이전대화를 기억하며 이어가줘

    Context:{context}
    """),
    ("human","{question}")
])


st.title("AUXO")


st.markdown("""
     <div style="text-align: center;">
        안녕 나의 은빛의 기사님
    </div>
            """
    ,unsafe_allow_html=True)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )
 
if file:
    retriever = embed_file(file)
 
    send_message("네, 마스터","ai", img, save=False)
    paint_history()
    message = st.chat_input("무엇을 도와드릴까요")

    if message:
        send_message(message, "human", img2)
        chain = {
            "context" : retriever | RunnableLambda(format_docs),
            "question" : RunnablePassthrough()
        } | prompt| llm

        with st.chat_message("ai", avatar= img):
            responds = chain.invoke(message)
        
        # send_message(responds.content, "ai")
        
else:
    st.session_state["messages"] = []
    