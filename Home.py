import streamlit as st


st.set_page_config(
    page_title = "THE FATIMAS",
    page_icon = "f"
)

st.title("FATIMA GPT")


# with st.sidebar:
#     st.title("sidebar")
#     st.text_input("xxx")





# tab_one, tab_two, tab_three = st.tabs(["A","B","C"])

# with tab_one:
#     st.write('a')

# with tab_two:
#     st.write('b')

# with tab_three:
#     st.write('c')




# from langchain.prompts import PromptTemplate
# from datetime import datetime

# today = datetime.today().strftime("%H:%M:%S")

# st.title(today)

# model = st.selectbox(
#     "Choose your model",
#     (
#         "GPT-3",
#         "GPt-4"
#         )
#     )

# st.write(model)

# name = st.text_input("Name")
# st.write(name)

# value = st.slider("temperature", min_value=0.1, max_value=1.0)
# st.write(value)