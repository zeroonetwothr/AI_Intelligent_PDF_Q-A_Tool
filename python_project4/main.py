import streamlit as st
from utils import qa_agent
st.title("📑 AI 智能 PDF 问答工具")
with st.sidebar:
    api_key = st.text_input("请输入火山引擎 API 密钥：", type="password")
    st.markdown(
        "[获取火山引擎 API Key]"
        "(https://console.volcengine.com/ark/region:ark+cn-beijing/model?groupType=ModelGroups&vendor=Bytedance&view=DEFAULT_VIEW)"
    )
if "messages" not in st.session_state:
    st.session_state["messages"] = []
uploaded_file = st.file_uploader("上传你的 PDF 文件：", type="pdf")
question = st.text_input(
    "对 PDF 的内容进行提问",
    disabled=not uploaded_file
)
if uploaded_file and question and not api_key:
    st.info("请输入你的 API 密钥")
if uploaded_file and question and api_key:
    st.session_state["messages"].append(
        {"role": "human", "content": question}
    )
    with st.spinner("AI 正在思考中，请稍等..."):
        answer = qa_agent(uploaded_file, question)
    st.session_state["messages"].append(
        {"role": "ai", "content": answer}
    )
    st.write("### 答案")
    st.write(answer)
if st.session_state["messages"]:
    with st.expander("历史问答记录"):
        for msg in st.session_state["messages"]:
            if msg["role"] == "human":
                st.markdown(f"**你：** {msg['content']}")
            else:
                st.markdown(f"**AI：** {msg['content']}")
            st.divider()
