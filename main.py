import ollama
import streamlit as st
import chromadb


class AI:
    def __init__(self):
        db = chromadb.PersistentClient()
        self.collection = db.get_collection("food-data")

    def query(self, q, top=10):
        res_db = self.collection.query(query_texts=[q])["documents"][0][:top]

        context = ''.join(res_db).replace("\n", " ")
        return context

    def respond(self, lst_messages, model="qwen2:1.5b", use_knowledge=False):

        q = lst_messages[-1]["content"]

        context = self.query(q)

        if use_knowledge:
            prompt = "Give the most accurate answer using your knowledge and the folling additional information: \n"+context
        else:
            prompt = "Give the most accurate answer using only the folling information: \n"+context

        res_ai = ollama.chat(model=model,
                             messages=[
                                 {"role": "system", "content": prompt}]+lst_messages,
                             stream=True)
        for res in res_ai:
            chunk = res["message"]["content"]
            app["full_response"] += chunk
            yield chunk


ai = AI()


app = st.session_state

st.set_page_config(page_title="🍲 Food RAG-LLM", page_icon="🍲")

# Main Title
st.title('🍲 Ask Your Recipe Questions Here!')

# Sidebar Title
st.sidebar.title("📝 Chat History")

# Initialize session state
app = st.session_state

if "messages" not in app:
    app["messages"] = [{"role": "assistant",
                        "content": "👩‍🍳 I'm ready to help you with recipes!"}]

if 'history' not in app:
    app['history'] = []

if 'full_response' not in app:
    app['full_response'] = ''

# Keep messages in the chat
for msg in app["messages"]:
    if msg["role"] == "user":
        st.chat_message(msg["role"], avatar="🧑‍🍳").write(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message(msg["role"], avatar="👩‍🍳").write(msg["content"])

if txt := st.chat_input():
    # User writes
    app["messages"].append({"role": "user", "content": txt})

    st.chat_message("user", avatar="🧑‍🍳").write(txt)

    # AI responds with chat stream
    app["full_response"] = ""
    st.chat_message("assistant", avatar="👩‍🍳").write_stream(
        ai.respond(app["messages"]))
    app["messages"].append(
        {"role": "assistant", "content": app["full_response"]})

    # Show sidebar history
    app['history'].append("🧑‍🍳: "+txt)
    app['history'].append("👩‍🍳: "+app["full_response"])
    st.sidebar.markdown(
        "<br />".join(app['history'])+"<br /><br />", unsafe_allow_html=True)

# Add some color
st.markdown(
    """
    <style>
    .stSidebar {background-color: #fff8e1;}
    .stTitle {color: #ff7043;}
    .stMarkdown {color: #ff7043;}
    </style>
    """,
    unsafe_allow_html=True
)
