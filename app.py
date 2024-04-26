import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from IPython.display import Markdown as md

# Define your RAG components

embedding_model = GoogleGenerativeAIEmbeddings(google_api_key="**************", 
                                               model="models/embedding-001")

db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)
retriever = db_connection.as_retriever(search_kwargs={"k": 5})

chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a Helpful AI Bot. 
    You take the context and question from user. Your answer should be based on the specific context."""),
    HumanMessagePromptTemplate.from_template("""Answer the question based on the given context.
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer: """)
])
chat_model = ChatGoogleGenerativeAI(google_api_key="**************", model="gemini-1.5-pro-latest")
output_parser = StrOutputParser()
# Define Streamlit app


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    st.title("PaperSage")
    st.markdown("<h2 style='text-align: center; color: #008080;'>Welcome to PaperSage</h2>", unsafe_allow_html=True)
    question = st.text_input("Enter your question:")
    if st.button("Ask"):
        response = get_response(question)
        st.markdown(response)

# Function to get response from RAG model
def get_response(question):
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | chat_template
        | chat_model
        | output_parser
    )
    response = rag_chain.invoke(question)
    return response

if __name__ == "__main__":
    main()