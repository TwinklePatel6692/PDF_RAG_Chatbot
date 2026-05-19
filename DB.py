from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()


docs = [
    Document(page_content='TYPES OF MACHINE LEARNING Supervised Learning Model learns using labeled data.Input and correct out' , metadata={'source': 'document loader/ML.txt'} ),
    Document(page_content='tput are provided.Used for prediction and classification.Examples:Email spam detection House pr', metadata={'source': 'document loader/ML.txt'})
    ]

embedding_model = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="Chroma-db"
)

result = vectorstore.similarity_search("what is use of machine learning",k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)

retriver = vectorstore.as_retriever()

docs = retriver.invoke("explain prediction")

for d in docs:
    print(d.page_content)