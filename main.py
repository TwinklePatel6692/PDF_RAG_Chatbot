from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

load_dotenv()

embedding_model = OpenAIEmbeddings()

vectorstore = Chroma(
    persist_directory= "Chroma-db",
    embedding_function= embedding_model
)

retriever = vectorstore.as_retriever(       
    search_type = "mmr",
    search_kwargs={
        "k": 4,                   # k: Amount of documents to return 
        "fetch_k":10,              # Amount of documents to pass to MMR algorithm 
        "lambda_mult":0.5           # Diversity of results returned by MMR; 1 for minimum diversity and 0 for maximum.
    }
)

llm = ChatMistralAI(model="mistral-small-2506")

# Prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
            """you are a helpful AI assistant.
            Use ONLY the provided context to answer the Question.
            If the answer is not present in the context,
            say: "I could not find the answer in the document."
                
                 """
         ),
         (
             "human",
             """context:
             {context}
             
             Question:
             {question}"""
         )
    ]
)

print("RAG sysytem created")

print("press 0 to exit.")

while True:
    query = input("you : ")
    if query == "0":
       break 

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    }) 

    response = llm.invoke(final_prompt)

    print(f"\n AI: {response.content}")