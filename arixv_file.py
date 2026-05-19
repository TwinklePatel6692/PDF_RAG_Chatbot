from langchain_community.retrievers import ArxivRetriever

# create the retriever

retriever = ArxivRetriever(
    load_max_docs=2,           # number of paper to retrieve
    load_all_available_meta=False
)

# Query Arixv

docs = retriever.invoke("large language model")

# print results

for i, doc in  enumerate(docs):
    print(f"\nResult {i+1}")
    print("Title:",doc.metadata.get("Title"))
    print("Authors:", doc.metadata.get("Authors"))
    print("summary:", doc.page_content[:500]) # print first 500

