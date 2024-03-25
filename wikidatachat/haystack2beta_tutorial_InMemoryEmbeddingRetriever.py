from haystack import Document
from haystack.pipeline import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

documents = [
    Document(
        content="There are over 7,000 languages spoken around the world today."
    ),
    Document(
        content="Elephants have been observed to behave in a way that indicates a high level of self-awareness, such as recognizing themselves in mirrors."
    ),
    Document(
        content="In certain parts of the world, like the Maldives, Puerto Rico, and San Diego, you can witness the phenomenon of bioluminescent waves."
    )
]

document_embedder = SentenceTransformersDocumentEmbedder(
    model="BAAI/bge-large-en-v1.5"
)
document_embedder.warm_up()
documents_with_embeddings = document_embedder.run(documents)
# print(type(documents_with_embeddings['documents'][0]))

# documents_with_embeddings = document_embedder.run(documents)

documents_with_embeddings = document_embedder.run(documents)
document_store.write_documents(documents_with_embeddings['documents'])
retriever = InMemoryEmbeddingRetriever(document_store=document_store)


# query_pipeline = Pipeline()
# query_pipeline.add_component("text_embedder", document_embedder)
# query_pipeline.add_component("retriever", retriever)
# query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
# query_pipeline.connect("text_embedder.documents", "retriever.query_embedding")

# # Assuming the retriever can handle documents directly
# query_pipeline.connect("text_embedder.embedding_backend",
#                        "retriever.query_embedding")


query = "How many languages are there?"

doc = Document(content=query)


result = document_embedder.run([doc])
results = retriever.run(
    result['documents'][0].embedding
)
for doc_ in results['documents']:
    print(doc_.score, doc_.content)

# result = query_pipeline.run(query)

# results = query_pipeline.run(
#     {
#         "text_embedder": {"query": [Document(content=query)]},
#         "retriever.query_embedding": {"query": query},
#     }
# )

# print(results)
# answer = results["retriever"]["answers"][0]

# print(
#     {
#         "answer": answer.data,
#         "sources": [{
#             "src": d.meta["src"],
#             "content": d.content,
#             "score": d.score
#         } for d in answer.documents]
#     }
# )

# print(result['retriever']['documents'][0])
