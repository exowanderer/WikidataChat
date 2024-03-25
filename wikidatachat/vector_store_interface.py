import os
import json

from tqdm import tqdm

from haystack import Document  # , Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.types.policy import DuplicatePolicy
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.preprocessors import DocumentCleaner

import torch

from .logger import get_logger

# Create logger instance from base logger config in `logger.py`
logger = get_logger(__name__)

HUGGING_FACE_HUB_TOKEN = os.environ.get('HUGGING_FACE_HUB_TOKEN')

# disable this line to disable the embedding cache
EMBEDDING_CACHE_FILE = '/root/.cache/wdchat_embeddings.json'

top_k = 5
input_documents = []

device = "cpu"

if torch.cuda.is_available():
    logger.info('GPU is available.')
    device = "cuda"


# TODO: Add the json strings as env variables
json_dir = 'json_input'
json_fname = 'excellent-articles_10.json'

json_fpath = os.path.join(json_dir, json_fname)

if os.path.isfile(json_fpath):
    logger.info(f'Loading data from {json_fpath}')
    with open(json_fpath, 'r') as finn:
        json_obj = json.load(finn)

    if isinstance(json_obj, dict):
        input_documents = [
            Document(
                content=content_,
                meta={"src": url_}
            )
            for url_, content_ in tqdm(json_obj.items())
        ]
    elif isinstance(json_obj, list):
        input_documents = [
            Document(
                content=obj_['content'],
                meta={'src': obj_['meta']}
            )
            for obj_ in tqdm(json_obj)
        ]
else:
    input_documents = [
        Document(
            content="My name is Asra, I live in Paris.",
            meta={"src": "doc_1"}
        ),
        Document(
            content="My name is Lee, I live in Berlin.",
            meta={"src": "doc2"}
        ),
        Document(
            content="My name is Giorgio, I live in Rome.",
            meta={"src": "doc_3"}
        ),
    ]

splitter = DocumentSplitter(
    split_by="sentence",
    split_length=5,
    split_overlap=0
)
input_documents = splitter.run(input_documents)['documents']

cleaner = DocumentCleaner(
    remove_empty_lines=True,
    remove_extra_whitespaces=True,
    remove_repeated_substrings=False
)
input_documents = cleaner.run(input_documents)['documents']


document_store = InMemoryDocumentStore(
    embedding_similarity_function="cosine",
    # embedding_dim=768,
    # duplicate_documents="overwrite"
)

# https://huggingface.co/svalabs/german-gpl-adapted-covid
sentence_transformer_model = 'svalabs/german-gpl-adapted-covid'
logger.info(f'Sentence Transformer Name: {sentence_transformer_model}')

embedder = SentenceTransformersDocumentEmbedder(
    model=sentence_transformer_model,
    device=device
)
embedder.warm_up()


if EMBEDDING_CACHE_FILE and os.path.isfile(EMBEDDING_CACHE_FILE):
    logger.info('Loading embeddings from cache')

    with open(EMBEDDING_CACHE_FILE, 'r') as f_in:
        documents_dict = json.load(f_in)
        document_store.write_documents(
            documents=[Document.from_dict(d_) for d_ in documents_dict],
            policy=DuplicatePolicy.OVERWRITE
        )

else:
    logger.debug("Generating embeddings")

    embedded = embedder.run(input_documents)
    document_store.write_documents(
        documents=embedded['documents'],
        policy=DuplicatePolicy.OVERWRITE
    )

    if EMBEDDING_CACHE_FILE:
        with open(EMBEDDING_CACHE_FILE, 'w') as f_out:
            documents_dict = [
                Document.to_dict(d_)
                for d_ in embedded['documents']
            ]
            json.dump(documents_dict, f_out)

retriever = InMemoryEmbeddingRetriever(document_store=document_store)
