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


# disable this line to disable the embedding cache
# EMBEDDING_CACHE_FILE = '/root/.cache/wdchat_embeddings.json'
EMBEDDING_CACHE_FILE = None
EMBEDDING_MODEL = os.environ.get(
    'EMBEDDING_MODEL',
    'svalabs/german-gpl-adapted-covid'
)


def build_document_store_from_json(
        json_dir: str = 'json_input',
        json_fname: str = 'excellent-articles_10.json',
        **kwargs):

    input_documents = []

    # TODO: Add the json strings as env variables
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
        return get_dummy_document_list()

    return input_documents


def build_document_store_from_dicts(
        dict_list: list,
        content_key: str,
        meta_keys: list = [],
        **kwargs):

    return [
        Document(
            content=obj_[content_key],
            meta={m_key: obj_[m_key] for m_key in meta_keys}
        )
        for obj_ in tqdm(dict_list)
    ]


def get_dummy_document_list():
    return [
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


def split_documents(input_documents: list):
    splitter = DocumentSplitter(
        split_by="sentence",
        split_length=5,
        split_overlap=0
    )
    return splitter.run(input_documents)['documents']


def clean_documents(input_documents: list):
    cleaner = DocumentCleaner(
        remove_empty_lines=True,
        remove_extra_whitespaces=True,
        remove_repeated_substrings=False
    )

    return cleaner.run(input_documents)['documents']


def make_embedder(embedding_model: str = EMBEDDING_MODEL, device: str = 'cpu'):

    if torch.cuda.is_available():
        logger.info('GPU is available.')
        device = "cuda"

    # https://huggingface.co/svalabs/german-gpl-adapted-covid
    logger.info(f'Sentence Transformer Name: {embedding_model}')

    embedder = SentenceTransformersDocumentEmbedder(
        model=embedding_model,
        device=device
    )
    embedder.warm_up()

    return embedder


def build_documentstore_embedder_retriever(
        input_documents: list,
        embedder: SentenceTransformersDocumentEmbedder = None,
        embedding_similarity_function: str = "cosine",
        **kwargs):

    document_store = InMemoryDocumentStore(
        embedding_similarity_function=embedding_similarity_function,
        # embedding_dim=768,
        # duplicate_documents="overwrite"
    )
    """
    if EMBEDDING_CACHE_FILE and os.path.isfile(EMBEDDING_CACHE_FILE):
        logger.info('Loading embeddings from cache')

        with open(EMBEDDING_CACHE_FILE, 'r') as f_in:
            documents_dict = json.load(f_in)
            document_store.write_documents(
                documents=[Document.from_dict(d_) for d_ in documents_dict],
                policy=DuplicatePolicy.OVERWRITE
            )
    elif embedder is not None:
    """
    if embedder is not None:
        logger.debug("Generating embeddings")

        embedded = embedder.run(input_documents)
        document_store.write_documents(
            documents=embedded['documents'],
            policy=DuplicatePolicy.OVERWRITE
        )
        """
        if EMBEDDING_CACHE_FILE:
            logger.debug(
                f'Grabbing Embedding from file: {EMBEDDING_CACHE_FILE}'
            )
            with open(EMBEDDING_CACHE_FILE, 'w') as f_out:
                documents_dict = [
                    Document.to_dict(d_)
                    for d_ in embedded['documents']
                ]
                json.dump(documents_dict, f_out)
        """

    return document_store


def make_retriever(document_store):
    return InMemoryEmbeddingRetriever(document_store=document_store)


def setup_document_stream(common_setup_fn, *args, **kwargs):
    """
    A generic function to set up the document stream by performing common setup tasks
    and then delegating to a specific setup function provided as an argument.

    Args:
        common_setup_fn (callable): A function specific to the type of document source (JSON or list).
        *args: Positional arguments to be passed to the common_setup_fn.
        **kwargs: Keyword arguments to be passed to the common_setup_fn.

    Returns:
        Tuple[InMemoryDocumentStore, InMemoryEmbeddingRetriever]: A tuple containing the document store and retriever.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f'Using device: {device}')

    embedder = kwargs.get('embedder')
    if embedder is None:
        if 'embedding_model' not in kwargs:
            kwargs['embedding_model'] = EMBEDDING_MODEL

        embedder = make_embedder(**kwargs)

    input_documents = common_setup_fn(*args, **kwargs)

    if 'split_documents' in kwargs and kwargs['split_documents']:
        input_documents = split_documents(input_documents)

    if 'clean_documents' in kwargs and kwargs['clean_documents']:
        input_documents = clean_documents(input_documents)

    document_store = build_documentstore_embedder_retriever(
        input_documents=input_documents,
        embedder=embedder,
        device=device
    )

    retriever = make_retriever(document_store)

    return document_store, retriever


def setup_document_stream_from_json(**kwargs):
    """
    Wrapper function for setting up document stream from JSON source.

    KWArgs:
        json_dir (str): Director location with JSON for embedding vectors.
            Defaults to 'json_input'.
        json_fname (str): JSON filename with statements to be embedded.
            Defaults to 'excellent-articles_10.json'.
        embedder (SentenceTransformersDocumentEmbedder): Haystack embedder 
            function to transform sentences into embedding vectors.
            Defaults to None.
        embedding_similarity_function (str): Similarity function to implement 
            when vector searching embedding databse. Defaults to "cosine".
        device (str): PyTorch implementation requirement to establish on which 
            chip to process the embedding model. Defaults to "cpu".
        split_documents (bool): Toggle whether to split documents with Haystack 
            DocumentSplitter. Defaults to False.
        clean_documents (bool): Toggle whether to clean documents with Haystack 
            DocumentCleaner. Defaults to False.

    Returns:
        Function call to setup_document_stream with specific setup function and 
            kwargs. setup_document_stream return document_store and retriever.
    """
    return setup_document_stream(build_document_store_from_json, **kwargs)


def setup_document_stream_from_list(**kwargs):
    """
    Wrapper function for setting up document stream from a list source.

    KWArgs:
        dict_list (list): List of dictionary items to store in DocumentStore
        content_key (str): Key for dictionary as content in DocumentStore
        meta_keys (list): List of keys for meta data in DocumentStore. 
            Defaults to [].
        embedder (SentenceTransformersDocumentEmbedder): Haystack embedder 
            function to transform sentences into embedding vectors.
            Defaults to None.
        embedding_similarity_function (str): Similarity function to implement 
            when vector searching embedding databse. Defaults to "cosine".
        device (str): PyTorch implementation requirement to establish on which 
            chip to process the embedding model. Defaults to "cpu".
    Returns:
        Function call to setup_document_stream with specific setup function and 
            kwargs. setup_document_stream return document_store and retriever.
    """
    return setup_document_stream(build_document_store_from_dicts, **kwargs)
