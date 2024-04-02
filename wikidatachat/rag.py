import os

# from haystack import Pipeline
from haystack import Document
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.dataclasses import ChatMessage

from multiprocessing import cpu_count

from .llm_config import llm
from .logger import get_logger
from .prompt import user_prompt_builders, system_prompts
from .textification import get_wikidata_statements_from_query
from .vector_store_interface import (
    build_document_store_from_dicts,
    make_embedder,
    setup_document_stream_from_json,
    setup_document_stream_from_list
)

# Create logger instance from base logger config in `logger.py`
logger = get_logger(__name__)

# TODO: create simple object oriented pipeline to minimise global variables
global embedder
embedder = None


def rag_pipeline(
        query: str, top_k: int = 3, lang: str = 'de', device: str = 'cpu',
        content_key: str = None, meta_keys: list = [],
        embedding_similarity_function: str = "cosine",
        sentence_transformer_model: str = 'svalabs/german-gpl-adapted-covid'):

    # TODO: Test if this loads more than once or needs a global variable
    global embedder
    if embedder is None:
        embedder = make_embedder(
            sentence_transformer_model=sentence_transformer_model,
            device=device
        )

    query_document = Document(content=query)
    query_embedded = embedder.run([query_document])
    query_embedding = query_embedded['documents'][0].embedding

    wikidata_statements = get_wikidata_statements_from_query(
        query,
        lang='en',
        timeout=10,
        n_cores=cpu_count(),
        verbose=False,
        api_url='https://www.wikidata.org/w',
        wikidata_base='"wikidata.org"',
        serapi_api_key=os.environ.get("SERAPI_API_KEY"),
        return_list=True
    )

    logger.debug(f'{wikidata_statements=}')
    for wds_ in wikidata_statements:
        logger.debug(f'{wds_=}')

    # logger.debug(f'{len(wikidata_statements)=}')
    # logger.debug(f'{type(wikidata_statements)=}')

    _, retriever = setup_document_stream_from_list(
        dict_list=wikidata_statements,
        content_key=content_key,
        meta_keys=meta_keys,
        embedder=embedder,
        embedding_similarity_function=embedding_similarity_function,
        device=device
    )

    retriever_results = retriever.run(
        query_embedding=list(query_embedding),
        filters=None,
        top_k=top_k,
        scale_score=None,
        return_embedding=None
    )

    logger.debug('retriever results:')
    for retriever_result_ in retriever_results['documents']:
        logger.debug(retriever_result_)

    system_prompt = system_prompts[lang]
    user_prompt_builder = user_prompt_builders[lang]

    user_prompt_build = user_prompt_builder.run(
        question=query_document.content,
        documents=retriever_results['documents']
    )

    prompt = user_prompt_build['prompt']

    logger.debug(f'{prompt=}')

    messages = [
        ChatMessage.from_system(system_prompt),
        ChatMessage.from_user(prompt),
    ]

    response = llm.run(
        messages,
        # generation_kwargs={"temperature": 0.2}
    )

    logger.debug(response)

    answer_builder = AnswerBuilder()
    answer_build = answer_builder.run(
        query=query_document.content,
        replies=response['replies'],
        meta=[r.meta for r in response['replies']],
        documents=retriever_results['documents'],
        pattern=None,
        reference_pattern=None
    )

    logger.debug(f'{answer_build=}')

    return answer_build['answers'][0]
