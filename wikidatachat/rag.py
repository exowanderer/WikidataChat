# from haystack import Pipeline
from haystack import Document
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.dataclasses import ChatMessage

from .llm_config import llm
from .logger import get_logger
from .prompt import user_prompt_builders, system_prompts
from .vector_store_interface import embedder, retriever, input_documents

# Create logger instance from base logger config in `logger.py`
logger = get_logger(__name__)


def rag_pipeline(query: str, top_k: int = 3, lang: str = 'de'):

    query_document = Document(content=query)
    query_embedded = embedder.run([query_document])
    query_embedding = query_embedded['documents'][0].embedding

    retriever_results = retriever.run(
        query_embedding=list(query_embedding),
        filters=None,
        top_k=top_k,
        scale_score=None,
        return_embedding=None
    )

    logger.debug('retriever results:')
    for retriever_result_ in retriever_results:
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
