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

SERAPI_API_KEY = os.environ.get("SERAPI_API_KEY")


class RetreivalAugmentedGenerationPipeline:
    def __init__(
            self, embedding_model='svalabs/german-gpl-adapted-covid',
            device='cpu'):

        self.logger = get_logger(__name__)
        self.embedder = make_embedder(
            embedding_model=embedding_model,
            device=device
        )

    def process_query(
            self, query: str, top_k: int = 3, lang: str = 'de',
            content_key: str = None, meta_keys: list = [],
            embedding_similarity_function: str = "cosine",
            wikidata_kwargs: dict = None):

        if wikidata_kwargs is None:
            wikidata_kwargs = {
                'timeout': 10,
                'n_cores': cpu_count(),
                'verbose': False,
                'api_url': 'https://www.wikidata.org/w',
                'wikidata_base': '"wikidata.org"',
                'return_list': True
            }

        query_document = Document(content=query)
        query_embedded = self.embedder.run([query_document])
        query_embedding = query_embedded['documents'][0].embedding

        wikidata_statements = get_wikidata_statements_from_query(
            query,
            lang=lang,
            serapi_api_key=SERAPI_API_KEY,
            **wikidata_kwargs
        )

        self.logger.debug(f'{wikidata_statements=}')
        for wds_ in wikidata_statements:
            self.logger.debug(f'{wds_=}')

        _, retriever = setup_document_stream_from_list(
            dict_list=wikidata_statements,
            content_key=content_key,
            meta_keys=meta_keys,
            embedder=self.embedder,
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

        self.logger.debug('retriever results:')
        for retriever_result_ in retriever_results['documents']:
            self.logger.debug(retriever_result_)

        system_prompt = system_prompts[lang]
        user_prompt_builder = user_prompt_builders[lang]

        user_prompt_build = user_prompt_builder.run(
            question=query_document.content,
            documents=retriever_results['documents']
        )

        prompt = user_prompt_build['prompt']
        self.logger.debug(f'{prompt=}')

        messages = [
            ChatMessage.from_system(system_prompt),
            ChatMessage.from_user(prompt),
        ]

        response = llm.run(messages)
        self.logger.debug(response)

        answer_builder = AnswerBuilder()
        answer_build = answer_builder.run(
            query=query_document.content,
            replies=response['replies'],
            meta=[r.meta for r in response['replies']],
            documents=retriever_results['documents']
        )

        self.logger.debug(f'{answer_build=}')
        return answer_build['answers'][0]
