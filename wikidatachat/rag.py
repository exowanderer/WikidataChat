import os

# Import Document class for creating document objects.
from haystack import Document

# Import AnswerBuilder for constructing answers.
from haystack.components.builders.answer_builder import AnswerBuilder

# Import ChatMessage for creating chat messages.
from haystack.dataclasses import ChatMessage

# Import cpu_count for determining the number of cores available.
from multiprocessing import cpu_count

# Import custom modules for various functionalities.
from .llm_config import llm
from .logger import get_logger
from .prompt import user_prompt_builders, system_prompts
from .textification import get_wikidata_statements_from_query
from .vector_store_interface import (
    # build_document_store_from_dicts,
    make_embedder,
    # setup_document_stream_from_json,
    setup_document_stream_from_list
)

# Retrieve the SERAPI API key from environment variables.
SERAPI_API_KEY = os.environ.get("SERAPI_API_KEY")


class RetreivalAugmentedGenerationPipeline:
    def __init__(
            self, embedding_model='svalabs/german-gpl-adapted-covid',
            device='cpu'):
        """
        Initializes the retrieval-augmented generation pipeline with the specified embedding model and device.

        Args:
            embedding_model (str): The name of the embedding model to use.
            device (str): The device to run the embedding model on, e.g., 'cpu' or 'cuda'.
        """
        # Initialize a logger for this class.
        self.logger = get_logger(__name__)
        self.device = device
        self.embedding_model = embedding_model
        # Initialize the embedder with the specified model and device.
        self.embedder = make_embedder(
            embedding_model=self.embedding_model,
            device=self.device
        )

    def process_query(
            self, query: str, top_k: int = 3, lang: str = 'de',
            content_key: str = None, meta_keys: list = [],
            embedding_similarity_function: str = "cosine",
            wikidata_kwargs: dict = None):
        """
        Processes the given query to generate an answer using the retrieval-augmented generation pipeline.

        Args:
            query (str): The user's query to process.
            top_k (int): The number of top results to consider.
            lang (str): The language of the query.
            content_key (str): The key to extract content from the documents.
            meta_keys (list): A list of keys to extract metadata from the documents.
            embedding_similarity_function (str): The similarity function to use for embedding comparison.
            wikidata_kwargs (dict, optional): Additional keyword arguments for querying Wikidata.

        Returns:
            The first answer from the generated answers.
        """
        if wikidata_kwargs is None:
            # Default Wikidata query parameters.
            wikidata_kwargs = {
                'timeout': 10,
                'n_cores': cpu_count(),
                'verbose': False,
                'api_url': 'https://www.wikidata.org/w',
                'wikidata_base': '"wikidata.org"',
                'return_list': True
            }

        # Create a Document object from the query.
        query_document = Document(content=query)

        # Embed the query document.
        query_embedded = self.embedder.run([query_document])

        # Extract the embedding of the query document.
        query_embedding = query_embedded['documents'][0].embedding

        # Retrieve Wikidata statements related to the query.
        wikidata_statements = get_wikidata_statements_from_query(
            query,
            lang=lang,
            serapi_api_key=SERAPI_API_KEY,
            **wikidata_kwargs
        )

        # Log the retrieved Wikidata statements for debugging.
        self.logger.debug(f'{wikidata_statements=}')
        for wds_ in wikidata_statements:
            # Log each Wikidata statement for debugging.
            self.logger.debug(f'{wds_=}')

        # Setup the document stream from the list of Wikidata statements.
        _, retriever = setup_document_stream_from_list(
            dict_list=wikidata_statements,
            content_key=content_key,
            meta_keys=meta_keys,
            embedder=self.embedder,
            embedding_similarity_function=embedding_similarity_function,
            device=self.device
        )

        # Run the retriever to find relevant documents
        #   based on the query embedding.
        retriever_results = retriever.run(
            query_embedding=list(query_embedding),
            filters=None,
            top_k=top_k,
            scale_score=None,
            return_embedding=None
        )

        # Log the start of retriever results for debugging.
        self.logger.debug('retriever results:')
        for retriever_result_ in retriever_results['documents']:
            # Log each retriever result for debugging.
            self.logger.debug(retriever_result_)

        # Get the system prompt for the specified language.
        system_prompt = system_prompts[lang]

        # Get the user prompt builder for the specified language.
        user_prompt_builder = user_prompt_builders[lang]

        # Build the user prompt based on the retrieved documents
        #   and the original query.
        user_prompt_build = user_prompt_builder.run(
            question=query_document.content,
            documents=retriever_results['documents']
        )

        # Extract the constructed prompt.
        prompt = user_prompt_build['prompt']

        # Log the constructed prompt for debugging.
        self.logger.debug(f'{prompt=}')

        # Create chat messages from the system and user prompts.
        messages = [
            ChatMessage.from_system(system_prompt),
            ChatMessage.from_user(prompt),
        ]

        # Run the language model with the constructed chat messages.
        response = llm.run(messages)

        # Log the language model response for debugging.
        self.logger.debug(response)

        answer_builder = AnswerBuilder()  # Initialize the AnswerBuilder.

        # Build the answer based on the language model's response
        #   and the retrieved documents.
        answer_build = answer_builder.run(
            query=query_document.content,
            replies=response['replies'],
            meta=[r.meta for r in response['replies']],
            documents=retriever_results['documents']
        )

        # Log the constructed answer for debugging.
        self.logger.debug(f'{answer_build=}')

        # Return the first answer from the constructed answers.
        return answer_build['answers'][0]
