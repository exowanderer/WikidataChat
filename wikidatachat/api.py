import os  # Import the os module to interact with the operating system.

from typing import Annotated

# Import necessary types and classes from FastAPI and other libraries.
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Header

# Import custom modules for the RAG pipeline and logging.
from .rag import RetreivalAugmentedGenerationPipeline
from .logger import get_logger

# Create logger instance from base logger config in `logger.py`
logger = get_logger(__name__)  # Initialize a logger for this module.

# Retrieve the frontend static directory path from environment variables, falling back to a default if not set.
FRONTEND_STATIC_DIR = os.environ.get('FRONTEND_STATIC_DIR', './frontend/dist')
API_SECRET = os.environ.get('API_SECRET', 'Thou shall [not] pass')
EMBEDDING_MODEL = os.environ.get(
    'EMBEDDING_MODEL',
    'svalabs/german-gpl-adapted-covid'
)

app = FastAPI()  # Create an instance of the FastAPI application.

# Serve static files from the '/assets' endpoint,
#   pulling from the frontend static directory.
app.mount(
    "/assets",
    StaticFiles(directory=f"{FRONTEND_STATIC_DIR}/assets"),
    name="frontend-assets"
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipeline = RetreivalAugmentedGenerationPipeline(
    embedding_model=EMBEDDING_MODEL,
    device=device
)


@app.get("/")
async def root():
    """
    Serve the main HTML file for the root endpoint.

    Returns:
        FileResponse: The index.html file from the frontend static directory.
    """
    return FileResponse(f"{FRONTEND_STATIC_DIR}/index.html")


@app.get("/favicon.ico")
async def favicon():
    """
    Serve the favicon.ico file.

    Returns:
        FileResponse: The favicon.ico file from the frontend static directory.
    """
    return FileResponse(f"{FRONTEND_STATIC_DIR}/favicon.ico")


# async def api(query, top_k=10, lang='en'):
@app.get("/api")
async def api(
        x_api_secret: Annotated[str, Header()], query, top_k=3, lang='en'):
    """
    Handle the API requests to process and respond with relevant information based on the query.

    Args:
        x_api_secret (str): API Secret to confirm user is authorised.
        query (str): The query string to be processed.
        top_k (int, optional): The number of top results to return. 
            Defaults to 10.
        lang (str, optional): The language code for the query processing 
            ('en' for English, 'de' for German). Defaults to 'en'.

    Raises:
        ValueError: If the provided language is not supported 
            (not 'en' or 'de').

    Returns:
        dict: A dictionary containing the 'answer' and 'sources' 
            based on the query processing.
    """
    if not API_SECRET in [x_api_secret, 'Thou shall [not] pass']:
        raise ValueError("API key is missing or incorrect")

    if not lang in ['en', 'de']:
        # return {
        #     "answer": 'At the moment, only English (en) and Deutsch (de) are available',
        #     "sources": []
        # }
        raise ValueError("language must be 'en' or 'de'")

    # Log the input parameters for debugging.
    logger.debug(f'{query=}')  # Assuming we change the input name
    logger.debug(f'{top_k=}')
    logger.debug(f'{lang=}')

    # Process the query using the RAG pipeline.
    answer = pipeline.process_query(
        query=query,  # User query as a string
        top_k=top_k,  # Number of nearest neighbors to return
        lang=lang,  # Language passed through embedding
        content_key='statement',  # Content key for embedding
        meta_keys=[  # List of expected Wikidata keys
            'qid',
            'pid',
            'value',
            'item_label',
            'property_label',
            'value_content'
        ],
        wikidata_kwargs={  # Customising the Wikidata REST API routines
            'timeout': 10,
            'n_cores': cpu_count(),
            'verbose': False,
            'api_url': 'https://www.wikidata.org/w',
            'wikidata_base': '"wikidata.org"',
            'return_list': True
        }
    )

    # Log the metadata of documents in the answer for debugging.
    for doc in answer.documents:
        logger.debug(f'{doc.meta}')

    # Construct the sources list from the documents,
    #   only including those with a 'qid'.
    sources = [
        {
            "src": f"https://www.wikidata.org/wiki/{doc.meta['qid']}",
            "content": doc.content,
            "score": doc.score
        } for doc in answer.documents if 'qid' in doc.meta
        # Ensure 'qid' is present in the document metadata.
    ]

    # Log the complete answer for debugging.
    logger.debug(f'{answer=}')

    # Return the processed answer and sources.
    return {
        "answer": answer.data.content,
        "sources": sources
    }
