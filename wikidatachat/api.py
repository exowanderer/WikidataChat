import os

from typing import Annotated

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Header
from multiprocessing import Pool, cpu_count
from multiprocessing.dummy import Pool as ThreadPool

from .rag import rag_pipeline
from .textification import get_wikidata_statements_from_query
from .logger import get_logger

# Create logger instance from base logger config in `logger.py`
logger = get_logger(__name__)

FRONTEND_STATIC_DIR = './frontend/dist'
# API_SECRET = os.environ.get("API_SECRET")

app = FastAPI()

app.mount(
    "/assets",
    StaticFiles(directory=f"{FRONTEND_STATIC_DIR}/assets"),
    name="frontend-assets"
)


@app.get("/")
async def root():
    return FileResponse(f"{FRONTEND_STATIC_DIR}/index.html")


@app.get("/favicon.ico")
async def favicon():
    return FileResponse(f"{FRONTEND_STATIC_DIR}/favicon.ico")


# async def api(x_api_secret: Annotated[str, Header()], query, top_k=3, lang='en'):
@app.get("/api")
async def api(query, top_k=3, lang='en'):
    # if not API_SECRET == x_api_secret:
    #     raise Exception("API key is missing or incorrect")

    if not lang in ['en', 'de']:
        raise Exception("language must be 'en' or 'de'")

    logger.debug(f'{query=}')  # Assuming we change the input name
    logger.debug(f'{top_k=}')
    logger.debug(f'{lang=}')

    wikidata_statements = get_wikidata_statements_from_query(
        query,
        lang='en',
        timeout=10,
        n_cores=cpu_count(),
        verbose=False,
        api_url='https://www.wikidata.org/w',
        wikidata_base='"wikidata.org"',
        serapi_api_key=os.environ.get("SERAPI_API_KEY")
    )

    logger.debug(f'{wikidata_statements=}')

    answer = rag_pipeline(
        query=query,
        top_k=top_k,
        lang=lang
    )

    sources = [
        {
            "src": d_.meta['src'],
            "content": d_.content,
            "score": d_.score
        } for d_ in answer.documents
    ]

    logger.debug(f'{answer=}')

    return {
        "answer": answer.data.content,
        "sources": sources
    }
