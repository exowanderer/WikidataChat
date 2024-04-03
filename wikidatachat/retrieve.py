import os
import json
import requests  # For making HTTP requests.
import urllib  # For opening and reading URLs.

from bs4 import BeautifulSoup  # For parsing HTML and extracting information.

# For performing Google searches using SerpApi.
from serpapi import GoogleSearch

from .logger import get_logger  # For logging within the application.
logger = get_logger(__name__)  # Initialize the logger for this module.

# Base URL for Wikidata API, with a default value.
WIKIDATA_API_URL = os.environ.get(
    'WIKIDATA_API_URL',
    'https://www.wikidata.org/w'
)


def search_query(query, serapi_api_key, num_results=10):
    """
    Performs a Google search for the given query using SerpApi and returns the links of the organic results.

    Args:
        query (str): The search query.
        serapi_api_key (str): The API key for SerpApi.
        num_results (int): The number of search results to return.

    Returns:
        list: A list of URLs from the organic search results.
    """

    # Set up the parameters for the SerpApi search.
    params = {
        "engine": "google",  # Specify the search engine.
        "q": query,  # The search query.
        "api_key": serapi_api_key,  # The SerpApi key for authentication.
        "num": num_results  # The number of results to return.
    }

    # Perform the search using SerpApi and get the results as a dictionary.
    search = GoogleSearch(params).get_dict()

    # Extract the organic results from the search response.
    results = search.get("organic_results", [])

    # Return the links of the organic search results.
    return [result["link"] for result in results if "link" in result]


def download_and_extract_text(
        urls, wiki_url='https://www.wikidata.org/wiki/Q', timeout=100):
    """
    Downloads the content from the given URLs and extracts text using BeautifulSoup.

    Args:
        urls (list): A list of URLs to download and extract text from.
        wiki_url (str): The base URL for Wikidata items. Only URLs containing this base URL are processed.
        timeout (int): The timeout for the HTTP requests in seconds.

    Returns:
        list: A list of dictionaries, each containing the URL and the extracted text from that URL.
    """
    # Set the User-Agent header for the HTTP request.
    headers = {'User-Agent': 'Mozilla/5.0'}

    texts = []  # Initialize an empty list to store the extracted texts.

    # Iterate over the provided URLs.
    for url_ in urls:
        if wiki_url not in url_:
            continue  # Skip all non wiki urls

        try:
            # Make an HTTP GET request to the URL.
            response = requests.get(url_, headers=headers, timeout=timeout)

            # Check if the request was successful.
            if response.status_code == 200:
                # Parse the response content using BeautifulSoup.
                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract and append the text content to the list.
                texts.append({url_: soup.get_text()})
        except Exception as e:
            # Log any exceptions that occur.
            logger.debug(f"Failed to process {url_}: {e}")

    return texts  # Return the list of extracted texts.


def get_json_from_wikidata(
        thing_id, thing='items', key=None, lang='en',
        api_url=WIKIDATA_API_URL, timeout=100,
        return_blank=False, verbose=False):
    """
    Retrieves JSON data from the Wikidata API for a specified item or property.

    Args:
        thing_id (str): The ID of the item or property to retrieve.
        thing (str): The type of thing to retrieve ('items' or 'properties').
        key (str, optional): A specific part of the data to retrieve.
        lang (str): The language of the data to retrieve.
        api_url (str): The base URL of the Wikidata API.
        timeout (int): The number of attempts to make before giving up.
        return_blank (bool): Whether to return a blank result on failure.
        verbose (bool): Whether to output verbose debug information.

    Returns:
        tuple: A tuple containing the JSON data and the final URL used for the API request.
    """
    # if thing_id in master_cache.keys():
    #     return master_cache[thing_id]

    # Adjust the API URL if it ends with 'wiki' by removing
    #   the last 3 characters.
    api_url = api_url[:-3] if api_url[:-3] == 'wiki' else api_url

    # Construct the URL for the API request.
    entity_restapi = 'rest.php/wikibase/v0/entities'
    thing_url = '/'.join([api_url, entity_restapi, thing, thing_id])

    # Add additional parts to the URL if 'key' and 'lang' are specified.
    if key is not None:
        thing_url = '/'.join([thing_url, key])

        if lang is not None:
            thing_url = '/'.join([thing_url, lang])

    counter = 0  # Initialize a counter for tracking attempts.
    while True:
        if 'items//' in thing_url:
            # Return empty result if the URL is malformed.
            return {}, thing_url

        try:
            # Open the URL and read the response.
            with urllib.request.urlopen(thing_url) as j_inn:
                # Decode and parse the JSON data
                j_inn_text = j_inn.read().decode('utf-8')

            # master_cache[thing_id] = json.loads(j_inn_text), thing_url
            # Parse the JSON data and return it along with the URL.
            return json.loads(j_inn_text), thing_url

        except Exception as e:
            # Log errors if verbose mode is enabled.
            if verbose:
                logger.debug(f"Error downloading {thing_url}: {e}")
                # master_cache[thing_id] = {}, thing_url

            if return_blank:
                # Return empty result if 'return_blank' is True
                #   or if the timeout is reached.

                return {}, thing_url

            if counter == timeout:
                logger.debug(f"Timout({counter}) reached; Error downloading ")
                logger.debug(f"{thing}:{thing_id}:{key}:{thing_url}")
                logger.debug(f"Error: {e}")

                return {}, thing_url

        counter = counter + 1  # Increment the counter for each attempt.

    # Log if the function exits the loop without returning.
    logger.debug("End up with None-thing")


def get_item_json_from_wikidata(
        qid, key=None, lang='en',
        api_url=WIKIDATA_API_URL, verbose=False):
    """
    Fetches JSON data for a specified Wikidata item using its QID.

    Args:
        qid (str): The unique identifier for the Wikidata item.
        key (str, optional): A specific part of the item data to retrieve. Defaults to None.
        lang (str): The language code for the data retrieval. Defaults to 'en' for English.
        api_url (str): The base URL of the Wikidata API. Defaults to the value of WIKIDATA_API_URL.
        verbose (bool): If True, enables verbose output for debugging purposes. Defaults to False.

    Returns:
        tuple: A tuple containing the item JSON data and the URL used for the API request.
    """
    # if qid in item_cache.keys():
    #     return item_cache[qid]

    # Fetch JSON data from Wikidata using the general-purpose
    #   function get_json_from_wikidata.
    item_json, item_url = get_json_from_wikidata(
        thing_id=qid,
        thing='items',
        key=key,
        lang=lang,
        api_url=WIKIDATA_API_URL
    )

    # If the JSON data is not empty, return it along with the URL.
    if len(item_json):
        # JSON Exists, store in property_cache
        # item_cache[qid] = item_json, item_url
        return item_json, item_url

    # If there's no data, return empty dictionary and the URL.
    return {}, item_url


def get_property_json_from_wikidata(
        pid, key=None, lang='en',
        api_url=WIKIDATA_API_URL, verbose=False):
    """
    Fetches JSON data for a specified Wikidata property using its PID.

    Args:
        pid (str): The unique identifier for the Wikidata property.
        key (str, optional): A specific part of the property data to retrieve. Defaults to None.
        lang (str): The language code for the data retrieval. Defaults to 'en' for English.
        api_url (str): The base URL of the Wikidata API. Defaults to the value of WIKIDATA_API_URL.
        verbose (bool): If True, enables verbose output for debugging purposes. Defaults to False.

    Returns:
        tuple: A tuple containing the property JSON data and the URL used for the API request.
    """
    # if pid in property_cache.keys():
    #     return property_cache[pid]

    # Fetch JSON data from Wikidata using the general-purpose
    #   function get_json_from_wikidata.

    property_json, property_url = get_json_from_wikidata(
        thing_id=pid,
        thing='properties',
        key=key,
        lang=lang,
        api_url=WIKIDATA_API_URL
    )

    # If the JSON data is not empty, return it along with the URL.
    if len(property_json):
        # JSON Exists, store in property_cache
        # property_cache[pid] = property_json, property_url
        return property_json, property_url

    # If there's no data, return empty dictionary and the URL.
    return {}, property_url


def download_and_extract_items(
        urls, wiki_url='https://www.wikidata.org/wiki', lang='en', timeout=100):
    """
    Downloads and extracts item information from a list of Wikidata URLs.

    Args:
        urls (list): A list of URLs to process.
        wiki_url (str): The base URL for Wikidata. Defaults to 'https://www.wikidata.org/wiki'.
        lang (str): The language code for the data retrieval. Defaults to 'en' for English.
        timeout (int): The timeout in seconds for the network requests. Defaults to 100.

    Returns:
        list: A list of dictionaries containing item information extracted from each URL.
    """
    items = []  # Initialize an empty list to hold item information.

    # json_template = f"{wiki_url}/Special:EntityData/" + "{}.json"
    qid_base = f'{wiki_url}/Q'  # Construct the base URL for item QIDs.

    for url_ in urls:
        # Skip URLs that do not contain the QID base URL.
        if qid_base not in url_:
            continue  # Skip all non wiki urls

        try:
            qid_ = url_.split('/')[-1]  # Extract the QID from the URL.

            # Fetch item JSON data from Wikidata using the QID.
            item_json, item_url = get_item_json_from_wikidata(
                qid=qid_,
                api_url=wiki_url  # Send html version for safe keeping
            )

            # Skip processing if no item data is found.
            if len(item_json) == 0:
                return

            # # Cache this as well
            # item_cache[qid_] = item_json, item_url

            # Append a dictionary with item details to the items list.
            items.append({
                'html_url': url_,  # Wikidata URL.
                'item_url': item_url,  # API URL used to fetch the item.
                'item_data': item_json,  # item data extracted from response.
            })

        except Exception as e:
            # Log any exceptions that occur during processing.
            logger.debug(f'Failed to process {url_}: {e}')

    return items  # Return the list of extracted item information.
