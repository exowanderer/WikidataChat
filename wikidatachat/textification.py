from functools import partial  # For partial function application.

# For threading-based parallelism.
from multiprocessing import Pool, cpu_count
from multiprocessing.dummy import Pool as ThreadPool

from numpy import unique  # For removing duplicate elements in an array.
from tqdm import tqdm  # For displaying progress bars in loops.

# Importing utility functions from the 'retrieve' module.
from .retrieve import (
    download_and_extract_text,
    download_and_extract_items,
    get_item_json_from_wikidata,
    get_property_json_from_wikidata,
    search_query,
)
from .logger import get_logger  # Importing the logger configuration.

# Create logger instance from base logger config in `logger.py`
logger = get_logger(__name__)


WIKIDATA_API_URL = os.environ.get(
    'WIKIDATA_API_URL', 'https://www.wikidata.org/w'
)  # Default API URL for Wikidata.


def convert_value_to_string(
        wikidata_statement, property_label, lang='en',
        api_url: str = WIKIDATA_API_URL):
    """
    Converts a Wikidata statement's value to a string based on its data type.

    Args:
        wikidata_statement (dict): The Wikidata statement containing the value and data type.
        property_label (str): The label of the property for the statement.
        lang (str): Language code for label retrieval. Defaults to 'en'.
        api_url (str): Base URL for the Wikidata API. Defaults to 'https://www.wikidata.org/w'.

    Returns:
        tuple: A tuple containing the updated property label, value content, and raw value.
    """

    # Extracting the data type of the property.
    wikidata_data_type = wikidata_statement['property']['data-type']

    value = ''  # Initializing value and value content.
    value_content = ''
    if 'value' in wikidata_statement:  # Checking if the statement has a value.
        if 'content' in wikidata_statement['value']:
            # Checking if the value has content.
            value = wikidata_statement['value']['content']

    # Processing the value based on its data type and
    #   updating the property label accordingly.
    if wikidata_data_type == 'wikibase-item':
        # Fetching the item JSON for the value if it's a Wikibase item.
        value_content, _ = get_item_json_from_wikidata(
            qid=value,
            key='labels',
            lang=lang,
            api_url=api_url
        )
    elif wikidata_data_type == 'time':
        value_content = value['time']
        property_label = (
            f'has more information to be found at the {property_label}'
        )

    elif wikidata_data_type == 'external-id':
        property_label = (
            f'can be externally identified by the {property_label} as'
        )

    elif wikidata_data_type == 'commonsMediaid':
        property_label = (
            f'has the commonsMediaid of {property_label}'
        )

    elif wikidata_data_type == 'url':
        property_label = property_label.replace(' ', '_')
        property_label = (
            f'has more information to be found at {property_label}'
        )

    elif wikidata_data_type == 'quantity':
        value_content = value['amount']
        property_label = (
            f'has the quantity of {property_label} at'
        )

    elif wikidata_data_type == 'monolingualtext':
        lang_ = value['language']
        value_content = value['text']
        property_label = (
            f'has the {lang_} monolingual text identifier'
            f' of {property_label} at'
        )

    # elif wikidata_data_type == 'English':
    #     # logger.debug([
    #           wikidata_data_type,
    #           item_label,
    #           property_label,
    #           value
    #     ])
    #     value_content = value['text']
    #     property_label = (
    #         f'has the {lang_} monolingual text identifier'
    #         f' of {property_label} at'
    #     )

    # Return the updated property label, value content, and the raw value.
    return property_label, value_content, value


def make_statement(
        prop_input, item_label, qid=None, key=None, lang='en', timeout=100,
        api_url: str = WIKIDATA_API_URL, verbose=False):
    """
    Constructs a textual statement from a Wikidata property and its associated values.

    Args:
        prop_input (tuple): A tuple containing the property ID and the associated properties.
        item_label (str): The label of the Wikidata item.
        qid (str): The unique identifier of the Wikidata item. Optional.
        key (str): A specific part of the data to retrieve. Optional.
        lang (str): Language code for label retrieval. Defaults to 'en'.
        timeout (int): Timeout for API requests. Defaults to 100.
        api_url (str): Base URL for the Wikidata API. Defaults to 'https://www.wikidata.org/w'.
        verbose (bool): If True, enables verbose output for debugging. Defaults to False.

    Returns:
        list: A list of dictionaries containing statement information.
    """
    pid, properties = prop_input  # Unpacking the property ID and properties.

    # Fetching the property label from Wikidata.
    property_label, _ = get_property_json_from_wikidata(
        pid,
        key='labels',
        lang=lang,
        api_url=api_url
    )

    if len(property_label) == 0:
        return  # Skip this one

    statements = []  # Initializing a list to store constructed statements.
    for wikidata_statement_ in properties:
        # Converting each Wikidata statement to a textual statement.
        property_label, value_content, value = convert_value_to_string(
            wikidata_statement=wikidata_statement_,
            property_label=property_label,
            lang=lang,
            api_url=api_url
        )

        # Skipping if no property label is found.
        if len(value_content) == 0:
            continue  # Skipping statements with no value content.

        statement_ = ''  # Initializing the statement text.

        try:
            # Constructing the statement text.
            statement_ = ' '.join([item_label, property_label, value_content])

            # if verbose:
            #     logger.debug(statement_)

        except Exception as e:
            logger.debug(f'Found Error: {e}')  # Logging any exceptions.

            if verbose:
                logger.debug()
                logger.debug([
                    wikidata_statement_['property']['data-type'],
                    item_label,
                    property_label,
                    value_content
                ])

        # Appending the constructed statement information to the list.
        statements.append({
            'qid': qid,
            'pid': pid,
            'value': value if isinstance(value, str) else value_content,
            'item_label': item_label,
            'property_label': property_label,
            'value_content': value_content,
            'statement': statement_
        })

    return statements  # Returning the list of constructed statements.


def convert_wikidata_item_to_statements(
        item_json: dict = None, api_url: str = WIKIDATA_API_URL,
        lang: str = 'en', timeout: float = 10, n_cores: int = cpu_count(),
        return_list=False, verbose: bool = False):
    """
    Converts a Wikidata item JSON into structured statements.

    Args:
        item_json (dict, optional): JSON data of a Wikidata item.
        api_url (str): URL of the Wikidata API endpoint.
        lang (str): Language code for extracting labels and descriptions.
        timeout (float): Timeout duration for processing.
        n_cores (int): Number of cores to use for parallel processing.
        return_list (bool): Whether to return the statements as a list.
        verbose (bool): Enables verbose logging.

    Returns:
        list or str: A list of structured statements if 'return_list' is True, otherwise a string.
    """

    # Defaulting 'item_json' to an empty dict if not provided.
    if item_json is None:
        item_json = {}

    # Extracting essential item data from 'item_json'.
    qid = item_json['item_data']['id']
    item_label = item_json['item_data']['labels'][lang]
    item_desc = item_json['item_data'].get('descriptions', {}).get(lang, '')

    # Constructing an initial statement describing the item.
    wikidata_statements = [f'{item_label} can be described as {item_desc}']

    # Processing each statement associated with the item in parallel.
    item_statements = item_json['item_data']['statements']

    item_pool = partial(
        make_statement,
        qid=qid,
        item_label=item_label,
        lang=lang,
        timeout=timeout,
        api_url=api_url,
        verbose=verbose
    )

    with ThreadPool(n_cores) as pool:
        # Wrap pool.imap with tqdm for progress tracking
        pool_imap = pool.imap(item_pool, item_statements.items())
        results = list(tqdm(pool_imap, total=len(item_statements.items())))

    statements = []
    for res_ in results:
        statements.extend(res_)

    return statements
    # # statements = unique(statements)

    # # return string statements
    # statements = [wds_['statement']]
    # return '\n'.join(statements)


def convert_wikipedia_page_to_statements(
        item_json: dict = None, api_url: str = WIKIDATA_API_URL,
        lang: str = 'en', timeout: float = 10, n_cores: int = cpu_count(),
        verbose: bool = False):
    """
    Similar to 'convert_wikidata_item_to_statements', but specifically tailored for processing Wikipedia page data.

    The function structure and logic are similar, with differences potentially in handling specific Wikipedia data formats or additional processing specific to Wikipedia data.
    """
    qid = item_json['item_data']['id']
    item_label = item_json['item_data']['labels'][lang]
    item_desc = item_json['item_data']['descriptions'][lang]

    wikidata_statements = [f'{item_label} can be described as {item_desc}']

    # Processing each statement associated with the item in parallel.
    item_statements = item_json['item_data']['statements']

    # Setup a wrapper function to process each item per statement in parallel.
    item_pool = partial(
        make_statement,
        qid=qid,
        item_label=item_label,
        lang=lang,
        timeout=timeout,
        api_url=api_url,
        verbose=verbose
    )

    with ThreadPool(n_cores) as pool:
        # Parallel processing of item statements.
        pool_imap = pool.imap(item_pool, item_statements.items())

        # Wrap pool.imap with tqdm for progress tracking
        results = list(tqdm(pool_imap, total=len(item_statements.items())))

    # Aggregating results from parallel processing.
    for res_ in results:
        wikidata_statements.extend(res_)

    # Returning unique statements as list or joined statement
    return (
        unique(wikidata_statements)
        if return_list else
        '\n'.join(wikidata_statements)
    )


def search_and_extract_html(query, serapi_api_key):
    """
    Searches for a given query using SerpApi and extracts HTML content from the resulting URLs.

    Args:
        query (str): Search query.
        serapi_api_key (str): API key for SerpApi.

    Returns:
        list: Extracted HTML content from search result URLs.
    """
    # Step 1: Use SerpApi to search for the query and retrieve URLs
    urls = search_query(query, serapi_api_key)

    # Step 2: Download and extract HTML content from URLs.
    return download_and_extract_text(urls)


def search_and_extract_json(query, serapi_api_key):
    """
    Searches for a given query using SerpApi and extracts JSON data from the resulting URLs, transforming items into statements.

    Args:
        query (str): Search query.
        serapi_api_key (str): API key for SerpApi.

    Returns:
        list: Extracted JSON data transformed into statements.
    """
    # Step 1: Use SerpApi to search for the query and get URLs
    urls = search_query(query, serapi_api_key)

    # Step 2: Download JSON and transform items into statements.
    return download_and_extract_items(urls)


def get_wikidata_statements_from_query(
        question, lang='en', timeout=10, n_cores=cpu_count(), verbose=False,
        api_url=WIKIDATA_API_URL, wikidata_base='"wikidata.org"',
        return_list=True, serapi_api_key=None):
    """
    Retrieves structured statements from Wikidata based on a given query.

    Args:
        question (str): The query string to search for in Wikidata.
        lang (str): The language for the statements. Defaults to 'en'.
        timeout (int): The timeout for API requests. Defaults to 10.
        n_cores (int): The number of cores to use for parallel processing. Defaults to the CPU count.
        verbose (bool): If True, additional log information will be shown.
        api_url (str): The base URL for the Wikidata API. Defaults to WIKIDATA_API_URL.
        wikidata_base (str): The base search string for Wikidata. Defaults to '"wikidata.org"'.
        return_list (bool): Determines the format of the returned statements. If True, returns a list; otherwise, returns a string.
        serapi_api_key (str): The API key for SerpApi.

    Returns:
        list or str: A list of statements if return_list is True; otherwise, a string of concatenated statements.
    """

    # Ensure the SerpApi key is provided.
    assert serapi_api_key is not None, (
        'get_wikidata_statements_from_query received serapi_api_key = None'
    )

    # Construct the search query.
    search_wikidata_query = ' '.join([wikidata_base, question])

    # Fetch items from Wikidata using the constructed query.
    wikidata_items = search_and_extract_json(
        search_wikidata_query,
        serapi_api_key
    )

    # Ensure items were successfully fetched.
    assert wikidata_items is not None, {
        'api_url': api_url,
        'lang': lang,
        'timeout': timeout,
        'n_cores': n_cores,
        'return_list': return_list,
        'verbose': verbose
    }

    # Initialize an empty list to store the statements.
    wikidata_statements = []

    # Convert each item fetched from Wikidata into statements.
    for wikidata_item_ in wikidata_items:
        wikidata_statements.extend(
            convert_wikidata_item_to_statements(
                item_json=wikidata_item_,
                api_url=api_url,
                lang=lang,
                timeout=timeout,
                n_cores=n_cores,
                # return_list should always be True
                #   because we're extending the list.
                return_list=return_list,
                verbose=verbose
            )
        )

    # Return the statements either as a list or as a concatenated string,
    #   based on the return_list flag.
    return (
        wikidata_statements if return_list else
        '\n'.join(
            [wds_['statement'] for wds_ in wikidata_statements]
        ).replace('\n\n', '\n')
    )


def get_wikipedia_statements_from_query(
        question, wikipedia_base='"wikipedia.org"', serapi_api_key=None):
    """
    Retrieves structured statements from Wikipedia pages based on a given query.

    Args:
        question (str): The query string to search for in Wikipedia.
        wikipedia_base (str): The base search string for Wikipedia. Defaults to '"wikipedia.org"'.
        serapi_api_key (str): The API key for SerpApi.

    Returns:
        str: A string of concatenated statements extracted from the Wikipedia pages.
    """

    # Ensure the SerpApi key is provided.
    assert serapi_api_key is not None, (
        'get_wikipedia_statements_from_query received serapi_api_key = None'
    )

    # Construct the search query.
    search_wikipedia_query = ' '.join([wikipedia_base, question])

    # Fetch and extract texts from Wikipedia using the constructed query.
    wikipedia_extracted_texts = search_and_extract_html(
        search_wikipedia_query,
        serapi_api_key
    )

    # Convert each extracted text into structured statements.
    wikipedia_statements = [
        convert_wikipedia_item_to_statements(wikipedia_text_)
        for wikipedia_text_ in wikipedia_extracted_texts
    ]

    # Return the statements as a concatenated string.
    return '\n'.join(wikipedia_statements)
