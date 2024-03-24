

def convert_value_to_string(
        wikidata_statement, property_label, lang='en',
        api_url: str = 'https://www.wikidata.org/w'):

    wikidata_data_type = wikidata_statement['property']['data-type']
    value_content = wikidata_statement['value']['content']

    if wikidata_data_type == 'wikibase-item':
        value_content, _ = get_item_json_from_wikidata(
            qid=value_content,
            key='labels',
            lang=lang,
            api_url=api_url
        )

    elif wikidata_data_type == 'time':
        value_content = value_content['time']
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
        value_content = value_content['amount']
        property_label = (
            f'has the quantity of {property_label} at'
        )

    elif wikidata_data_type == 'monolingualtext':
        lang_ = value_content['language']
        value_content = value_content['text']
        property_label = (
            f'has the {lang_} monolingual text identifier'
            f' of {property_label} at'
        )

    # elif wikidata_data_type == 'English':
    #     # print(wikidata_data_type, item_label, property_label, value_content)
    #     value_content = value_content['text']
    #     property_label = (
    #         f'has the {lang_} monolingual text identifier'
    #         f' of {property_label} at'
    #     )

    return property_label, value_content


def make_statement(
        prop_input, item_label, key=None, lang='en', timeout=100,
        api_url: str = 'https://www.wikidata.org/w', verbose=False):

    # if verbose:
    #     print('Now! Textify!')

    pid, properties = prop_input

    property_label, _ = get_property_json_from_wikidata(
        pid,
        key='labels',
        lang=lang,
        api_url=api_url
    )

    if len(property_label) == 0:
        return  # Skip this one

    statements = []
    for wikidata_statement_ in properties:
        property_label, value_content = convert_value_to_string(
            wikidata_statement=wikidata_statement_,
            property_label=property_label,
            lang=lang,
            api_url=api_url
        )

        if len(value_content) == 0:
            continue  # Skip this one

        statement_ = ''

        try:
            statement_ = ' '.join([item_label, property_label, value_content])

            # if verbose:
            #     print(statement_)

        except Exception as e:
            print(f'Found Error: {e}')

            if verbose:
                print()
                print(
                    wikidata_statement_['property']['data-type'],
                    item_label,
                    property_label,
                    value_content
                )

        statements.append(statement_)

    return statements


def convert_wikidata_item_to_statements(
        item_json: dict = None, api_url: str = 'https://www.wikidata.org/w',
        lang: str = 'en', timeout: float = 10, n_cores: int = cpu_count(),
        verbose: bool = False):

    if item_json is None:
        item_json = {}

    item_label = item_json['item_data']['labels'][lang]
    item_desc = item_json['item_data']

    if isinstance(item_desc, dict):
        item_desc = item_desc['descriptions']
    else:
        return ''  # blank

    if isinstance(item_desc, dict) and lang in item_desc:
        item_desc = item_desc[lang]
    else:
        return ''  # blank

    wikidata_statements = [f'{item_label} can be described as {item_desc}']

    item_statements = item_json['item_data']['statements']
    n_statements = len(item_statements.keys())

    item_pool = partial(
        make_statement,
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

    statements = np.unique(statements)

    # return statements
    return '\n'.join(statements)


def convert_wikipedia_item_to_statements(
        item_json: dict = None, api_url: str = 'https://www.wikidata.org/w',
        lang: str = 'en', timeout: float = 10, n_cores: int = cpu_count(),
        verbose: bool = False):

    item_label = item_json['item_data']['labels'][lang]
    item_desc = item_json['item_data']['descriptions'][lang]

    wikidata_statements = [f'{item_label} can be described as {item_desc}']

    item_statements = item_json['item_data']['statements']
    n_statements = len(item_statements.keys())

    item_pool = partial(
        make_statement,
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

    statements = np.unique(statements)

    # return statements
    return '\n'.join(statements)


def search_and_extract_html(query, serapi_api_key):
    # Step 1: Use SerpApi to search for the query and get URLs
    urls = search_query(query, serapi_api_key)

    # Step 2: Download the webpages and extract the text
    return download_and_extract_text(urls)


def search_and_extract_json(query, serapi_api_key):
    # Step 1: Use SerpApi to search for the query and get URLs
    urls = search_query(query, serapi_api_key)

    # Step 2: Download the json and transform the item into statements
    return download_and_extract_items(urls)


def get_wikidata_statements_from_query(
        question, lang='en', timeout=10, n_cores=cpu_count(), verbose=False,
        api_url='https://www.wikidata.org/w', wikidata_base='"wikidata.org"',
        serapi_api_key=None):

    assert (serapi_api_key is not None), (
        'get_wikidata_statements_from_query received serapi_api_key = None'
    )

    search_wikidata_query = ' '.join([wikidata_base, question])

    wikidata_items = search_and_extract_json(
        search_wikidata_query,
        serapi_api_key
    )

    wikidata_statements = [
        convert_wikidata_item_to_statements(
            item_json=wikidata_item_,
            api_url=api_url,
            lang=lang,
            timeout=timeout,
            n_cores=n_cores,
            verbose=verbose
        )

        for wikidata_item_ in wikidata_items
    ]

    text_output = '\n'.join(wikidata_statements)
    return text_output.replace('\n\n', '\n')


def get_wikipedia_statements_from_query(
        question, wikipedia_base='"wikipedia.org"', serapi_api_key=None):

    assert (serapi_api_key is not None), (
        'get_wikipedia_statements_from_query received serapi_api_key = None'
    )

    search_wikipedia_query = ' '.join([wikipedia_base, question])

    wikipedia_extracted_texts = search_and_extract_html(
        search_wikipedia_query,
        serapi_api_key
    )

    wikipedia_statements = [
        convert_wikipedia_item_to_statements(wikipedia_text_)
        for wikipedia_text_ in wikipedia_extracted_texts
    ]

    return '\n'.join(wikipedia_statements)
