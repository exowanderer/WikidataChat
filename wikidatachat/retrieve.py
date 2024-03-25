import requests
import json
import torch
import urllib

from bs4 import BeautifulSoup
from serpapi import GoogleSearch


def search_query(query, serapi_api_key, num_results=10):
    params = {
        "engine": "google",
        "q": query,
        "api_key": serapi_api_key,
        "num": num_results
    }

    search = GoogleSearch(params).get_dict()

    results = search.get("organic_results", [])

    return [result["link"] for result in results if "link" in result]


def download_and_extract_text(
        urls, wiki_url='https://www.wikidata.org/wiki/Q', timeout=100):
    headers = {'User-Agent': 'Mozilla/5.0'}
    texts = []

    for url in urls:
        if wiki_url not in url:
            continue  # Skip all non wiki urls

        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                texts.append({url: soup.get_text()})
        except Exception as e:
            print(f"Failed to process {url}: {e}")

    return texts


def get_json_from_wikidata(
        thing_id, thing='items', key=None, lang='en',
        api_url='https://www.wikidata.org/w', timeout=100,
        return_blank=False, verbose=False):

    # if thing_id in master_cache.keys():
    #     return master_cache[thing_id]

    # Check for HTML or API url version
    api_url = api_url[:-3] if api_url[:-3] == 'wiki' else api_url

    entity_restapi = 'rest.php/wikibase/v0/entities'
    thing_url = '/'.join([api_url, entity_restapi, thing, thing_id])

    # thing_url = f'{api_url}/rest.php/wikibase/v0/entities/{thing}/{thing_id}'

    if key is not None:
        thing_url = '/'.join([thing_url, key])

        if lang is not None:
            thing_url = '/'.join([thing_url, lang])

    counter = 0
    while True:
        try:
            with urllib.request.urlopen(thing_url) as j_inn:
                # Decode and parse the JSON data
                j_inn_text = j_inn.read().decode('utf-8')

            # master_cache[thing_id] = json.loads(j_inn_text), thing_url
            return json.loads(j_inn_text), thing_url

        except Exception as e:
            if verbose:
                print(f"Error downloading {thing_url}: {e}")
                # master_cache[thing_id] = {}, thing_url

            if return_blank:
                return {}, thing_url

            if counter == timeout:
                print(f"Timout({counter}) reached; Error downloading ")
                print(f"{thing}:{thing_id}:{key}:{thing_url}")
                print(f"Error: {e}")

                return {}, thing_url

        counter = counter + 1


def get_item_json_from_wikidata(
        qid, key=None, lang='en',
        api_url='https://www.wikidata.org/w', verbose=False):

    # if qid in item_cache.keys():
    #     # if verbose:
    #     #     print(f'Returning {qid} from cache')
    #     return item_cache[qid]

    # if verbose:
    #     print(f'Computing new {qid} into cache')

    item_json, item_url = get_json_from_wikidata(
        thing_id=qid,
        thing='items',
        key=key,
        lang=lang,
        api_url='https://www.wikidata.org/w'
    )

    if len(item_json):
        # JSON Exists, store in property_cache
        # item_cache[qid] = item_json, item_url
        return item_json, item_url

    # No not store in cache
    return {}, item_url


def get_property_json_from_wikidata(
        pid, key=None, lang='en',
        api_url='https://www.wikidata.org/w', verbose=False):

    # if pid in property_cache.keys():
    #     # if verbose:
    #     #     print(f'Returning {pid} from cache')
    #     return property_cache[pid]

    # if verbose:
    #     print(f'Computing new {pid} into cache')

    property_json, property_url = get_json_from_wikidata(
        thing_id=pid,
        thing='properties',
        key=key,
        lang=lang,
        api_url='https://www.wikidata.org/w'
    )

    if len(property_json):
        # JSON Exists, store in property_cache
        # property_cache[pid] = property_json, property_url
        return property_json, property_url

    # No not store in cache
    return {}, property_url


def download_and_extract_items(
        urls, wiki_url='https://www.wikidata.org/wiki', lang='en', timeout=100):

    headers = {'User-Agent': 'Mozilla/5.0'}
    items = []

    # json_template = f"{wiki_url}/Special:EntityData/" + "{}.json"
    qid_base = f'{wiki_url}/Q'

    for url in urls:
        if qid_base not in url:
            continue  # Skip all non wiki urls

        try:
            qid_ = url.split('/')[-1]
            item_json, item_url = get_item_json_from_wikidata(
                qid=qid_,
                api_url=wiki_url  # Send html version for safe keeping
            )

            if len(item_json) == 0:
                return

            # # Cache this as well
            # item_cache[qid_] = item_json, item_url

            items.append({
                'html_url': url,
                'item_url': item_url,
                'item_data': item_json,
            })

        except Exception as e:
            print(f'Failed to process {url}: {e}')

    return items
