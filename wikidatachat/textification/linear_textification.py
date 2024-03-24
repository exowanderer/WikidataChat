
def linear_convert_wikidata_item_to_statements(
        item_json: dict = {},
        api_url: str = 'https://www.wikidata.org/w', lang='en', timeout=100):

    item_label = item_json['item_data']['labels'][lang]
    item_desc = item_json['item_data']

    if isinstance(item_desc, dict):
        item_desc = item_desc['descriptions']
    else:
        return ''  # blank

    if isinstance(item_desc, dict):
        item_desc = item_desc[lang]
    else:
        return ''  # blank

    wikidata_statements = [f'{item_label} can be described as {item_desc}']

    item_data = item_json['item_data']
    for pid_, properties_ in tqdm(item_data['statements'].items()):
        property_label, _ = get_property_json_from_wikidata(
            pid=pid_,
            key='labels',
            lang=lang,
            api_url=api_url
        )

        if len(property_label) == 0:
            continue  # Skip this one

        # property_label = property_labels[lang]
        # property_desc = property_json['descriptions'][lang]

        for wikidata_statement_ in properties_:
            wikidata_data_type = wikidata_statement_['property']['data-type']
            value_content = wikidata_statement_['value']['content']
            if wikidata_data_type == 'wikibase-item':
                qid_ = value_content
                value_content, _ = get_item_json_from_wikidata(
                    qid=qid_,
                    key='labels',
                    lang=lang,
                    api_url=api_url
                )

                if len(value_content) == 0:
                    continue

                # value_content = value_contents[lang]
                # value_desc = value_json['descriptions'][lang]
            elif wikidata_data_type == 'time':
                value_content = value_content['time']

            elif wikidata_data_type == 'external-id':
                property_label = (
                    f'can be externally identified by the {property_label} as'
                )

            wikidata_statements.append(
                ' '.join(
                    [item_label, property_label, value_content]
                )
            )

    return wikidata_statements
