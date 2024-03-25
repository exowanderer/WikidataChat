import json
import numpy as np
import pandas as pd

from argparse import ArgumentParser

clargs = ArgumentParser()
clargs.add_argument('-f', '--filename', type=str, default=None)
clargs.add_argument('-v', '--verbose', action='store_true', default=False)

args = clargs.parse_args()

with open(args.filename, 'r') as fin:
    wktext = json.load(fin)

# Chunking Strategy: paragrahs: split on \n\n
documents = []
for url_, wktext_ in wktext.items():
    for chunk_ in wktext_.split('\n\n'):
        documents.append(dict(content=chunk_, meta=url_))


fout_name = args.filename.replace('.json', '_paragraphs.json')

df_docs = pd.DataFrame(documents)
df_docs['content'] = df_docs['content'].replace({'': np.nan})
df_docs.dropna(inplace=True)
df_docs = df_docs.dropna().reset_index(drop=True)

"""
To forge the output in the format that Haystack requires:
    Pandas `to_dict()` produces three keys: `content`, `meta`, and `index`
    We must first tranpose the dict from a dict of lists into a list of dicts.
    That list of dicts includes the index as the outer key to each list element.
    The follow line transposes the dict and strips the indices to forge a list of dicts that only contain the `content` and `meta` keys.
"""
json_out = list(df_docs.T.to_dict().values())
with open(fout_name, 'w') as fout:
    json.dump(json_out, fout)

print(f'Paragraph chunked entries reorganised and saved to {fout_name}')
