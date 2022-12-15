import pandas as pd
from ast import literal_eval
from tqdm import tqdm
import pyrootutils
import json

root_path = pyrootutils.find_root(search_from=__file__, indicator=".git")
print("Set WD location to", root_path)
pyrootutils.set_root(
    path=root_path,
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
        cwd=True,
)

zillow = pd.read_csv('data/zillow_verified_data/nyu_object_dataset_sample.csv')[['image_keyword', 'image_keyword_hash']]
zillow['image_keyword_list'] = zillow['image_keyword'].apply(lambda x: x.split(','))
zillow['image_keyword_hash'] = zillow['image_keyword_hash'].apply(lambda x: literal_eval(x))

hash_keyword_map = pd.DataFrame()
zillow = zillow[['image_keyword_hash', 'image_keyword_list']]
for i in tqdm(range(len(zillow)), desc='building mapper'):
    row = zillow.iloc[i,:]
    hashes = row['image_keyword_hash']
    keywords = row['image_keyword_list']
    for j in range(len(hashes)):
        hash_keyword_map_row = [{'hash': hashes[j], 'keyword': keywords[j]}]
        hash_keyword_map = pd.concat([hash_keyword_map, pd.DataFrame(hash_keyword_map_row)])

mapper = {}
hash_keyword_map.drop_duplicates(inplace=True)
for i in range(len(hash_keyword_map)):
    row = hash_keyword_map.iloc[i,:]
    mapper[row['hash']] = row['keyword']

json.dump(mapper, open('notebooks/hash_keyword_mapper.json', 'w'))
print('Saved hash_keyword_mapper.json to notebooks folder.')