train_data = '/home/ubuntu/ck/FastKV/data/SQuAD/plain_text/train.json'
flitered_train_data = '/home/ubuntu/ck/FastKV/data/SQuAD/plain_text/train_longer_than_four.json'

import json

with open(train_data, 'r', encoding='utf-8') as f:
    records = json.load(f)

records = [record for record in records if len(record['answers']['text'][0].split(' ')) > 4]
print(len(records))

with open(flitered_train_data, 'w', encoding='utf-8') as f:
    json.dump(records, f, indent=4, ensure_ascii=False)