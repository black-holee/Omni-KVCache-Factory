import pandas as pd
import json

# 输入和输出文件路径
parquet_file = "/home/ubuntu/ck/FastKV/data/SQuAD/plain_text/train-00000-of-00001.parquet"
json_file = "/home/ubuntu/ck/FastKV/data/SQuAD/plain_text/train.json"

# 读取 Parquet 文件
print("正在读取 Parquet 文件...")
df = pd.read_parquet(parquet_file)

# 转换为 JSON 格式（每行一个 JSON 对象）
print("正在转换为 JSON 格式...")
records = df.to_dict(orient='records')
print(records[0])
# int32这种类型的数据居然不能被序列化，还得先转换成int才能存到json里
records = [{'id': record['id'], 'title': record['title'], 'context': record['context'], 'question': record['question'], 'answers': {'text': [ans_text_item for ans_text_item in record['answers']['text']], 'answer_start': [int(answer_start_item) for answer_start_item in record['answers']['answer_start']]}} for record in records]
print(records[0])

# 写入 JSON 文件
print("正在写入 JSON 文件...")
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(records, f, ensure_ascii=False, indent=4)

print(f"转换完成！JSON 文件已保存到: {json_file}")