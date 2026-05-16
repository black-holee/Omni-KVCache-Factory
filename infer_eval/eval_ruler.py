import os
import json
import argparse
import numpy as np

from eval.metrics import (
    string_match_all
)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument("--retain_rate", type=float, default=0.2, help="retain rate of KV entries")
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--tsp_rate", type=float, default=0.6, help="tsp_rate used for proportional eviction mode")
    parser.add_argument("--tsp_idx", type=int, default=15, help="")
    parser.add_argument("--context_length", type=int, default=8192, help="")
    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()
    
    dataset_list = [
        "niah_single_1", 
        "niah_single_2", 
        "niah_single_3", 
        "niah_multikey_1", 
        "niah_multikey_2", 
        "niah_multikey_3",
        "niah_multiquery", 
        "niah_multivalue", 
        "cwe", 
        "fwe", 
        "vt",
        "qa_1",
        "qa_2",
    ]

    method_list = [
        "fullkv",
        "streamingllm",
        "h2o",
        "snapkv",
        "gemfilter",
        "fastkv"
    ]
    
    results_list = [["dataset"]]
    for method in method_list:
        results_list.append([method])
    
    model2maxlen = {
        "llama2": 3950,
        "llama-2": 3950,
        "llama3": 7950,
        "llama-3": 7950,
        "mistral": 127500,
        "ministral": 127500,
        "llama-3.1": 127500,
        "qwen3": 127500,
    }
    model_path = args.model_path.lower()
    for key in model2maxlen:
        if key in model_path:
            model_max_len = model2maxlen[key]

    context_length = args.context_length
    retain_rate=args.retain_rate
    window_size=args.window_size
    tsp_rate=args.tsp_rate
    tsp_idx=args.tsp_idx
    
    for dataset in dataset_list:
        
        results_list[0].append(dataset)
        
        for idx, method in enumerate(method_list):
            try:
                args.method = method
                args.dataset = dataset
                if args.method in ["fastkv"]:
                    args.eval_file = os.path.join(args.results_dir, str(context_length), args.dataset, f"{args.method}_{model_max_len}_{retain_rate}_{window_size}_tsp_rate_{tsp_rate}_tsp_idx_{tsp_idx}.json")
                elif args.method in ["fullkv"]:
                    args.eval_file = os.path.join(args.results_dir, str(context_length), args.dataset, f"{args.method}_{model_max_len}.json")
                else:
                    args.eval_file = os.path.join(args.results_dir, str(context_length), args.dataset, f"{args.method}_{model_max_len}_{args.retain_rate}_{args.window_size}.json")
                
                scores = dict()
                predictions, answers, lengths = [], [], []
                # dataset = filename.split('.')[0]
                with open(args.eval_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            predictions.append(data["pred"])
                            answers.append(data["answers"])
                            if "length" in data:
                                lengths.append(data["length"])
                        except:
                            print("error")
                
                score = string_match_all(predictions, answers)
                scores[args.dataset] = score
                results_list[idx+1].append(score)
            except:
                results_list[idx+1].append(-1)
                # print(f"dataset {args.dataset} method {args.method} scores {None}")
                
    import csv
    with open(os.path.join(args.results_dir, str(context_length), f"results_{model_max_len}_{retain_rate}_{window_size}.csv"), 'w') as fp:
        writer = csv.writer(fp)
        writer.writerows(results_list)
