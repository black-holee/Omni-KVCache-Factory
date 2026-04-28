import os
import json
import argparse
import numpy as np

from infer_eval.metrics import (
    qa_f1_score,
)

dataset2metric = {
    "train_longer_than_four": qa_f1_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=None)
    return parser.parse_args(args)

def scorer(dataset, predictions, answers):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    args = parse_args()
    
    dataset_list = [
        "train_longer_than_four",
        ]
    
    results_list = [
        ["dataset"],
        ["fullkv"],
        ["fastkv"],
        ["streamingllm"],
        ["h2o"],
        ["snapkv"],
        ["gemfilter"],
        ["pyramidinfer"]
    ]
    
    for dataset in dataset_list:
        results_list[0].append(dataset)
        
        for idx, method in enumerate(["fullkv", "streamingllm", "h2o", "snapkv", "pyramidinfer", "gemfilter", "fastkv"]):
            try:
                args.method = method
                args.dataset = dataset
                args.eval_file = os.path.join(args.results_dir, dataset, f"{method}.json")
                
                # try:
                
                scores = dict()

                predictions, answers, lengths = [], [], []

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
                score = scorer(args.dataset, predictions, answers)
                scores[args.dataset] = score
                    
                output_dir = os.path.dirname(args.eval_file)
                
                results_list[idx+1].append(score)
                
                with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                    json.dump(scores, f, ensure_ascii=False, indent=4)
            
                print(f"dataset {args.dataset} method {args.method} scores {scores}")
            except:

                results_list[idx+1].append(-1)
                
    import csv
    with open(os.path.join(args.results_dir, f"results.csv"), 'w') as fp:
        writer = csv.writer(fp)
        writer.writerows(results_list)

    # Print Excel-friendly CSV summary at the end for easy paste
    print("\nExcel-friendly summary (CSV):")
    header = ["method"] + dataset_list
    print(",".join(header))
    for idx, method in enumerate(["fullkv", "streamingllm", "h2o", "snapkv", "pyramidinfer", "gemfilter", "fastkv"]):
        # results_list[idx+1] is the row for this method; skip the first cell (method name)
        values = []
        for val in results_list[idx+1][1:]:
            if isinstance(val, (int, float, np.floating)):
                values.append(f"{float(val):.2f}")
            else:
                try:
                    values.append(f"{float(val):.2f}")
                except:
                    values.append(str(val))
        print(",".join([method] + values))
