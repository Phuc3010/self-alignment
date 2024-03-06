import json
import pandas as pd

# with 
# CATEGORIES = ["Writing", "Roleplay", "Reasoning", "Math", "Coding", "Extraction", "STEM", "Humanities"]
# RESULTS_PATH = "data/mt_bench/model_judgment/gpt-4_single.jsonl"

# def get_model_df():
#     cnt = 0
#     q2result = []
#     fin = open(RESULTS_PATH, "r")
#     for line in fin:
#         obj = json.loads(line)
#         obj["category"] = CATEGORIES[(obj["question_id"]-81)//10]
#         q2result.append(obj)
#     df = pd.DataFrame(q2result)
#     return df

# def toggle(res_str):
#     if res_str == "win":
#         return "loss"
#     elif res_str == "loss":
#         return "win"
#     return "tie"

# def get_scores(df):
#     all_models = df["model"].unique()
#     scores_all = []
#     for model in all_models:
#         for cat in CATEGORIES:
#             res = df[(df["category"]==cat) & (df["model"]==model) & (df["score"] >= 0)]
#             score = res["score"].mean()
#             scores_all.append({"model": model, "category": cat, "score": score})
#     return scores_all

# df = get_model_df()
# scores_all = get_scores(df)
# target_models = ["zephyr-7b-sft", "zephyr-7b-SPIN-iter0", "zephyr-7b-SPIN-iter1", "zephyr-7b-SPIN-iter2", "zephyr-7b-SPIN-iter3", "zephyr-7b-dpo"]
# scores_target = [scores_all[i] for i in range(len(scores_all)) if scores_all[i]["model"] in target_models]

# df_score = pd.DataFrame(scores_target)
# df_score = df_score[df_score["model"].isin(target_models)]

# print(df_score)