import json
import pandas as pd

if __name__ == "__main__":
    experiment = "experiment_name"
    results_file = f"tested_{experiment}.json"

    with open(results_file) as f:
        obj = json.load(f)

    ds = []
    for row in obj.values():
        model = row['model_info']
        perf = row['performance']
        ds.append((model['model'], model['chunk_size'], model['fold'], perf['f1'][0], perf['f1'][1], perf['f1'][2], perf['accuracy']))

    df_x = pd.DataFrame(data=ds, columns=['model','chunk','type','f1_red','f1_yellow','f1_green','accuracy'])
    df_fold = df_x[df_x.type=="kfold"].groupby(["model", "chunk"]).mean()
    df_tt = df_x[df_x.type=="tt"].groupby(["model", "chunk"]).mean()

    df_harm = 2 * df_tt * df_fold / (df_tt + df_fold)
    print(df_harm)