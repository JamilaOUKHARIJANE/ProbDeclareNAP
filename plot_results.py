import pandas as pd
import matplotlib.pyplot as plt
import os

from pathlib import Path
from src.commons import log_utils, shared_variables as shared

import pdb
import csv
import numpy as np
from statistics import mean


def aggregate_results(log_path, alg,models_folder, beam_size, resource=False,timestamp=False,outcome=False,BK=False ):
    df = pd.DataFrame()
    for fold in range(shared.folds):
        eval_algorithm = alg + "_cf" + "r"*resource + "t"*timestamp + "o"*outcome
        folder_path = shared.output_folder / models_folder / str(fold) / 'results' / eval_algorithm
        log_name = log_path.stem
        if log_name == 'helpdesk':
            evaluation_prefix_start = 5 // 2 - 1
        elif log_name == "BPI2011":
            evaluation_prefix_start = 92 // 2 - 2
        elif log_name == "BPI2013":
            evaluation_prefix_start = 4 // 2 - 1
        elif log_name == "BPI2012":
            evaluation_prefix_start = 32 // 2 - 2
        elif log_name == "BPI2017":
            evaluation_prefix_start = 54 // 2 - 2
        if alg == "beamsearch":
            filename = f'{log_name}_beam{str(beam_size)}_fold{str(fold)}_cluster{evaluation_prefix_start}{"_mean_BK" * BK}.csv'
        else:
            filename = f'{log_name}_{str(alg)}_fold{str(fold)}_cluster{evaluation_prefix_start}.csv'

        df_results = pd.read_csv(os.path.join(folder_path, filename))
        grouped_df = df_results.groupby(['Prefix length'], as_index=False).agg(
            {'Damerau-Levenshtein Acts': 'mean', 'Damerau-Levenshtein Resources': 'mean'})
        df = pd.concat([df, grouped_df])
    grouped_df = df_results.groupby(['Prefix length'], as_index=False).agg(
        {'Damerau-Levenshtein Acts': 'mean', 'Damerau-Levenshtein Resources': 'mean'})
    return grouped_df

def add_plot(axs, dataset, metric, results,encoder):
    handles = []
    labels = []
    for i in results.keys():
        result_list = results[i][metric]
        prefix_length_list = results[i]["Prefix length"]
        line, =axs.plot(prefix_length_list, result_list,
                 color=shared.method_color[i],
                 marker=shared.method_marker[i],
                 label=i)
        handles.append(line)
        labels.append(i)
    return handles, labels

def plot_results(dataset_results,prefix_length):
    for metric in ["Damerau-Levenshtein Acts", "Damerau-Levenshtein Resources"]:
        folder_path = Path.cwd() / 'evaluation plots' / metric
        if not Path.exists(folder_path):
            Path.mkdir(folder_path, parents=True)

        f, ax = plt.subplots(2, 4, figsize=(16, 8))
        titles = ["One-hot Encoding", "Index-based Encoding", "Product Index-based", "Multi-Encoders"]
        for i, dataset in enumerate(dataset_results.keys()):
            if i == 0:
                group = [ax[0][0], ax[0][1], ax[1][0], ax[1][1]]
                a = 0.25
            else:
                group = [ax[0][2], ax[0][3], ax[1][2], ax[1][3]]
                a = 0.75
            for j, (encoder, subplot) in enumerate(zip(dataset_results[dataset].keys(), group)):
                results = dataset_results[dataset][encoder]
                subplot.set_title(titles[j], fontsize=12)#, pad=15)
                subplot.set_xlabel('Prefix length')#, labelpad=10)
                subplot.set_ylabel(f'Avg. {metric}')#, labelpad=15)
                handles, labels = add_plot(subplot, dataset, metric, results, encoder)
                subplot.grid()
            f.text(a, 0.97, dataset, fontsize=16, fontweight="bold", ha="center")  # Increased vertical position

        plt.tight_layout(rect=[0, 0.15, 1, 0.6])
        f.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
        # Create the legend
        legend = f.legend(labels=labels, title="Method", bbox_to_anchor=(0.5, 0.03), loc="lower center",
                          ncol=3, borderaxespad=0., title_fontsize='large', fontsize='large')

        # Set the title font weight to bold
        legend.get_title().set_fontweight('bold')

        # Save the figure
        title = f"average_{metric}_similarity_results"
        plt.savefig(os.path.join(folder_path, f'{title}.pdf'))
        plt.close()

def prepare_data():
    dataset_results = {}
    prefix_result = []
    for dataset_id, dataset in enumerate(shared.log_list):
        # read evaluation data
        if dataset == 'helpdesk.xes':
            evaluation_prefix_start = 5 // 2 - 1
        elif dataset == "BPI2011.xes":
            evaluation_prefix_start = 92 // 2 - 2
        elif dataset == "BPI2013.xes":
            evaluation_prefix_start = 4 // 2 - 1
        elif dataset == "BPI2012.xes":
            evaluation_prefix_start = 32 // 2 - 2
        elif dataset == "BPI2017.xes":
            evaluation_prefix_start = 54 // 2 - 2
        log_path = shared.log_folder / dataset
        df = pd.DataFrame()
        results_enc = {}
        for encoder in ["_One_hot","_Simple_categorical", "_Combined_Act_res","_Multi_Enc"]:
            results = {}
            try:
                data = aggregate_results(log_path, "baseline","keras_trans"+encoder, 0, resource=True)
                results["baseline"] = data
            except FileNotFoundError as not_found:
                pass
            for i in [3, 5,10]:
                try:
                    data = aggregate_results(log_path, "beamsearch", "keras_trans"+encoder, i, resource=True)
                    x= "beamsearch " + "(beam size = " + str(i) + ")"
                    results[x] = data
                except FileNotFoundError as not_found:
                    pass
            for i in [3, 5, 10]:
                try:
                    data = aggregate_results(log_path, "beamsearch", "keras_trans"+encoder, i, resource=True, BK=True)
                    x = "beamsearch" + " with BK" + " (beam size = " + str(i) + ")"
                    results[x] = data
                except FileNotFoundError as not_found:
                    pass
            if encoder == "_One_hot":
                encoders= "One-hot Encoding"
            elif encoder == "_Simple_categorical":
                encoders = "Index-based Encoding"
            elif encoder == "_Combined_Act_res":
                encoders = "Product Index-based Encoding"
            elif encoder =="_Multi_Enc":
                encoders = "Multi-Encoders"
            results_enc[encoders] = results
        dataset_results[(dataset.removesuffix('.xes')).capitalize() if dataset.removesuffix('.xes').islower() else dataset.removesuffix('.xes')] = results_enc
        prefix_result.append(list(range(evaluation_prefix_start, evaluation_prefix_start+ 5)))
    plot_results(dataset_results,prefix_result)

prepare_data()
