from sklearn.metrics import mean_squared_error

from src.ProbDeclmonitor.probDeclPredictor import AggregationMethod
from src.commons import shared_variables as shared
import csv
import os
import pandas as pd
from statistics import mean


def aggregate_results(log_path, alg,models_folder,beam_size=3, resource=False,timestamp=False,outcome=False,probability_reduction=False, BK=False, weight=0.0, method=AggregationMethod.SUM):
    average_act=[]
    average_res =[]
    average_length = []
    average_length_truth = []
    average_length_res = []
    for fold in range(shared.folds):
        eval_algorithm = alg + "_cf" + "r" * resource + "t" * timestamp + "o" * outcome
        if alg == "beamsearch":
            eval_algorithm = alg + "_cf" + "r" * resource + "t" * timestamp + "o" * outcome #+ "_old"
        folder_path = shared.output_folder / models_folder/ str(fold) / 'results' / eval_algorithm
        log_name = log_path.stem
        # if log_name =='helpdesk':
        evaluation_prefix_start = 5 // 2 - 1
        if alg == "beamsearch":
            if BK:
                filename =f'{log_name}_beam{str(beam_size)}_fold{str(fold)}_cluster{evaluation_prefix_start}{method}{"_BK" * BK}.csv'
            else:
                filename = f'{log_name}_beam{str(beam_size)}_fold{str(fold)}_cluster{evaluation_prefix_start}{"_probability_reduction" * probability_reduction}{"_BK" * BK}.csv'
        else:
            filename = f'{log_name}_{str(alg)}_fold{str(fold)}_cluster{evaluation_prefix_start}.csv'

        file_path = os.path.join(folder_path, filename)
        if not os.path.exists(file_path):
            return 0, 0, 0, 0
        df_results = pd.read_csv(file_path, delimiter=',')
        if resource:
            average_act.append(df_results['Damerau-Levenshtein Acts'].mean())
            average_res.append(df_results['Damerau-Levenshtein Resources'].mean())
            average_length.append(df_results['Predicted Acts'].str.len().mean())
            average_length_res.append(df_results['Predicted Resources'].str.len().mean())
            average_length_truth.append(df_results['Ground truth'].str.len().mean())
        else:
            df_results = df_results[df_results['Weight'] == weight]
            average_act.append(df_results['Damerau-Levenshtein'].mean())
            average_length.append(df_results['Predicted'].str.len().mean())
            average_length_truth.append(df_results['Ground truth'].str.len().mean())
    if resource:
        print(f"{log_name}_{models_folder} - {eval_algorithm}", ":", round(mean(average_act), 3), ":",
              round(mean(average_res), 3), ":", round(mean(average_length), 3), ":",
              round(mean(average_length_truth), 3))
        return round(mean(average_act), 3) , round(mean(average_res), 3), round(mean(average_length), 3) , round(mean(average_length_truth), 3)
    else:
        print(f"{log_name}_{models_folder} - {eval_algorithm}", ":", round(mean(average_act), 3), ":", round(mean(average_length), 3), ":",
              round(mean(average_length_truth), 3))
        return round(mean(average_act), 3), 0.0, round(mean(average_length), 3) , round(mean(average_length_truth), 3)


def getresults(log_list, algo, encoder, models_folder,prob_true_df,prob_pred_df,constraints, beam_size=3 ,resource=False, BK=False, weight=0.0, method=AggregationMethod.SUM):
    results = []
    for log in log_list:
        log_path = shared.log_folder / log
        average_act, average_res, length, _ = aggregate_results(log_path, algo, models_folder + encoder, beam_size=beam_size, resource=resource, BK=BK, weight=weight, method=method)
        results.append(average_act)
        #results.append(length)
        if BK:
            prob_true = prob_true_df[prob_true_df['Log'] == log]
            prob_pred = prob_pred_df[
            (prob_pred_df['Log'] == log) & (prob_pred_df['Encoder'] == encoder) & (prob_pred_df['Aggregation method'] == str(method)) & (prob_pred_df['weight'] == weight)]
            mse = mean_squared_error(prob_true[constraints], prob_pred[constraints])
            results.append(round(mse,3))
        else:
            results.append("")
        if resource:
            results.append(average_res)
    return results


if __name__ == "__main__":
   encoders = ["_One_hot","_Simple_categorical"]
   weights = [i / 10 for i in range(5, 10)]
   #weights = [0.9]
   beam_sizes = [0, 3]
   resource = False
   log_list = []
   for i in range(1, 6):
       log_list.append(shared.log_list[0].replace('.xes',"_D"+str(i)))
       log_list.append(shared.log_list[0].replace('.xes',"_D5_"+str(i)))

   prob_true = pd.read_csv(os.path.join(shared.output_folder, "prob_true.csv"), delimiter=',')
   prob_pred = pd.read_csv(os.path.join(shared.output_folder, "probabilities_results.csv"), delimiter=',')
   constraints = [
       "Alternate Precedence[Wait, Closed] | | |",
       "Alternate Response[Assign seriousness, Wait] | | |",
       "Alternate Precedence[Wait, Resolve ticket] | | |",
       "Exactly1[Wait] | |",
       "Chain Response[Take in charge ticket, Wait] | | |"
   ]
   with (open(os.path.join(shared.output_folder, f"aggregated_results.csv"), mode='w') as out_file):
        writer = csv.writer(out_file, delimiter=',')
        headers = ["Method", "Beam size", "Encoder"]
        sub_headers = ["", "", ""]
        for log in log_list:
            headers.extend([log, ""]) #if resource else headers.extend([log])
            sub_headers.extend(["Activities", "Resources"])  if resource else sub_headers.extend(["Activities", "MSE", "length"])
        headers.extend(['weight', "Aggregation method"])
        writer.writerow(headers)
        writer.writerow(sub_headers)
        for models_folder in ["keras_trans"]:
            for beam_size in beam_sizes:
                if beam_size == 0:
                    algo = 'baseline'
                    for encoder in encoders:
                        results = getresults(log_list,algo, encoder, models_folder, prob_true,prob_pred,constraints)
                        writer.writerow([algo, beam_size, encoder.removeprefix("_")]+[res for res in results]+[0.0, ""])
                else:
                    algo = 'beamsearch'
                    for encoder in encoders:
                        results = getresults(log_list, algo,encoder,models_folder,prob_true,prob_pred,constraints,beam_size=beam_size)
                        writer.writerow([algo,beam_size,encoder.removeprefix("_")]+[res for res in results]+[0.0,""])
                    for encoder in encoders:
                        for weight in weights:
                            for method in AggregationMethod:
                                results = getresults(log_list, algo,encoder,models_folder,prob_true,prob_pred,constraints,beam_size=beam_size, resource=resource, BK=True, weight=weight, method=method)
                                writer.writerow([algo + " + BK",beam_size,encoder.removeprefix("_")]+[res for res in results]+ [weight]+ [method])