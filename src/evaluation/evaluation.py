import time
import pm4py
import itertools
from pathlib import Path

from src.commons.log_utils import LogData
from src.commons import shared_variables as shared
from src.commons.utils import extract_bk_filename, extract_last_model_checkpoint, extract_Declare_bk_model
from src.evaluation.prepare_data import prepare_encoded_data
from src.evaluation.inference_algorithms import beamsearch_cf, baseline_cf
from pm4py.algo.simulation.playout.petri_net import algorithm
from pm4py.algo.simulation.playout.petri_net.variants import extensive
from pm4py.algo.simulation.playout.petri_net.variants.extensive import Parameters
from src.ProbDeclmonitor.probDeclPredictor import AggregationMethod

def evaluate_all(log_data: LogData, models_folder: str, alg: str, method_fitness: str, weight: list, resource: bool, timestamp: bool, outcome: bool):
    start_time = time.time()

    maxlen = log_data.maxlen
    predict_size = maxlen
    chars, chars_group, act_to_int, target_act_to_int, target_int_to_act,res_to_int, target_res_to_int, target_int_to_res \
        = prepare_encoded_data(log_data,resource)

    bk_filename = extract_bk_filename(log_data.log_name.value, log_data.evaluation_prefix_start)

    if (method_fitness == "conformance_diagnostics_alignments_prefix"):
        if 'bpmn' in str(bk_filename):
            bpmn = pm4py.read_bpmn(str(bk_filename))
            net, initial_marking, final_marking = pm4py.convert_to_petri_net(bpmn)
            pm4py.write_pnml(net, initial_marking, final_marking, str(bk_filename).split(".")[0] + ".pnml")

        else:
            net, initial_marking, final_marking = pm4py.read_pnml(bk_filename)
            
        print("Start unfolding petrinet")
        sim_log = algorithm.apply(net, initial_marking, final_marking=final_marking, variant=extensive, parameters= {Parameters.MAX_TRACE_LENGTH: min(30, maxlen) })
        sim_data = pm4py.convert_to_dataframe(sim_log)
        print("Finished unfolding petrinet")

        prefix = list(sim_data.groupby('case:concept:name').apply(lambda x: list(range(1, len(x)+1))))
        prefix = list(itertools.chain(*prefix))
        sim_data['prefix'] = prefix
        
        for prefix_len in range(log_data.evaluation_prefix_start, predict_size+1):
            sim_data_prefix = sim_data.loc[sim_data['prefix'] < prefix_len+1].reset_index(drop= True)
            net_prefix, im_prefix, fm_prefix = pm4py.discover_petri_net_inductive(sim_data_prefix)
            pm4py.write_pnml(net_prefix, im_prefix, fm_prefix, str(bk_filename).split(".")[0] + "_" + str(prefix_len) + ".pnml")
    
    evaluation_traces = log_data.log[log_data.log[log_data.case_name_key].isin(log_data.evaluation_trace_ids)]
    compliant_traces = evaluation_traces

    tree = pm4py.discover_process_tree_inductive(evaluation_traces, noise_threshold = 0.45,  activity_key=log_data.act_name_key,
                                                        case_id_key=log_data.case_name_key,
                                                        timestamp_key= log_data.timestamp_key)    
    net, initial_marking, final_marking = pm4py.convert_to_petri_net(tree)
    
    pm4py.write_pnml(net, initial_marking, final_marking, bk_filename)

    print("Compliant traces: " + str(compliant_traces[log_data.case_name_key].nunique())
          + " out of " + str(len(log_data.evaluation_trace_ids)))
    print('Elapsed time:', time.time() - start_time)

    models_folder += "_One_hot" * (shared.One_hot_encoding and not shared.use_modulator) + \
                     "_Combined_Act_res" * (shared.combined_Act_res and not shared.use_modulator) + \
                     "_Multi_Enc" * (shared.use_modulator and not shared.One_hot_encoding) + \
                     "_Multi_One_hot_Enc" * (shared.use_modulator and shared.One_hot_encoding) + \
                     "_Simple_categorical" * (not shared.One_hot_encoding and not shared.combined_Act_res and not shared.use_modulator)

    # extract declare model
    bk_model = None
    if shared.declare_BK or shared.BK_end:
        bk_model = extract_Declare_bk_model(log_data.log_name.value)
    for fold in range(shared.folds):
        eval_algorithm = alg + "_cf" + "r"*resource + "t"*timestamp + "o"*outcome
        start_time = time.time()

        folder_path = shared.output_folder / models_folder / str(fold) / 'results' / eval_algorithm
        if not Path.exists(folder_path):
            Path.mkdir(folder_path, parents=True)

        print(f"fold {fold} - {eval_algorithm}")
        if alg == "beamsearch":
            if shared.declare_BK:
                for method in AggregationMethod:
                    shared.aggregationMethod = method
                    output_filename = folder_path / (
                    f'{log_data.log_name.value}{shared.test_log}_beam{str(shared.beam_size)}_fold{str(fold)}_cluster{log_data.evaluation_prefix_start}'
                    f'{method}{("_BK") * shared.declare_BK}.csv')
                    print('beamsearch')
                    model_filename = extract_last_model_checkpoint(log_data.log_name.value, models_folder, fold,
                                                               'CF' + 'R' * resource + 'O' * outcome)
                    beamsearch_cf.run_experiments(log_data, compliant_traces, maxlen, predict_size, act_to_int,
                                              target_act_to_int, target_int_to_act, res_to_int, target_res_to_int,
                                              target_int_to_res, model_filename, output_filename, bk_filename,
                                              method_fitness, resource, outcome, weight, bk_model)
            else:
                output_filename = folder_path / (f'{log_data.log_name.value}{shared.test_log}_beam{str(shared.beam_size)}_fold{str(fold)}_cluster{log_data.evaluation_prefix_start}'
                                             f'{"_probability_reduction" * shared.useProb_reduction}{("_BK")* shared.declare_BK}{("_BK_at_end")* shared.BK_end}.csv')
                print('beamsearch')
                model_filename = extract_last_model_checkpoint(log_data.log_name.value, models_folder, fold, 'CF' + 'R'*resource + 'O'*outcome)
                beamsearch_cf.run_experiments(log_data, compliant_traces, maxlen, predict_size, act_to_int,
                                            target_act_to_int, target_int_to_act, res_to_int, target_res_to_int,
                                            target_int_to_res, model_filename, output_filename, bk_filename, method_fitness, resource, outcome, weight, bk_model)
        elif alg=="baseline":
            output_filename = folder_path / f'{log_data.log_name.value}{shared.test_log}_{str(alg)}_fold{str(fold)}_cluster{log_data.evaluation_prefix_start}.csv'

            print('baseline')
            model_filename = extract_last_model_checkpoint(log_data.log_name.value, models_folder, fold, 'CF' + 'R'*resource + 'O'*outcome)
            baseline_cf.run_experiments(log_data, compliant_traces, maxlen, predict_size, act_to_int,
                                            target_act_to_int, target_int_to_act, res_to_int, target_res_to_int,
                                            target_int_to_res, model_filename, output_filename, bk_filename, method_fitness, resource, outcome, weight)
        else:
            raise RuntimeError(f"No evaluation algorithm called: '{eval_algorithm}'.")

        print("TIME TO FINISH --- %s seconds ---" % (time.time() - start_time))
