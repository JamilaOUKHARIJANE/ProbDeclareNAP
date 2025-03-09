import pandas as pd
from enum import Enum
from pathlib import Path
import pm4py

import src.commons.shared_variables as shared
from src.commons.shared_variables import train_ratio, variant_split, validation_split

from pm4py.utils import get_properties
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.statistics.variants.log.get import get_variants

class LogName(Enum):
    SEPSIS = 'Sepsis_cases'
    UBE = "Synthetic"
    HELPDESK = 'helpdesk'
    BPIC11 = "BPI2011"
    BPIC12 = "BPI2012"
    BPIC13 = "BPI2013"
    BPIC13_In = "BPI2013_In"
    BPIC13_OP = "BPI2013_OP"
    BPIC13_CP = "BPI2013_CP"
    BPIC15_1 = "BPIC15_1"
    BPIC15_2 = "BPIC15_2"
    BPIC15_3 = "BPIC15_3"
    BPIC15_4 = "BPIC15_4"
    BPIC15_5 = "BPI2015_5"
    BPIC17 = "BPI2017"
    BPIC18 = "BPI2018"
    BPIC19 = "BPIC19"
    BPIC20_1 = "PrepaidTravelCost"
    BPIC20_2 = "PermitLog"
    BPIC20_3 = "RequestForPayment"
    BPIC20_4 = "DomesticDeclarations"
    BPIC20_5 = "InternationalDeclarations"

class LogExt(Enum):
    CSV = '.csv'
    XES = '.xes'
    XES_GZ = '.xes.gz'

class LogData:
    log: pd.DataFrame
    log_name: LogName
    log_ext: LogExt
    maxlen: int
    training_trace_ids = [str]
    evaluation_trace_ids = [str]

    # Gathered from encoding
    act_enc_mapping: {str, str}
    res_enc_mapping: {str, str}

    # Gathered from manual log analysis
    case_name_key: str
    act_name_key: str
    res_name_key: str
    timestamp_key: str
    timestamp_key2: str
    timestamp_key3: str
    timestamp_key4: str
    label_name_key: str
    label_pos_val: str
    label_neg_val: str
    compliance_th: float
    evaluation_th: float
    evaluation_prefix_start: int
    evaluation_prefix_end: int

    def __init__(self, log_path: Path, use_variant_split=False, resource = False):
        file_name = log_path.name
        if file_name.endswith('.xes') or file_name.endswith('.xes.gz'):
            if file_name.endswith('.xes'):
                self.log_name = LogName(log_path.stem)
                self.log_ext = LogExt.XES
            else:  # endswith '.xes.gz'
                self.log_name = LogName(log_path.with_suffix("").stem)
                self.log_ext = LogExt.XES_GZ

            self._set_log_keys_and_ths()
            if shared.use_train_test_logs:
                cols = [self.case_name_key, self.act_name_key, self.res_name_key, self.timestamp_key] if resource else [self.case_name_key, self.act_name_key, self.timestamp_key]
                train_log = pm4py.read_xes(str(log_path).replace(".xes", "_train.xes"))[cols]
                test_log = pm4py.read_xes(str(log_path).replace(".xes", f"_test{shared.test_log}.xes"))[cols]
                if resource:
                    test_log = test_log.dropna(subset=[self.res_name_key])
                    train_log = train_log.dropna(subset=[self.res_name_key])
                self.log = pd.concat([train_log, test_log], axis=0, ignore_index=True)
                print("nb of activities", len(self.log[self.act_name_key].unique()))
                if resource: print("nb of resource", len(self.log[self.res_name_key].unique()))
            else:
                self.log = pm4py.read_xes(str(log_path))[
                    [self.case_name_key, self.act_name_key, self.res_name_key, self.timestamp_key]]
            self.log[self.timestamp_key] = pd.to_datetime(self.log[self.timestamp_key])

        elif file_name.endswith('_1.csv')  or file_name.endswith('_2.csv')  or file_name.endswith('_3.csv')   : # for sepsis_1~3, hospital_billing_2~3
            self.log_name = LogName(log_path.stem)
            self.log_ext = LogExt.CSV
            self._set_log_keys_and_ths()
            self.log = pd.read_csv(
                log_path, sep= ";",
                usecols=[self.case_name_key, self.label_name_key, self.act_name_key, self.res_name_key, self.timestamp_key]
            )
            self.log[self.timestamp_key] = pd.to_datetime(self.log[self.timestamp_key])

        elif file_name.endswith('.csv') :
            self.log_name = LogName(log_path.stem)
            self.log_ext = LogExt.CSV
            self._set_log_keys_and_ths()
            self.log = pd.read_csv(
                log_path,
                usecols=[self.case_name_key, self.act_name_key, self.res_name_key, self.timestamp_key]
            )
            self.log.columns = self.log.columns.str.strip()  # Trim whitespace from headers
            self.log[self.case_name_key] = self.log[self.case_name_key].astype(str)
            self.log[self.timestamp_key] = pd.to_datetime(self.log[self.timestamp_key])

        else:
            raise RuntimeError(f"Extension of {file_name} must be in ['.xes', '.xes.gz', '.csv'].")
        print("DataFrame columns:", self.log.columns)

        trace_ids = self.log[self.case_name_key].unique().tolist()
        if not shared.use_train_test_logs:
            # Variant extraction
            parameters = get_properties(self.log, case_id_key=self.case_name_key, activity_key=self.act_name_key, timestamp_key=self.timestamp_key)
            log = log_converter.apply(self.log, variant=log_converter.Variants.TO_EVENT_LOG, parameters=parameters)
            variants = get_variants(log, parameters=parameters)
            print("Size of variants: ", len(variants))

            v_id = 0
            dict_cv = {}
            df_clusters = pd.DataFrame(columns=['prefix', 'variant', "variant_ID", 'case', 'supp'])
            prefix_length = self.evaluation_prefix_start

            for variant in variants:
                c = []
                for trace in variants[variant]:
                    dict_cv[trace.attributes['concept:name']] = "variant_" + str(v_id)
                    c.append(trace.attributes['concept:name'])

                prefix = str(variant[:prefix_length])
                row = [prefix, variant, v_id, str(c), len(c)]
                df_clusters = pd.concat([df_clusters, pd.Series(row, index=['prefix', 'variant', "variant_ID", "case", 'supp']).to_frame().T], ignore_index=True)
                v_id += 1

            prefix_count = df_clusters['prefix'].value_counts()
            list_prefix = prefix_count[prefix_count > 1].index
            df_clusters = df_clusters.loc[df_clusters['prefix'].isin(list_prefix)].reset_index(drop=True)

            if use_variant_split:
                variant_top2 = df_clusters.groupby('prefix', as_index=False).apply(lambda x: x['case'].tolist()[sorted(range(len(x['supp'])), reverse=True, key=lambda k: x['supp'].tolist()[k])[1]])
                variant_top2.columns = ['prefix', 'case']
                variant_top2['case'] = variant_top2['case'].apply(lambda x: eval(x))
                variant_top2['freq'] = variant_top2['case'].apply(lambda x: len(x))

                list_variant_top2 = variant_top2['case'].tolist()
                test_ids = []
                for i in list_variant_top2:
                    test_ids = test_ids + i
            else:
                # Simple Train/Test Split based on shared variables
                sorting_cols = [self.timestamp_key, self.act_name_key, self.res_name_key]
                self.log = self.log.sort_values(sorting_cols, ascending=True, kind='mergesort')
                grouped = self.log.groupby(self.case_name_key)
                start_timestamps = grouped[self.timestamp_key].min().reset_index()
                start_timestamps = start_timestamps.sort_values(self.timestamp_key, ascending=True, kind='mergesort')
                train_ids = list(start_timestamps[self.case_name_key])[:int(train_ratio * len(start_timestamps))]
                test_ids = [trace for trace in trace_ids if trace not in train_ids]
            filterByKey = lambda keys: {x: dict_cv[x] for x in keys}
            dict_cv_train = filterByKey(trace_ids)
            dict_cv_test = filterByKey(test_ids)
            # Outputs
            self.training_trace_ids = trace_ids if use_variant_split else train_ids
            self.case_to_variant = dict_cv
            self.case_to_variant_train = dict_cv_train
            self.case_to_variant_test = dict_cv_test
            self.evaluation_trace_ids = test_ids
        else:
            # Outputs
            self.training_trace_ids = train_log[self.case_name_key].unique().tolist()
            self.evaluation_trace_ids = test_log[self.case_name_key].unique().tolist()

        # check for new resources and activities
        training_traces = self.log[self.log[self.case_name_key].isin(self.training_trace_ids)]
        act_chars = list(training_traces[self.act_name_key].unique())
        testing_traces = self.log[self.log[self.case_name_key].isin(self.evaluation_trace_ids)]
        check_new_act = list(testing_traces[self.act_name_key].unique())
        new_chars = [na for na in check_new_act if na not in act_chars]
        if new_chars: print("new activities un-found in the training set", new_chars)
        if resource:
            chars_group = list(training_traces[self.res_name_key].unique())
            check_new_group = list(testing_traces[self.res_name_key].unique())
            new_chars_group = [na for na in check_new_group if na not in chars_group]
            if new_chars_group: print("new resource un-found in the training set", new_chars_group)

    def encode_log(self, resource: bool, timestamp: bool, outcome: bool):
        act_set = list(self.log[self.act_name_key].unique())
        self.act_enc_mapping = dict((chr(idx+shared.ascii_offset), elem) for idx, elem in enumerate(act_set))
        self.log.replace(to_replace={self.act_name_key: {v: k for k, v in self.act_enc_mapping.items()}}, inplace=True)

        if resource:
            res_set = list(self.log[self.res_name_key].unique())
            self.res_enc_mapping = dict((chr(idx+shared.ascii_offset), elem) for idx, elem in enumerate(res_set))
            self.log.replace(to_replace={self.res_name_key: {v: k for k, v in self.res_enc_mapping.items()}}, inplace=True)

        if timestamp:
            temp_time1 = self.log[[self.case_name_key, self.timestamp_key]]
            temp_time1['diff'] = temp_time1.groupby(self.case_name_key)[self.timestamp_key].diff().dt.seconds
            temp_time1['diff'].fillna(0, inplace=True)
            temp_time1['diff'] = temp_time1['diff']/max(temp_time1['diff'])  
            temp_time1['diff_cum'] = temp_time1['diff'].cumsum()
            temp_time1['diff_cum'] = temp_time1['diff_cum'] /max(temp_time1['diff_cum'])
            temp_time1['midnight'] = temp_time1[self.timestamp_key].apply(lambda x:  x.replace(hour=0, minute=0, second=0, microsecond=0))
            temp_time1['times3'] = (temp_time1[self.timestamp_key] - temp_time1['midnight']).dt.seconds / 86400
            temp_time1['times4'] = temp_time1[self.timestamp_key].apply(lambda x:  x.weekday() / 7)
            self.log[self.timestamp_key] = temp_time1['diff']
            del temp_time1

        if outcome:
            self.log.replace(to_replace={self.label_name_key: {self.label_pos_val: '1', self.label_neg_val: '0'}}, inplace=True)
            

    def _set_log_keys_and_ths(self):
        if self.log_ext == LogExt.XES:
            self.case_name_key = 'case:concept:name'
            self.act_name_key = 'concept:name'
            self.res_name_key = 'org:resource'
            self.timestamp_key = 'time:timestamp'

        if self.log_name == LogName.UBE:
            self.evaluation_prefix_start = 3
            self.evaluation_prefix_end = 7
            self.compliance_th = 1.00
        elif self.log_name == LogName.HELPDESK:
            median =5
            self.evaluation_prefix_start = median//2 - 1
            self.evaluation_prefix_end = median//2 + 2
            self.compliance_th = 1.00
        elif self.log_name == LogName.BPIC11:
            median = 92
            self.evaluation_prefix_start = median//2 - 2
            self.evaluation_prefix_end = median//2 + 2
            self.compliance_th = 1.00
        elif self.log_name == LogName.BPIC12:
            median = 32
            self.evaluation_prefix_start = median//2 - 2
            self.evaluation_prefix_end = median//2 + 2
            self.compliance_th = 1.00
        elif self.log_name == LogName.BPIC13:
            median = 4
            self.evaluation_prefix_start = median//2 - 1
            self.evaluation_prefix_end = median//2 + 2
            self.compliance_th = 1.00
        elif self.log_name == LogName.BPIC17:
            median = 54
            self.evaluation_prefix_start = median//2 - 2
            self.evaluation_prefix_end = median//2 + 2
            self.compliance_th = 1.00
        elif self.log_name == LogName.BPIC13_In:
            median = 6
            self.evaluation_prefix_start = median//2 - 2
            self.evaluation_prefix_end = median//2 + 2
            self.compliance_th = 1.00
        elif self.log_name == LogName.BPIC13_CP:
            median = 3
            self.evaluation_prefix_start = median//2
            self.evaluation_prefix_end = median//2 + 2
            self.compliance_th = 1.00
        elif self.log_name == LogName.SEPSIS:
            median = 13
            self.evaluation_prefix_start = median//2 - 2
            self.evaluation_prefix_end = median//2 + 2
            self.compliance_th = 1.00
        elif self.log_name == LogName.BPIC15_1:
            median = 15
            self.evaluation_prefix_start = median//2 - 2
            self.evaluation_prefix_end = median//2 + 2
            self.compliance_th = 1.00
        elif self.log_name == LogName.BPIC15_1:
            median = 44
            self.evaluation_prefix_start = median//2 - 2
            self.evaluation_prefix_end = median//2 + 2
            self.compliance_th = 1.00
        elif self.log_name == LogName.BPIC15_2:
            median = 54
            self.evaluation_prefix_start = median//2 - 2
            self.evaluation_prefix_end = median//2 + 2
            self.compliance_th = 1.00
        elif self.log_name == LogName.BPIC15_3:
            median = 42
            self.evaluation_prefix_start = median//2 - 2
            self.evaluation_prefix_end = median//2 + 2
            self.compliance_th = 1.00
        elif self.log_name == LogName.BPIC15_4:
            median = 44
            self.evaluation_prefix_start = median//2 - 2
            self.evaluation_prefix_end = median//2 + 2
            self.compliance_th = 1.00
        elif self.log_name == LogName.BPIC15_5:
            median = 50
            self.evaluation_prefix_start = median//2 - 2
            self.evaluation_prefix_end = median//2 + 2
            self.compliance_th = 1.00
        elif self.log_name == LogName.BPIC20_1:
            median = 8
            self.evaluation_prefix_start = median // 2 - 2
            self.evaluation_prefix_end = median // 2 + 2
            self.compliance_th = 1.00
        elif self.log_name == LogName.BPIC20_2:
            median = 11
            self.evaluation_prefix_start = median // 2 - 2
            self.evaluation_prefix_end = median // 2 + 2
            self.compliance_th = 1.00
        elif self.log_name == LogName.BPIC20_3:
            median = 5
            self.evaluation_prefix_start = median // 2 - 2
            self.evaluation_prefix_end = median // 2 + 2
            self.compliance_th = 1.00
        elif self.log_name == LogName.BPIC20_4:
            median = 5
            self.evaluation_prefix_start = median // 2 - 2
            self.evaluation_prefix_end = median // 2 + 2
            self.compliance_th = 1.00
        elif self.log_name == LogName.BPIC20_5:
            median = 10
            self.evaluation_prefix_start = median // 2 - 2
            self.evaluation_prefix_end = median // 2 + 2
            self.compliance_th = 1.00
        else:
            raise RuntimeError(f"No settings defined for log: {self.log_name.value}.")