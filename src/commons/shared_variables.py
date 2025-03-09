"""
This file was created in order to bring
common variables and functions into one file to make
code more clear
"""
from pathlib import Path
from src.ProbDeclmonitor.probDeclPredictor import AggregationMethod
aggregationMethod= AggregationMethod.SUM
test_log="_D5_1"
ascii_offset = 161
beam_size = 3
th_reduction_factor = 1
One_hot_encoding=False
combined_Act_res = False
useProb_reduction = False
use_modulator = False
use_train_test_logs = False
BK_end = False
declare_BK = False
root_folder = Path.cwd() #/ 'implementation_real_logs'
data_folder = root_folder / 'data'
input_folder = data_folder / 'input'
output_folder = data_folder / 'output'
newlog_folder = output_folder / 'predictedlogs'

declare_folder_old = input_folder / 'declare_models'
xes_log_folder =  input_folder / 'log_xes'
log_folder = input_folder / 'logs'
pn_folder = input_folder / 'petrinets'
declare_folder = input_folder  / 'declare_models'

epochs = 100
folds = 3
train_ratio = 0.8
variant_split = 0.9
validation_split = 0.2


log_list = [
   'helpdesk.xes',
    #'BPI2012.xes' ,
    #'BPI2013.xes',
    #'BPI2017.xes'
]

synthetic_log_list = [
    'BPI2013_In_testnew.xes',
    'helpdesk_test.xes',
    'BPI2013_CP_test.xes',
'DomesticDeclarations_test.xes',
    'BPI2012_test.xes',
    'PrepaidTravelCost_test.xes',
    'PermitLog_test.xes',
    'RequestForPayment_test.xes',
    'InternationalDeclarations_test.xes',
    'Sepsis_cases_test.xes',
    'Road_traffic_test.xes'
]

method_marker = {'baseline': 'x', 'beamsearch (beam size = 3)': '1', 'beamsearch (beam size = 5)': '.', 'beamsearch (beam size = 10)': '',
                 'beamsearch with BK (beam size = 10)': '+', 'beamsearch with BK (beam size = 5)':'*', 'frequency':'+', 'beamsearch with BK (beam size = 3)': '.'}
method_color = {'baseline': 'mediumpurple', 'beamsearch (beam size = 3)': 'deepskyblue', 'beamsearch (beam size = 5)': 'orange',
                'beamsearch (beam size = 10)': 'purple', 'beamsearch with BK (beam size = 10)': 'brown',  'beamsearch with BK (beam size = 5)':'red',
                'beamsearch with BK (beam size = 3)': 'green'}
