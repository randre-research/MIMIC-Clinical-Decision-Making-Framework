# Constants for paths
# MIMIC_BASE = r'D:\Users\orosh\Documents\Research\Datasets\mimic-iv-ext-cdm-1.1-from-iv'
# EXPERIMENTS_BASE = r'D:\Users\orosh\Documents\Research\Datasets\mimic-iv-ext-cdm-1.1-from-iv\logs\analysis'
# OUTPUT_BASE = r'D:\Users\orosh\Documents\Research\Datasets\mimic-iv-ext-cdm-1.1-from-iv\logs\analysis\output'

MIMIC_BASE = '/container/data'
EXPERIMENTS_BASE = '/container/data/logs/analysis'
OUTPUT_BASE = '/container/data/logs/analysis/output'

DIAG = "Gracious Diagnosis"

import os
# import paramiko
import seaborn as sns
import matplotlib.pyplot as plt
# from paramiko import Ed25519Key
from datetime import datetime
import shutil
from cmcrameri import cm
# import palettable
from matplotlib.colors import ListedColormap

MODELS = [
    "Llama-3.2-1B-Instruct-exl2-4.0bpw", 
    "Llama-3.2-1B-Instruct-exl2-4.0bpw_stella_en_400M_v5",
    "Llama-3.1-70B-Instruct-exl2-4.0bpw",
    ]

prettify_model_name = {
#     "Llama-2-70B-chat-GPTQ": "Llama 2 Chat",
#     "Llama2-70B-OASST-SFT-v10-GPTQ": "OASST",
#     "WizardLM-70B-V1.0-GPTQ": "WizardLM",
#     "axiong_PMC_LLaMA_13B": "PMC Llama",
#     "ClinicalCamel-70B-GPTQ": "Clinical Camel",
#     "Meditron-70B-GPTQ": "Meditron",
    "MIMIC Doctors": "MIMIC Doctors",
    "Llama-3.2-1B-Instruct-exl2-4.0bpw": "Llama-3.2-1B-Instruct-exl2-4.0bpw",
    "Llama-3.2-1B-Instruct-exl2-4.0bpw_stella_en_400M_v5": "Llama-3.2-1B-Instruct-exl2-4.0bpw_stella_en_400M_v5",
    "Llama-3.1-70B-Instruct-exl2-4.0bpw": "Llama-3.1-70B-Instruct-exl2-4.0bpw",
}

color_map = {
    # "Llama 2 Chat": "#0077B6",
    # "OASST": "#00B4D8",
    # "WizardLM": "#90E0EF",
    # "Clinical Camel" : "#EC9898",
    # "Meditron" : "#F97F77",
    "Doctors": "#4c956c",
    "MIMIC Doctors": "#2C6E49",
    "Llama-3.2-1B-Instruct-exl2-4.0bpw": "#0077B6",
    "Llama-3.2-1B-Instruct-exl2-4.0bpw_stella_en_400M_v5": "#00B4D8",
    "Llama-3.1-70B-Instruct-exl2-4.0bpw": "#90E0EF",

    "Appendicitis": "#B0A0BA",
    "Cholecystitis": "#B392AC",
    "Diverticulitis": "#906487",
    "Pancreatitis" : "#735D78",

    "Mean" : "#e56b6f"
}

intensity=0.9

def download_most_recent(server_ip, username, private_key_path, base_folder, pathology, agent, model, addendum, destination_folder, folder_position = 0):
    # if server_ip:
    #     private_key = Ed25519Key.from_private_key_file(private_key_path)
    #     client = paramiko.SSHClient()  
    #     client.load_system_host_keys()  
    #     client.set_missing_host_key_policy(paramiko.WarningPolicy)
    #     client.connect(server_ip, username=username, pkey=private_key)
    #     sftp = client.open_sftp()
    #     listdir = sftp.listdir
    #     copy = sftp.get
    # else:
    #     listdir = os.listdir
    #     copy = shutil.copy
        
    listdir = os.listdir
    copy = shutil.copy

    all_folder_files = listdir(base_folder)

    base_folder = base_folder.rstrip('/')

    folder_date_mapping = {}
    if not addendum:
        addendum = tuple(str(i) for i in range(10))
    for item in all_folder_files:
        if item.startswith(f'{pathology}_{agent}_{model}_') and item.endswith(addendum):
            n_underscore = addendum.count("_")
            if n_underscore == 0:
                date_time_str = '_'.join(item.split('_')[-2:])
            else:
                date_time_str = '_'.join(item.split('_')[-(2+n_underscore):-n_underscore])
            date_time_obj = datetime.strptime(date_time_str, '%d-%m-%Y_%H:%M:%S')
            folder_date_mapping[item] = date_time_obj

    latest_folder = sorted(folder_date_mapping, key=folder_date_mapping.get, reverse=True)[folder_position]

    
    files = listdir(os.path.join(base_folder, latest_folder))
    for file in files:
        if '_results' in file:
            remote_file_path = os.path.join(base_folder, latest_folder, file)
            local_file_path = os.path.join(destination_folder, file)
            copy(remote_file_path, local_file_path)
    
    # if server_ip:
    #     client.close()

def download_most_recent_FI(server_ip, username, private_key_path, base_folder, pathology, model, addendum, destination_folder, folder_position=0):
    # if server_ip:
    #     private_key = Ed25519Key.from_private_key_file(private_key_path)
    #     client = paramiko.SSHClient()  
    #     client.load_system_host_keys()  
    #     client.set_missing_host_key_policy(paramiko.WarningPolicy)
    #     client.connect(server_ip, username=username, pkey=private_key)
    #     sftp = client.open_sftp()
    #     listdir = sftp.listdir
    #     copy = sftp.get
    # else:
    #     listdir = os.listdir
    #     copy = shutil.copy

    listdir = os.listdir
    copy = shutil.copy

    base_folder = base_folder.rstrip('/')

    all_folder_files = listdir(base_folder)
    folder_date_mapping = {}
    for item in all_folder_files:
        if item.startswith(f'{pathology}_{model}_') and item.endswith(f'_FULL_INFO{addendum}'):
            n_underscore = addendum.count("_")
            date_time_str = '_'.join(item.split('_')[-(4+n_underscore):-(2+n_underscore)])
            date_time_obj = datetime.strptime(date_time_str, '%d-%m-%Y_%H:%M:%S')
            folder_date_mapping[item] = date_time_obj

    latest_folder = sorted(folder_date_mapping, key=folder_date_mapping.get, reverse=True)[folder_position]
    
    files = listdir(os.path.join(base_folder, latest_folder))
    for file in files:
        if '_results' in file:
            remote_file_path = os.path.join(base_folder, latest_folder, file)
            local_file_path = os.path.join(destination_folder, file)
            copy(remote_file_path, local_file_path)
    
    # if server_ip:
    #     client.close()

from utils.logging import read_from_pickle_file

from os.path import join
from dataset.utils import load_hadm_from_file

from evaluators.appendicitis_evaluator import AppendicitisEvaluator
from evaluators.cholecystitis_evaluator import CholecystitisEvaluator
from evaluators.diverticulitis_evaluator import DiverticulitisEvaluator
from evaluators.pancreatitis_evaluator import PancreatitisEvaluator
from run import load_evaluator
from utils.nlp import latex_escape

# Check new evaluation strategy
def load_evaluator(pathology):
    # Load desired evaluator
    if pathology == "appendicitis":
        evaluator = AppendicitisEvaluator()
    elif pathology == "cholecystitis":
        evaluator = CholecystitisEvaluator()
    elif pathology == "diverticulitis":
        evaluator = DiverticulitisEvaluator()
    elif pathology == "pancreatitis":
        evaluator = PancreatitisEvaluator()
    else:
        raise NotImplementedError
    return evaluator

import glob

import numpy as np
import pickle

def calculate_average(evals, field, pathology):
    average = 0
    for patient in evals.keys():
        if field not in evals[patient]['scores']:
            print(f"{field} not in {patient}")
        average += evals[patient]['scores'][field]

    average /= len(evals)
    #print(f'{pathology}: {average:0.02} (n={len(evals)})'.rjust(30))
    return average, len(evals)

def calculate_average_multipatho(evals, field, main_pathology):
    average = 0
    counts = {}
    for patho in ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]:
        counts[patho] = 0
    for patient in evals.keys():
        if field not in evals[patient][main_pathology]['scores']:
            print(f"{field} not in {patient}")
        counts[main_pathology] += evals[patient][main_pathology]['scores'][field]
        for patho in ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]:
            if patho != main_pathology:
                counts[patho] += evals[patient][patho]['scores'][field]
        
    # counts[main_pathology] /= len(evals)
    #print(f'{pathology}: {average:0.02} (n={len(evals)})'.rjust(30))
    return counts, len(evals)

def calculate_percentages(evals, field):
    max_field = field
    if "Late" in field:
        max_field = max_field.replace("Late ", "")
    max_field = max_field.replace(" Percentage", "")
    for patient in evals.keys():
        evals[patient]['scores'][field] = evals[patient]['scores'][field[:-len(" Percentage")]] / evals[patient]['max_scores'][max_field]
    return evals

def count_unnecessary(evals, field):
    for patient in evals.keys():
        evals[patient]['scores'][field] = len(evals[patient]['answers'][field])
    return evals


def print_results(difficulty, all_evals):
    # id_difficulty = pickle.load(open('/home/paulhager/Projects/data/mimic-iv/hosp/id_difficulty.pkl', 'rb'))
    id_difficulty = pickle.load(open(os.path.join(MIMIC_BASE, 'id_difficulty.pkl'), 'rb'))
    avg_scores = {}
    avg_samples = {}
    print(f'@@@ {difficulty} @@@'.center(30))
    print()
    if difficulty in ['easy', 'secondary', 'hard']:
        all_evals_diff = {}
        for patho in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']:
            all_evals_diff[patho] = {}
            for _id in all_evals[patho].keys():
                if _id in id_difficulty[patho][difficulty]:
                    all_evals_diff[patho][_id] = all_evals[patho][_id]
    else:
        all_evals_diff = all_evals
    for field in ['Diagnosis', 'Custom Parsings', 'Rounds', 'Physical Examination', 'Unnecessary Laboratory Tests', 'Unnecessary Imaging']:
        avg_scores[field] = {}
        avg_samples[field] = {}
        print(f'### {field} ###'.center(30))
        for patho in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']:
            if field in ['Unnecessary Laboratory Tests', 'Unnecessary Imaging']:
                all_evals_diff[patho] = count_unnecessary(all_evals_diff[patho], field)

            avg, n = calculate_average(all_evals_diff[patho], field, patho)
            print(f'{patho}: {avg:0.02} (n={n})'.rjust(30))

            avg_scores[field][patho] = avg
            avg_samples[field][patho] = n
        print(f'AVERAGE: {np.mean(list(avg_scores[field].values())):0.2} (n={round(np.mean(list(avg_samples[field].values())))})'.rjust(30))
        print()

# Check new evaluation strategy
def load_evaluator(pathology):
    # Load desired evaluator
    if pathology == "appendicitis":
        evaluator = AppendicitisEvaluator()
    elif pathology == "cholecystitis":
        evaluator = CholecystitisEvaluator()
    elif pathology == "diverticulitis":
        evaluator = DiverticulitisEvaluator()
    elif pathology == "pancreatitis":
        evaluator = PancreatitisEvaluator()
    else:
        raise NotImplementedError
    return evaluator

import pandas as pd

def generate_latex_tables(model_scores, experiment_name):
    model_dicts = list(model_scores.values())
    model_names = list(model_scores.keys())
    diseases = ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']
    categories = list(model_dicts[0].keys())

    model_names = [prettify_model_name[name] for name in model_names]
    category_names  = [category.replace(" ", " \\\\ ") for category in categories]

    # for mean performance
    data = []
    for model_dict, model_name in zip(model_dicts, model_names):
        row = [model_name] + [sum(model_dict[category].values())/len(diseases) for category in categories]
        data.append(row)
    df = pd.DataFrame(data, columns=["Model"] + [f"\\thead{{{category}}}" for category in category_names])
    styler = df.style
    styler.format({f"\\thead{{{category}}}": "{:.2f}" for category in category_names}).hide(level=0, axis=0)
    print("\\begin{table}[ht]")
    print("\\begin{adjustwidth}{-1in}{-1in}")
    print("\\centering")
    if experiment_name:
        experiment_name = f" - {experiment_name}"
    print(f"\\caption{{Mean model performance{experiment_name}}}")
    print(styler.to_latex(column_format='l' + 'c'*len(categories), hrules=True))
    print("\\end{adjustwidth}")
    print("\\end{table}\n")

    for disease in diseases:
        data = []
        for model_dict, model_name in zip(model_dicts, model_names):
            row = [model_name] + [model_dict[category][disease] for category in categories]
            data.append(row)
        df = pd.DataFrame(data, columns=["Model"] + [f"\\thead{{{category}}}" for category in category_names])
        styler = df.style
        styler.format({f"\\thead{{{category}}}": "{:.2f}" for category in category_names}).hide(level=0, axis=0)
        print("\\begin{table}[ht]")
        print("\\begin{adjustwidth}{-1in}{-1in}")
        print("\\centering")

        print(f"\\caption{{Model performance on {disease.capitalize()}{experiment_name}}}")
        print(styler.to_latex(column_format='l' + 'c'*len(categories), hrules=True))
        print("\\end{adjustwidth}")
        print("\\end{table}\n")

def generate_latex_tables_full_info(model_scores, experiment_name, bold_max=True, underline_second_max=True):
    diseases = ['Mean', 'Appendicitis', 'Cholecystitis', 'Diverticulitis', 'Pancreatitis']
    models = list(model_scores.keys())

    data = []
    for model_name in models:
        row = []
        for disease in diseases:
            row.append(model_scores[model_name][disease])
        data.append(row)

    df = pd.DataFrame(data, columns=diseases, index=models)

    if bold_max or underline_second_max:
        def format_number(number, column, max_number, second_max_number):
            if bold_max and number == max_number:
                return "\\textbf{%.2f}" % number
            elif underline_second_max and number == second_max_number:
                return "\\underline{%.2f}" % number
            else:
                return "%.2f" % number

        for column in df.columns:
            max_number = df[column].max()
            second_max_number = df[df[column] != max_number][column].max() if underline_second_max else None
            df[column] = df[column].apply(lambda num: format_number(num, column, max_number, second_max_number))

    latex_content = df.to_latex(escape=False, multirow=True, column_format="|c"* (len(diseases) + 1) + "|")

    for disease in diseases:
        latex_content = latex_content.replace(f'\\multicolumn{{2}}{{r}}{{{disease}}}', f'\\multicolumn{{2}}{{|c|}}{{\\textbf{{{disease}}}}}')

    print("\\begin{table}[ht]")
    print("\\centering")
    if experiment_name:
        experiment_name = f" - {latex_escape(experiment_name)}"
    print(f"\\caption{{Diagnostic Accuracy (\%) with Full Information{experiment_name}}}")
    print(latex_content)
    print("\\end{table}\n")

def load_scores(experiments, difficulty="first_diag", fields=['Diagnosis'], models = ["Llama-2-70B-chat-GPTQ", "Llama2-70B-OASST-SFT-v10-GPTQ", "WizardLM-70B-V1.0-GPTQ"], pathos = ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis'], f1=False):
    # id_difficulty = pickle.load(open('/home/paulhager/Projects/data/mimic-iv/hosp/id_difficulty.pkl', 'rb'))
    id_difficulty = pickle.load(open(os.path.join(MIMIC_BASE, 'id_difficulty.pkl'), 'rb'))

    experiment_results = {}
    experiment_evals = {}
    experiment_scores = {}
    experiment_retrievals = {} #RAG
    for experiment in experiments:
        model_results = {}
        model_evals = {}
        model_scores = {}
        model_retrievals = {} #RAG
        for model in models:
            # m_results = pickle.load(open(f'/home/paulhager/Projects/ClinicalBenchmark/logs/SOTA/{experiment}/{model}_results{"_f1" if f1 else ""}.pkl', 'rb'))
            m_results = pickle.load(open(os.path.join(EXPERIMENTS_BASE, experiment, f'{model}_results{"_f1" if f1 else ""}.pkl'), 'rb'))
            # m_evals = pickle.load(open(f'/home/paulhager/Projects/ClinicalBenchmark/logs/SOTA/{experiment}/{model}_evals{"_f1" if f1 else ""}.pkl', 'rb'))
            m_evals = pickle.load(open(os.path.join(EXPERIMENTS_BASE, experiment, f'{model}_evals{"_f1" if f1 else ""}.pkl'), 'rb'))         
    
            m_retrievals = pickle.load(open(os.path.join(EXPERIMENTS_BASE, experiment, f'{model}_retrievals{"_f1" if f1 else ""}.pkl'), 'rb')) #RAG   
            model_results[model] = m_results
            model_evals[model] = m_evals
            model_retrievals[model] = m_retrievals #RAG

        for model in models:
            all_evals = {}
            all_results = {}
            all_retrievals = {} #RAG
            for patho in pathos:
                all_evals[patho] = model_evals[model][patho]
                all_results[patho] = model_results[model][patho]
                all_retrievals[patho] = model_retrievals[model][patho]
                
                # Subset to only desired ids
                selected_evals = {}
                selected_results = {}
                #RAG: Oops killed the ids in evaluate_fi_custom.py
                for _id in id_difficulty[patho][difficulty]:
                    if _id not in all_evals[patho]:
                        # Manually tested and all were correct
                        if _id == 21285450:
                            selected_evals[_id] = {'scores': {'Diagnosis': 1.0, "Gracious Diagnosis": 1.0, "Action Parsing": 0, "Treatment Parsing": 0, "Diagnosis Parsing": 0, 'Invalid Tools': 0}}
                        else:
                            print(f"For experiment {experiment} and model {model}, {_id} not in {patho}")
                        continue
                    selected_evals[_id] = all_evals[patho][_id]
                    selected_results[_id] = all_results[patho][_id]

                all_evals[patho] = selected_evals
                all_results[patho] = selected_results
            
            model_evals[model] = all_evals
            model_results[model] = all_results
            model_retrievals[model] = all_retrievals #RAG
            avg_scores = {}
            avg_samples = {}

            field = 'Diagnosis'
            avg_scores[field] = {}
            avg_samples[field] = {}
            for field in fields:
                avg_scores[field] = {}
                avg_samples[field] = {}
                for patho in pathos:
                    if field in ['Unnecessary Laboratory Tests', 'Unnecessary Imaging']:
                        all_evals[patho] = count_unnecessary(all_evals[patho], field)
                    if f1:
                        avg, n = calculate_average_multipatho(all_evals[patho], field, patho)
                    else:
                        avg, n = calculate_average(all_evals[patho], field, patho)

                    avg_scores[field][patho] = avg
                    avg_samples[field][patho] = n
            model_scores[model] = avg_scores
        experiment_results[experiment] = model_results
        experiment_evals[experiment] = model_evals
        experiment_scores[experiment] = model_scores
        experiment_retrievals[experiment] = model_retrievals #RAG
    # return experiment_results, experiment_evals, experiment_scores
    return experiment_results, experiment_evals, experiment_scores, experiment_retrievals #RAG

def check_diagnoses_orig_dr_eval(ids, id_difficulty):
    for patho in ['appendicitis', 'cholecystitis', 'pancreatitis', 'diverticulitis']:
        id_difficulty[patho]['original_dr_eval'] = []
    for i in ids:
        for patho in ['appendicitis', 'cholecystitis', 'pancreatitis', 'diverticulitis']:
            if i in id_difficulty[patho]['first_diag']:
                print(f"{i} is in {patho}")
                id_difficulty[patho]['original_dr_eval'].append(i)
    for patho in "gastritis", "urinary_tract_infection", "hernia", "esophageal_reflux":
        id_difficulty[patho]['original_dr_eval'] = id_difficulty[patho]['dr_eval']
    return id_difficulty

def check_diagnoses(ids, id_difficulty):
    for i in ids:
        for patho in ['appendicitis', 'cholecystitis', 'pancreatitis', 'diverticulitis']:
            if i in id_difficulty[patho]['first_diag']:
                print(f"{i} is in {patho}")

def print_diagnoses(difficulty):
    # id_difficulty = pickle.load(open('/home/paulhager/Projects/data/mimic-iv/hosp/id_difficulty.pkl', 'rb'))
    id_difficulty = pickle.load(open(os.path.join(MIMIC_BASE, 'id_difficulty.pkl'), 'rb'))
    diags = {}
    for patho in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']:
        for _id in id_difficulty[patho][difficulty]:
            diags[_id] = patho

    diag_keys = list(diags.keys())
    diag_keys.sort()
    for key in diag_keys:
        print(f"{key} {diags[key]}")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# experiments = ['FI_PLI', 'FI_PLI_FEWSHOT']
# experiments = ["FULL_INFO_PLI_N_BIN_BINABNORMAL"]
experiments = ["FULL_INFO_PLI_N_ONLYABNORMAL_BIN_BINABNORMAL_VANILLA_PROBS"]
# models = ["Llama-2-70B-chat-GPTQ", "Llama2-70B-OASST-SFT-v10-GPTQ", "WizardLM-70B-V1.0-GPTQ"]
# models = ["Llama-3.2-1B-Instruct-exl2-4.0bpw"]
models = MODELS
fields = [DIAG]

# experiment_results, experiment_evals, experiment_scores = load_scores(experiments, models=models, fields=fields)
experiment_results, experiment_evals, experiment_scores, experiment_retrievals = load_scores(experiments, models=models, fields=fields) #RAG

data = []
for experiment in experiment_scores.keys():
    model_scores = experiment_scores[experiment]
    for model in model_scores.keys():
        mean_diagnosis = np.mean([model_scores[model][DIAG][patho] for patho in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']])
        for patho in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']:
            data.append([experiment, model, patho.capitalize(), model_scores[model][DIAG][patho]])
        data.append([experiment, model, 'Mean', mean_diagnosis])

df = pd.DataFrame(data, columns=['Experiment', 'Model', 'Pathology', 'Diagnostic Accuracy'])
df['Diagnostic Accuracy'] *= 100
df['Model'] = df['Model'].apply(lambda x: prettify_model_name[x])

# Extract best experiment of each model based on mean diagnostic accuracy
best_experiments = {}
for model in models:
    model = prettify_model_name[model]
    best_experiments[model] = df[(df['Model'] == model) & (df['Pathology']=='Mean')].sort_values(by=['Diagnostic Accuracy'], ascending=False).iloc[0]['Experiment']

# For each model, remove rows that are not the best experiment
df = df[df.apply(lambda x: x['Experiment'] == best_experiments[x['Model']], axis=1)]

# For experiment_retrievals, extract best experiment of each model based on mean diagnostic accuracy
best_experiments_retrievals = {}
for model in models:
    model = prettify_model_name[model]
    best_experiments_retrievals[model] = experiment_retrievals[best_experiments[model]][model] #RAG

sns.set(style="whitegrid", font_scale=1.4)

# Creating the bar plot
plt.figure(figsize=(12, 6))
bar_plot = sns.barplot(x='Pathology', y='Diagnostic Accuracy', hue='Model', data=df, palette=color_map, saturation=intensity)

# Calculate the number of unique pathologies and models
num_pathologies = len(df['Pathology'].unique())
num_models = len(df['Model'].unique())

# Adding the scores above the bars
for p in bar_plot.patches:
    if p.get_height() > 0:
        bar_plot.annotate(format(p.get_height(), '.0f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=14,)


# Additional plot formatting
plt.title('')
plt.ylabel('Diagnostic Accuracy (%)')
plt.xlabel('')
plt.ylim(0, 100)
plt.legend(bbox_to_anchor=(1.0, 1.13),  ncol=len(model_scores.keys()), frameon=False, fontsize=14)
#get current date and time
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
# plt.savefig(os.path.join(OUTPUT_BASE, f"DiagnosticAccuraciesFI_ED_Fig1_{dt_string}.eps"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_BASE, f"DiagnosticAccuraciesFI_ED_Fig1_{dt_string}.png"), dpi=300, bbox_inches='tight')
plt.show()



# RAG PART: Per Model, Create a 2x2 grid, one cell per pathology, each cell contains a 2x2 grid, one cell per document, each cell contains a barplot with for x the number of chunks of the document and for y the relative (relative to the pathology) count of said chunk.

#For general heatmap
import pandas as pd
#Create pandas dataframe with each document name as a column and each pathology as a row
#Each cell contains the relative count of chunks of the document for the pathology

for model in models:
    df_heatmap_dict = {}

    model_is_rag = False

    model = prettify_model_name[model]
    model_retrievals = best_experiments_retrievals[model]

    # #Only one image per model, 2x2 grid of pathology, each pathology has a 2x2 grid of documents
    # fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    # fig.suptitle(f'{model}')

    for i, pathology in enumerate(['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']):
    # for pathology in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']:
        df_heatmap_dict[pathology] = {}

        pathology_retrievals = model_retrievals[pathology]
        #Example of where to find the count per chunk: all_retrievals[patho][chunk["document_reference"]][chunk["page_number"]][chunk["order_in_document"]]["count"] += 1

        #Count total number of chunks per pathology
        total_chunks = 0
        for document in pathology_retrievals.keys():
            for page in pathology_retrievals[document].keys():
                for order in pathology_retrievals[document][page]:
                    total_chunks += pathology_retrievals[document][page][order]["count"]

        if total_chunks > 0:
            model_is_rag = True

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        # fig.suptitle(f'{model} - {pathology.capitalize()}')
        fig.suptitle(f'{pathology.capitalize()}')
        #Alphabetically order keys in pathology_retrievals
        pathology_retrievals = {key: pathology_retrievals[key] for key in sorted(pathology_retrievals.keys())}

        for j, document in enumerate(pathology_retrievals.keys()):
            document_retrievals = pathology_retrievals[document]
            ax = axs[j//2, j%2]
            #get document name
            document_name = document.split("/")[-1]
            #if length>25, end with ...
            if len(document_name) > 20:
                document_name = document_name[:25] + "..."

            df_heatmap_dict[pathology][document_name] = 0

            #always make the y axis go from 0 to 1
            # ax.set_ylim(0, 1)
            ax.set_ylim(0, .3)
            #exponential scale for y axis (opposite of log)
            # ax.set_yscale('symlog')
            #remove y axis if not the first column
            if j%2 != 0:
                ax.set_yticks([])

            #total counts per document
            doc_counts = 0

            for page in document_retrievals:
                #add up the counts of all chunks in the page
                counts = 0
                for order in document_retrievals[page]:
                    counts += document_retrievals[page][order]["count"]

                #Create one bar per page, make bar the color of the model
                if counts/total_chunks > .05:
                    ax.bar(page, counts/total_chunks, label=str(page), color=color_map[model])
                    #make sure the label (page) appears above the bar
                    ax.text(page, counts/total_chunks, str(page), ha='center', va='bottom')
                else:
                    ax.bar(page, counts/total_chunks, color=color_map[model])

                doc_counts += counts

            df_heatmap_dict[pathology][document_name] = doc_counts/total_chunks

            #convert doc_count to a percentage string with one decimal
            doc_counts = f'{doc_counts/total_chunks*100:.1f}%'

            #set title of the document
            ax.set_title(f'({doc_counts}) {document_name}')

            #reduce the empty space between the bars of the barplot
            ax.margins(x=0.01)

        if model_is_rag:
            # plt.savefig(os.path.join(OUTPUT_BASE, f"Retrievals_{model}_{pathology}_{dt_string}.eps"), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(OUTPUT_BASE, f"Retrievals_{model}_{pathology}_{dt_string}.png"), dpi=300, bbox_inches='tight')
        #close the figure
        plt.close()

    if model_is_rag:
        #Combine all last four plots into one big plot
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f'{model}')
        #load the images
        for i, pathology in enumerate(['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']):
            ax = axs[i//2, i%2]
            #remove margins
            ax.margins(x=0)
            #load the image
            img = plt.imread(os.path.join(OUTPUT_BASE, f"Retrievals_{model}_{pathology}_{dt_string}.png"))
            ax.imshow(img)
            ax.axis('off')

        # plt.savefig(os.path.join(OUTPUT_BASE, f"Retrievals_{model}_all_{dt_string}.eps"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_BASE, f"Retrievals_{model}_all_{dt_string}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        #delete the individual images
        for pathology in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']:
            os.remove(os.path.join(OUTPUT_BASE, f"Retrievals_{model}_{pathology}_{dt_string}.png"))

    if model_is_rag:
        #Create a heatmap of the relative counts of chunks per document per pathology per model
        df_heatmap_count = pd.DataFrame(df_heatmap_dict)
        #Transpose the dataframe
        df_heatmap_count = df_heatmap_count.T
        #save df_heatmap_count to csv
        df_heatmap_count.to_csv(os.path.join(OUTPUT_BASE, f"Retrievals_{model}_heatmap_{dt_string}.csv"))
        #create new plot
        plt.figure(figsize=(12, 12))
        sns.heatmap(df_heatmap_count, cmap='viridis', annot=True, cbar_kws={'label': 'Relative Retrieval Per Pathology'})
        #make the names be diagonal
        plt.xticks(rotation=45)
        #make the names be horizontal
        plt.yticks(rotation=0)
        plt.title(f'{model}\nRelative Document Retrieval Per Pathology\n')
        plt.savefig(os.path.join(OUTPUT_BASE, f"Retrievals_{model}_heatmap_{dt_string}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        #Compute Z-scores per pathology
        df_heatmap_z = df_heatmap_count.copy()
        df_heatmap_z = df_heatmap_z.T
        for pathology in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']:
            df_heatmap_z[pathology] = (df_heatmap_z[pathology] - df_heatmap_z[pathology].mean())/df_heatmap_z[pathology].std()
        df_heatmap_z = df_heatmap_z.T

        #save df_heatmap_z to csv
        df_heatmap_z.to_csv(os.path.join(OUTPUT_BASE, f"Retrievals_{model}_heatmap_zscores_{dt_string}.csv"))
        #create new plot
        plt.figure(figsize=(12, 12))
        sns.heatmap(df_heatmap_z, cmap='coolwarm', annot=True, cbar_kws={'label': 'Z-Score Relative Retrieval Per Pathology'})
        #make the names be diagonal
        plt.xticks(rotation=45)
        #make the names be horizontal
        plt.yticks(rotation=0)
        plt.title(f'{model}\nZ-Score Relative Document Retrieval Per Pathology\n')
        plt.savefig(os.path.join(OUTPUT_BASE, f"Retrievals_{model}_heatmap_zscores_{dt_string}.png"), dpi=300, bbox_inches='tight')
        plt.close()