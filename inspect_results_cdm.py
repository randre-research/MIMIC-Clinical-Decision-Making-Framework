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

import pandas as pd
import numpy as np

MODELS = [
    "Llama-3.2-1B-Instruct-exl2-4.0bpw", 
    "Llama-3.2-1B-Instruct-exl2-4.0bpw_stella_en_400M_v5",
    # "Llama-3.2-1B-Instruct-exl2-4.0bpw_stella_en_400M_v5_chunkr",

    # "Llama-3.1-70B-Instruct-exl2-4.0bpw",
    # "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_400M_v5",
    # "Llama-3.1-70B-Instruct-exl2-2.5bpw_stella_en_1.5B_v5",
    # "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_400M_v5_k12_8k",
    # "Llama-3.1-70B-Instruct-exl2-2.5bpw_stella_en_1.5B_v5_k12_8k",
    ]

prettify_model_name = {
#     "Llama-2-70B-chat-GPTQ": "Llama 2 Chat",
#     "Llama2-70B-OASST-SFT-v10-GPTQ": "OASST",
#     "WizardLM-70B-V1.0-GPTQ": "WizardLM",
#     "axiong_PMC_LLaMA_13B": "PMC Llama", 
#     "ClinicalCamel-70B-GPTQ": "Clinical Camel",
#     "Meditron-70B-GPTQ": "Meditron",
    "MIMIC Doctors": "MIMIC Doctors",
    "Llama-3.2-1B-Instruct-exl2-4.0bpw": "Llama3 1B 4.0bpw",
    "Llama-3.2-1B-Instruct-exl2-4.0bpw_stella_en_400M_v5": "Llama3 1B 4.0bpw + stella5 400M",
    "Llama-3.2-1B-Instruct-exl2-4.0bpw_stella_en_400M_v5_chunkr": "Llama3 1B 4.0bpw + stella5 400M (chunkr)",
    "Llama-3.1-70B-Instruct-exl2-4.0bpw": "Llama3 70B 4.0bpw",
    "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_400M_v5": "Llama3 70B 4.0bpw + stella5 400M",
    "Llama-3.1-70B-Instruct-exl2-2.5bpw_stella_en_1.5B_v5": "Llama3 70B 2.5bpw + stella5 1.5B",
    "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_400M_v5_k12_8k": "Llama3 70B 4.0bpw + stella5 400M (TopK=12, 8k Context)",
    "Llama-3.1-70B-Instruct-exl2-2.5bpw_stella_en_1.5B_v5_k12_8k": "Llama3 70B 2.5bpw + stella5 1.5B (TopK=12, 8k Context)",
}

color_map = {
    # "Llama 2 Chat": "#0077B6",
    # "OASST": "#00B4D8",
    # "WizardLM": "#90E0EF",
    # "Clinical Camel" : "#EC9898",
    # "Meditron" : "#F97F77",
    "Doctors": "#4c956c",
    "MIMIC Doctors": "#2C6E49",
    "Llama3 1B 4.0bpw": "#0077B6",
    "Llama3 1B 4.0bpw + stella5 400M": "#00B4D8",
    "Llama3 1B 4.0bpw + stella5 400M (chunkr)": "#1ee3ab",
    "Llama3 70B 4.0bpw": "#3EAD0A",
    "Llama3 70B 4.0bpw + stella5 400M": "#9BD415",
    "Llama3 70B 2.5bpw + stella5 1.5B": "#F9F871",
    "Llama3 70B 4.0bpw + stella5 400M (TopK=12, 8k Context)": "#F97F77",
    "Llama3 70B 2.5bpw + stella5 1.5B (TopK=12, 8k Context)": "#EC9898",

    "Appendicitis": "#B0A0BA",
    "Cholecystitis": "#B392AC",
    "Diverticulitis": "#906487",
    "Pancreatitis" : "#735D78",

    "Mean" : "#e56b6f"
}


intensity=0.9

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

experiments = [
    # 'CDM_VANILLA',
    'BIN',
    ]
fields = [DIAG]
# models = ["Llama-2-70B-chat-GPTQ", "Llama2-70B-OASST-SFT-v10-GPTQ", "WizardLM-70B-V1.0-GPTQ"]
models = MODELS

experiment_results, experiment_evals, experiment_scores, experiment_retrievals = load_scores(experiments, models=models, fields=fields)

data = []
experiment = experiments[0]
model_scores = experiment_scores[experiment]
for model in model_scores.keys():
    mean_diagnosis = np.mean([model_scores[model][DIAG][patho] for patho in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']])
    for patho in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']:
        data.append([model, patho.capitalize(), model_scores[model][DIAG][patho]])
    data.append([model, 'Mean', mean_diagnosis])

df = pd.DataFrame(data, columns=['Model', 'Pathology', 'Diagnostic Accuracy'])
df['Diagnostic Accuracy'] *= 100
df['Model'] = df['Model'].apply(lambda x: prettify_model_name[x])

sns.set(style="whitegrid", font_scale=1.4)

# Creating the bar plot
plt.figure(figsize=(12, 4))
bar_plot = sns.barplot(x='Pathology', y='Diagnostic Accuracy', hue='Model', data=df, palette=color_map, saturation=intensity)

unique_patho = df['Pathology'].unique()
for i in range(len(unique_patho) - 1):
    bar_plot.axvline(x=i + 0.5, color='gray', linestyle='--', linewidth=1)

# Adding the scores above the bars
for p in bar_plot.patches:
    if p.get_height() > 0:
        bar_plot.annotate(format(p.get_height(), '.1f'), 
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
plt.legend(bbox_to_anchor=(0.8, 1.18),  ncol=len(model_scores.keys()), frameon=False, fontsize=15)

# Save the plot
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(os.path.join(OUTPUT_BASE, dt_string), exist_ok=True)
plt.savefig(os.path.join(OUTPUT_BASE, dt_string, f"DiagAccCDM_Fig3_{dt_string}.png"), dpi=300, bbox_inches='tight')
plt.show()







# --- Plot Imaging Percentage ---

import matplotlib.patches as mpatches

# Use the existing variables from the main script
experiment = experiments[0]
model_scores = experiment_scores[experiment]
model_results = experiment_results[experiment]

modality_map = {
    'CT': 'CT', 
    'Ultrasound': 'Ultrasound', 
    'Radiograph': 'Radiograph',
    'MRI': 'MRI',
    'CTU': 'CT', 
    'ERCP': 'Radiograph', 
    'Upper GI Series': 'Radiograph',
    'MRE': 'MRI', 
    'MRCP': 'MRI',
    'MRA': 'MRI',
    'EUS': 'Ultrasound',
    'Carotid ultrasound': 'Ultrasound',
    'None': 'None', 
    'PTC': 'Other', 
    'HIDA': 'Other', 
    'Drainage': 'Other', 
}

data = []
for model in model_results.keys():
    for patho in model_results[model].keys():
        for _id in model_results[model][patho].keys():
            imaging_requested = False
            for step in model_results[model][patho][_id]['intermediate_steps']:
                if step[0].tool == 'Imaging':
                    tool_input = step[0].tool_input
                    region = tool_input['action_input']["region"]
                    modality = tool_input['action_input']["modality"]
                    if region == "Abdomen":
                        imaging_requested = True
                        modality = modality_map.get(modality, modality)
                        data.append({"Model": prettify_model_name[model], "Pathology": patho.capitalize(), "Modality": modality})
                        break
            if not imaging_requested:
                data.append({"Model": prettify_model_name[model], "Pathology": patho.capitalize(), "Modality": "None"})

app_hadm_info_firstdiag         = load_hadm_from_file('appendicitis_hadm_info_first_diag', base_mimic=MIMIC_BASE)
cholec_hadm_info_firstdiag      = load_hadm_from_file('cholecystitis_hadm_info_first_diag', base_mimic=MIMIC_BASE)
pancr_hadm_info_firstdiag       = load_hadm_from_file('pancreatitis_hadm_info_first_diag', base_mimic=MIMIC_BASE)
divert_hadm_info_firstdiag      = load_hadm_from_file('diverticulitis_hadm_info_first_diag', base_mimic=MIMIC_BASE)

for patho, hadm_info in zip(['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis'], [app_hadm_info_firstdiag, cholec_hadm_info_firstdiag, divert_hadm_info_firstdiag, pancr_hadm_info_firstdiag]):
    for patient in hadm_info.keys():
        for rad in hadm_info[patient]["Radiology"]:
            if rad["Region"] == "Abdomen":
                #data.append({"Model": "Dataset", "Pathology": patho.capitalize(), "Modality": rad["Modality"]})
                data.append({"Model": "MIMIC Doctors", "Pathology": patho.capitalize(), "Modality": modality_map.get(rad["Modality"], rad["Modality"])})
                break

df = pd.DataFrame(data, columns=['Model', 'Pathology', 'Modality'])

sns.set(style="whitegrid", font_scale=1.4)

# 1. Calculating Counts
counts = df.groupby(['Model', 'Pathology', 'Modality']).size().reset_index(name='Counts')

# 2. Calculating Total Counts for Each Pathology and Model
total_counts = counts.groupby(['Model', 'Pathology'])['Counts'].transform('sum')

# 3. Calculating Percentages
counts['Percentage'] = (counts['Counts'] / total_counts) * 100

# Preparing data for grouped bar chart
models_in_counts = counts['Model'].unique()
pathologies = counts['Pathology'].unique()
modalities = counts['Modality'].unique()

# Number of bars (models) per group (pathology)
n_bars = len(models_in_counts)

# Width of each bar
bar_width = 0.15

# Define the custom order for modalities
modality_order = ["CT", "Ultrasound", "MRI", "Radiograph", "Other", "None"]

# Defining a color map for modalities
modality_colors = {"CT": "#C2C2C2",
    "Ultrasound": "#8A817C",
    "MRI" : "#58534B",
    "Radiograph" : "#4F5B62",
    "None": "#D6CCC2",
    "Other": "#FE6D73"}

# Defining model order
model_order = [prettify_model_name[model] for model in MODELS] + ["MIMIC Doctors"]

# Define hatch patterns for models
model_hatches = ['/', '.', '-', "x", '*']

# Plotting
plt.figure(figsize=(14, 6))

for j, model in enumerate(model_order):
    offsets = np.arange(len(pathologies)) + (j - n_bars / 2) * bar_width + bar_width/2
    bottom_val = np.zeros(len(pathologies))

    for modality in modality_order:
        model_modality_data = counts[(counts['Model'] == model) & (counts['Modality'] == modality)]
        heights = model_modality_data.set_index('Pathology').reindex(pathologies)['Percentage'].fillna(0).tolist()

        # Adding the bars
        bars = plt.bar(offsets, heights, width=bar_width, bottom=bottom_val, color=modality_colors[modality], hatch=model_hatches[j], edgecolor="black", label=f"{modality}" if j == 0 else "", alpha=1)

        for bar in bars:
            height = bar.get_height()
            if height > 5:  # Only add text labels if the bar segment is visible
                y_position = bar.get_y() + height / 2  # Calculate the y position for the label
                text_color = 'white' if (modality=="MRI" or modality=="Radiograph") else 'black'  # Choose text color for readability
                plt.text(bar.get_x() + bar.get_width() / 2, y_position, f'{height:.0f}', ha='center', va='center', color=text_color, weight="bold")


        bottom_val += np.array(heights)

plt.xlabel('Pathology')
plt.ylabel('Imaging Modality Requested (%)')
plt.xticks(np.arange(len(pathologies)), pathologies)

# Adjusting legend to show modalities only
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.9, 1.1),  ncol=len(by_label.keys()), frameon=False, fontsize=16)

# Create custom legend handles for modalities
modality_handles = [mpatches.Patch(color=modality_colors[modality], label=modality) for modality in modality_order]

# Create custom legend handles for models using hatch patterns
dense_model_hatches = ['///', '..', '---', 'xxx', '**']
model_handles = [mpatches.Patch(facecolor='white', edgecolor='black', hatch=dense_model_hatches[j], label=model_order[j]) for j in range(len(model_order))]

# Combine the modality and model handles
legend_handles = modality_handles + model_handles

# Add the legend to the plot
plt.legend(handles=legend_handles, bbox_to_anchor=(0.95, 1.25), ncol=5, frameon=False, fontsize=15)

# Save the plot
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(os.path.join(OUTPUT_BASE, dt_string), exist_ok=True)
plt.savefig(os.path.join(OUTPUT_BASE, dt_string, f"ImagingPercentages_{dt_string}.png"), dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()

# --- Plot Imaging Percentage End ---






# --- Plot Treatment Percentage  ---

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

# Use the existing variables from the main script
experiment = experiments[0]
model_scores = experiment_scores[experiment]
model_evals = experiment_evals[experiment]

def colo_replace(x):
    if x == "Colonoscopy":
        return "Future\nColonoscopy"
    else:
        return x

# Transform data structure into pandas DataFrame
data = []
treatment_required_counts = {}
for model in model_evals:
    for pathology in model_evals[model]:
        treatment_required_counts[pathology] = {}
        for patient_id in model_evals[model][pathology]:
            treatment_requested = model_evals[model][pathology][patient_id]['answers'].get('Treatment Requested', {})
            treatment_required = model_evals[model][pathology][patient_id]['answers'].get('Treatment Required', {})
            correctly_diagnosed = model_evals[model][pathology][patient_id]['scores'][DIAG]

            for treatment in treatment_required:
                if treatment_required[treatment]:
                    treatment_required_counts[pathology][treatment] = treatment_required_counts[pathology].get(treatment, 0) + 1
                    if correctly_diagnosed:
                        requested = treatment_requested.get(treatment, False)
                        data.append([model, pathology, treatment, requested]) 

df = pd.DataFrame(data, columns=['Model', 'Pathology', 'Treatment', 'Requested'])
df['Model'] = df['Model'].replace(prettify_model_name)

# Compute percentage of correctly requested treatment
df_agg = df.groupby(['Model', 'Pathology', 'Treatment']).mean().reset_index()
df_agg['Request Correct'] = df_agg['Requested']*100

df_counts = df.groupby(['Pathology', 'Treatment', 'Model']).size().reset_index(name='Counts')

# then merge this df_counts with your df_agg DataFrame
df_agg = pd.merge(df_agg, df_counts,  how='left', left_on=['Pathology','Treatment', 'Model'], right_on = ['Pathology','Treatment', 'Model'])

unique_pathologies = df_agg['Pathology'].unique()

# Set the style and color palette
sns.set_theme(style="whitegrid", font_scale=1.8)

# Create a 2x2 grid for subplots
fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharey=True)

# Flatten the axes array for easy iteration
axes_flat = axes.flatten()

# Iterate through each pathology and plot on its respective subplot
for i, pathology in enumerate(unique_pathologies):
    if i >= 4:  # Skip if there are more than 4 pathologies
        break
    df_filtered = df_agg[df_agg['Pathology'] == pathology]

    bar_plot = sns.barplot(x='Treatment', y='Request Correct', hue='Model', data=df_filtered, ax=axes_flat[i], palette=color_map, saturation=intensity)

    unique_treatments = df_filtered['Treatment'].unique()
    for j in range(len(unique_treatments) - 1):
        bar_plot.axvline(x=j + 0.5, color='gray', linestyle='--', linewidth=1)

    # Print counts below each bar
    for j, bar in enumerate(bar_plot.patches):
        if j < len(df_filtered):
            font_size = 16
            bar_plot.text(bar.get_x() + bar.get_width() / 2, -6, 
                          '{}'.format(df_filtered['Counts'].iloc[j]), 
                          ha='center', va='bottom', fontsize=font_size)
    
    # Print % correct above each bar
    for i,p in enumerate(bar_plot.patches):
        if i < len(df_filtered):
            bar_plot.annotate(format(p.get_height(), '.1f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 9), 
                textcoords = 'offset points',
                fontsize=16)

    # Set axes labels and titles
    bar_plot.set_xlabel('')
    bar_plot.set_ylabel('Treatment Requested (%)')
    bar_plot.set_title(f'{pathology.capitalize()} Treatment')
    label_with_count = lambda x: f"\n{colo_replace(x)}\n(n={treatment_required_counts[pathology][x]})"
    tick_positions = range(len(df_filtered['Treatment'].unique()))
    bar_plot.xaxis.set_major_locator(FixedLocator(tick_positions))
    bar_plot.set_xticklabels([label_with_count(tick.get_text()) for tick in bar_plot.get_xticklabels()])
    bar_plot.get_legend().remove()

# Hide any unused subplots
for ax in axes_flat[len(unique_pathologies):]:
    ax.axis('off')

# Set global parameters
plt.ylim(0, 105)
plt.setp(axes, xlabel='', ylabel='Treatment Requested (%)')

# Create a single legend
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=False)

# Adjust layout and save the figure
plt.tight_layout()

# Save the plot
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(os.path.join(OUTPUT_BASE, dt_string), exist_ok=True)
plt.savefig(os.path.join(OUTPUT_BASE, dt_string, f"TreatmentRequested_{dt_string}.png"), dpi=300, bbox_inches='tight')

plt.show()


# --- Plot Treatment Percentage End ---




# RAG PART: Per Model, Create a 2x2 grid, one cell per pathology, each cell contains a 2x2 grid, one cell per document, each cell contains a barplot with for x the number of chunks of the document and for y the relative (relative to the pathology) count of said chunk.

#For general heatmap
import pandas as pd
#Create pandas dataframe with each document name as a column and each pathology as a row
#Each cell contains the relative count of chunks of the document for the pathology
experiment = experiments[0]
for model in models:
    df_heatmap_dict = {}

    model_is_rag = False

    # model = prettify_model_name[model]
    model_pretty_name = prettify_model_name[model]
    model_retrievals = experiment_retrievals[experiment][model]

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
                    ax.bar(page, counts/total_chunks, label=str(page), color=color_map[model_pretty_name])
                    #make sure the label (page) appears above the bar
                    ax.text(page, counts/total_chunks, str(page), ha='center', va='bottom')
                else:
                    ax.bar(page, counts/total_chunks, color=color_map[model_pretty_name])

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
            plt.savefig(os.path.join(OUTPUT_BASE, dt_string, f"Retrievals_{model}_{pathology}_{dt_string}.png"), dpi=300, bbox_inches='tight')
        #close the figure
        plt.close()

    if model_is_rag:
        #Combine all last four plots into one big plot
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f'{model_pretty_name}')
        #load the images
        for i, pathology in enumerate(['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']):
            ax = axs[i//2, i%2]
            #remove margins
            ax.margins(x=0)
            #load the image
            img = plt.imread(os.path.join(OUTPUT_BASE, dt_string, f"Retrievals_{model}_{pathology}_{dt_string}.png"))
            ax.imshow(img)
            ax.axis('off')

        # plt.savefig(os.path.join(OUTPUT_BASE, f"Retrievals_{model}_all_{dt_string}.eps"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_BASE, dt_string, f"Retrievals_{model}_all_{dt_string}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        #delete the individual images
        for pathology in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']:
            os.remove(os.path.join(OUTPUT_BASE, dt_string,f"Retrievals_{model}_{pathology}_{dt_string}.png"))

    if model_is_rag:
        #Create a heatmap of the relative counts of chunks per document per pathology per model
        df_heatmap_count = pd.DataFrame(df_heatmap_dict)
        #Transpose the dataframe
        df_heatmap_count = df_heatmap_count.T
        #save df_heatmap_count to csv
        # df_heatmap_count.to_csv(os.path.join(OUTPUT_BASE, dt_string, f"Retrievals_{model}_heatmap_{dt_string}.csv"))
        #create new plot
        plt.figure(figsize=(12, 12))
        sns.heatmap(df_heatmap_count, cmap='viridis', annot=True, cbar_kws={'label': 'Relative Retrieval Per Pathology'})
        #make the names be diagonal
        plt.xticks(rotation=45)
        #make the names be horizontal
        plt.yticks(rotation=0)
        plt.title(f'{model_pretty_name}\nRelative Document Retrieval Per Pathology\n')
        plt.savefig(os.path.join(OUTPUT_BASE, dt_string, f"Retrievals_{model}_heatmap_{dt_string}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        #Compute Z-scores per pathology
        df_heatmap_z = df_heatmap_count.copy()
        df_heatmap_z = df_heatmap_z.T
        for pathology in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']:
            df_heatmap_z[pathology] = (df_heatmap_z[pathology] - df_heatmap_z[pathology].mean())/df_heatmap_z[pathology].std()
        df_heatmap_z = df_heatmap_z.T

        #save df_heatmap_z to csv
        # df_heatmap_z.to_csv(os.path.join(OUTPUT_BASE, dt_string, f"Retrievals_{model}_heatmap_zscores_{dt_string}.csv"))
        #create new plot
        plt.figure(figsize=(12, 12))
        sns.heatmap(df_heatmap_z, cmap='coolwarm', annot=True, cbar_kws={'label': 'Z-Score Relative Retrieval Per Pathology'})
        #make the names be diagonal
        plt.xticks(rotation=45)
        #make the names be horizontal
        plt.yticks(rotation=0)
        plt.title(f'{model_pretty_name}\nZ-Score Relative Document Retrieval Per Pathology\n')
        plt.savefig(os.path.join(OUTPUT_BASE, dt_string, f"Retrievals_{model}_heatmap_zscores_{dt_string}.png"), dpi=300, bbox_inches='tight')
        plt.close()

