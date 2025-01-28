# Constants for paths
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

EVAL_MISMATCH = False

EXPERIMENTS = [
    # "CDM_VANILLA",
    "BIN",
]

MODELS = [
    # "Llama-3.2-1B-Instruct-exl2-4.0bpw", 
    # "Llama-3.2-1B-Instruct-exl2-4.0bpw_stella_en_400M_v5",
    # "Llama-3.2-1B-Instruct-exl2-4.0bpw_stella_en_400M_v5_markdown",
    # "Llama-3.2-1B-Instruct-exl2-4.0bpw_stella_en_400M_v5_chunkr",
    # "Llama-3.2-1B-Instruct-exl2-4.0bpw_MedCPT_md_k4",
    # "Llama-3.2-1B-Instruct-exl2-4.0bpw_MedCPT_md_k4_no_rerank",
    # "Llama-3.2-1B-Instruct-exl2-4.0bpw_all-MiniLM-L6-v2",
    "Llama-3.1-70B-Instruct-exl2-4.0bpw",
    "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_400M_v5",
    "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_400M_v5_markdown",
    # "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_400M_v5_smart_md_252k10",
    "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_400M_v5_smart_md",
    "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_400M_v5_smart_md_full",
    # "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_1.5B_v5",
    # "Llama-3.1-70B-Instruct-exl2-2.5bpw_stella_en_1.5B_v5",
    # "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_400M_v5_k12_8k",
    # "Llama-3.1-70B-Instruct-exl2-2.5bpw_stella_en_1.5B_v5_k12_8k",
    "Llama-3.1-70B-Instruct-exl2-4.0bpw_NV-Embed-v2",
    ]

BASELINE_MODEL = MODELS[0]
# BASELINE_MODEL = "Llama-3.1-70B-Instruct-exl2-4.0bpw"
# BASELINE_MODEL = "Llama-3.2-1B-Instruct-exl2-4.0bpw"

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
    "Llama-3.2-1B-Instruct-exl2-4.0bpw_stella_en_400M_v5_markdown": "Llama3 1B 4.0bpw + stella5 400M (cleaned markdown)",
    "Llama-3.2-1B-Instruct-exl2-4.0bpw_stella_en_400M_v5_chunkr": "Llama3 1B 4.0bpw + stella5 400M (chunkr)",
    "Llama-3.2-1B-Instruct-exl2-4.0bpw_MedCPT_md_k4": "Llama3 1B 4.0bpw + MedCPT (k=4)",
    "Llama-3.2-1B-Instruct-exl2-4.0bpw_MedCPT_md_k4_no_rerank": "Llama3 1B 4.0bpw + MedCPT (k=4, no rerank)",
    "Llama-3.2-1B-Instruct-exl2-4.0bpw_all-MiniLM-L6-v2": "Llama3 1B 4.0bpw + MiniLM-L6-v2 (k=8, rerank=3)",
    # "Llama-3.1-70B-Instruct-exl2-4.0bpw": "Llama3 70B 4.0bpw",
    "Llama-3.1-70B-Instruct-exl2-4.0bpw": "LLM (Llama3 70B 4.0bpw)",
    "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_400M_v5": "Llama3 70B 4.0bpw + stella5 400M",
    # "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_400M_v5_markdown": "Llama3 70B 4.0bpw + stella5 400M (cleaned markdown)",
    "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_400M_v5_markdown": "LLM + RAG (Llama3 70B 4.0bpw, stella5 400M)",
    "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_400M_v5_smart_md_252k10": "Llama3 70B 4.0bpw + stella5 400M (smart markdown 252k10)",
    "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_400M_v5_smart_md": "Llama3 70B 4.0bpw + stella5 400M (smart markdown chunked)",
    "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_400M_v5_smart_md_full": "Llama3 70B 4.0bpw + stella5 400M (smart markdown full)",
    "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_1.5B_v5": "Llama3 70B 4.0bpw + stella5 1.5B",
    "Llama-3.1-70B-Instruct-exl2-2.5bpw_stella_en_1.5B_v5": "Llama3 70B 2.5bpw + stella5 1.5B",
    "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_400M_v5_k12_8k": "Llama3 70B 4.0bpw + stella5 400M (TopK=12, 8k Context)",
    "Llama-3.1-70B-Instruct-exl2-2.5bpw_stella_en_1.5B_v5_k12_8k": "Llama3 70B 2.5bpw + stella5 1.5B (TopK=12, 8k Context)",
    "Llama-3.1-70B-Instruct-exl2-4.0bpw_NV-Embed-v2": "Llama3 70B 4.0bpw + NV-Embed-v2 (markdown chunked)",
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
    "Llama3 1B 4.0bpw + stella5 400M (cleaned markdown)": "#4abd98",
    "Llama3 1B 4.0bpw + stella5 400M (chunkr)": "#1ee3ab",
    "Llama3 1B 4.0bpw + MedCPT (k=4)": "#F97F77",
    "Llama3 1B 4.0bpw + MedCPT (k=4, no rerank)": "#EC9898",
    "Llama3 1B 4.0bpw + MiniLM-L6-v2 (k=8, rerank=3)": "#F9F871",
    "Llama3 70B 4.0bpw": "#3EAD0A",
    "Llama3 70B 4.0bpw + stella5 400M": "#9BD415",
    "Llama3 70B 4.0bpw + stella5 400M (cleaned markdown)": "#d165d6",
    "Llama3 70B 4.0bpw + stella5 400M (smart markdown 252k10)": "#F9F871",
    "Llama3 70B 4.0bpw + stella5 400M (smart markdown chunked)": "#943199",
    "Llama3 70B 4.0bpw + stella5 400M (smart markdown full)": "#6d1d7b",
    "Llama3 70B 4.0bpw + stella5 1.5B": "#d4bb15",
    "Llama3 70B 2.5bpw + stella5 1.5B": "#F9F871",
    "Llama3 70B 4.0bpw + stella5 400M (TopK=12, 8k Context)": "#F97F77",
    "Llama3 70B 2.5bpw + stella5 1.5B (TopK=12, 8k Context)": "#EC9898",
    "Llama3 70B 4.0bpw + NV-Embed-v2 (markdown chunked)": "#1ee3ab",

    "LLM (Llama3 70B 4.0bpw)": "#0077B6",
    "LLM + RAG (Llama3 70B 4.0bpw, stella5 400M)": "#00B4D8",

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

def load_mismatch_data(
    experiments,
    models=["Llama-2-70B-chat-GPTQ", "Llama2-70B-OASST-SFT-v10-GPTQ", "WizardLM-70B-V1.0-GPTQ"],
    pathos=["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"],
):
    """
    Load mismatch scores and mismatch retrievals for specified experiments, models, and pathologies.

    Args:
        experiments (list): List of experiment names.
        models (list): List of models to load data for.
        pathos (list): List of pathologies to load data for.

    Returns:
        tuple: A tuple containing:
            - experiment_mismatch_scores (dict): Mismatch scores for each experiment and model.
            - experiment_mismatch_retrievals (dict): Mismatch retrievals for each experiment and model.
    """
    experiment_mismatch_scores = {}
    experiment_mismatch_retrievals = {}

    for experiment in experiments:
        model_mismatch_scores = {}
        model_mismatch_retrievals = {}

        # Load mismatch scores
        mismatch_scores_path = os.path.join(EXPERIMENTS_BASE, experiment, f"mismatch_scores.pkl")
        if os.path.exists(mismatch_scores_path):
            m_scores = pickle.load(open(mismatch_scores_path, "rb"))
        else:
            print(f"Mismatch scores file not found: {mismatch_scores_path}")
            m_scores = {}

        # Load mismatch retrievals
        mismatch_retrievals_path = os.path.join(EXPERIMENTS_BASE, experiment, f"mismatch_retrievals.pkl")
        if os.path.exists(mismatch_retrievals_path):
            m_retrievals = pickle.load(open(mismatch_retrievals_path, "rb"))
        else:
            print(f"Mismatch retrievals file not found: {mismatch_retrievals_path}")
            m_retrievals = {}

        for model in models:
            model_mismatch_scores[model] = m_scores[model] if model in m_scores else {}
            model_mismatch_retrievals[model] = m_retrievals[model] if model in m_retrievals else {}

            #Fix scores (scores = avg)
            for patho in pathos:
                for patho2 in pathos:
                    # print("Patients per patho: ", patho, " - ", patho2, " - :: ", len(model_mismatch_scores[model][patho][patho2]))
                    model_mismatch_scores[model][patho][patho2] = np.mean(model_mismatch_scores[model][patho][patho2])

        experiment_mismatch_scores[experiment] = model_mismatch_scores
        experiment_mismatch_retrievals[experiment] = model_mismatch_retrievals

    return experiment_mismatch_scores, experiment_mismatch_retrievals

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

experiments = EXPERIMENTS
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




# --- P-Values ---

import numpy as np
from scipy.stats import bartlett, ttest_ind, levene, chi2
# from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.contingency_tables import StratifiedTable
# Example per-disease accuracies for two models
# modelA = np.array([0.70, 0.78, 0.82, 0.75])  # 4 diseases
# modelB = np.array([0.65, 0.74, 0.79, 0.80])

# # 1. Bartlett’s test for equal variance
# stat_b, p_bartlett = bartlett(modelA, modelB)
# print(f"Bartlett's test: statistic={stat_b:.3f}, p-value={p_bartlett:.3f}")

# # 2. Welch’s two-sided t-test
# t_stat, p_value = ttest_ind(modelA, modelB, equal_var=False)
# print(f"Welch's t-test: t-stat={t_stat:.3f}, raw p-value={p_value:.3f}")

# # Suppose this is one of several comparisons -> apply Bonferroni
# m = 3  # or 5, depending on how many comparisons you do overall
# p_value_corrected = min(1, p_value * m)
# print(f"Bonferroni corrected p-value={p_value_corrected:.3f}")

"""
7. Interpreting the results

    Bartlett’s test: If p < 0.05, it suggests the groups have unequal variances → it justifies using Welch’s t-test.
    Welch’s t-test: If the final, Bonferroni-corrected p-value < 0.05, you conclude a statistically significant difference in the mean accuracies (i.e., per-disease accuracies) between the two groups/models.
    Effect size: Keep in mind a p-value does not show how big or meaningful the difference is. You might also want to look at the raw difference in the average accuracies (e.g., aˉ−bˉaˉ−bˉ) or compute confidence intervals.

8. Additional notes

    Using per-disease accuracy as “sample points”: Strictly speaking, you only have 4 diseases, so your sample size for each group is n=4n=4. A t-test with 4 data points per group is quite small; interpret with caution. The original paper apparently does this (or something similar) due to the constraints mentioned.
    Bartlett vs. Levene test: In practice, many prefer Levene’s test over Bartlett’s test because Bartlett’s is sensitive to nonnormal data. But you should follow the paper’s methodology if that’s a requirement.
    Multiclass + single-class metrics: The paper highlights that “per-class accuracy … is the only metric that can be calculated without bias” given their constraints. That is why they focus on per-class accuracies rather than overall accuracy, sensitivity, specificity, etc., which might be misleading with a small or unrepresentative sample of diseases.

Final Takeaway

    To compute p-values in the same way as the paper:

        Compute per-disease (per-class) accuracies for each model/doctor.
        Check variance with Bartlett’s test.
        Use a two-sided Welch’s t-test (ttest_ind(..., equal_var=False)) for comparing means.
        Bonferroni correct the resulting p-values by multiplying each by the number of comparisons (5 or 3 as in the paper).

    This method gives you a single p-value per comparison (e.g., doctor vs. model or specialist vs. generalist), corrected for multiple testing.
"""

no_rag_model = BASELINE_MODEL


# #Save model_evals to text file
# model_evals = experiment_evals[experiment]
# with open(os.path.join(OUTPUT_BASE, dt_string, f"model_evals_{dt_string}.txt"), "w") as f:
#     f.write(f"Experiment: {experiment}\n\n")
#     for model in model_evals.keys():
#         f.write(f"Model: {model}\n")
#         for patho in model_evals[model].keys():
#             f.write(f"Pathology: {patho}\n")
#             for _id in model_evals[model][patho].keys():
#                 f.write(f"ID: {_id}\n")
#                 f.write(f"Scores: {model_evals[model][patho][_id]}\n\n")
#         f.write("\n\n")

# #For each pathology in model_evals, order patients by ID
# for model in model_evals.keys():
#     for patho in model_evals[model].keys():
#         model_evals[model][patho] = dict(sorted(model_evals[model][patho].items()))

# model_scores_long = {}
# for model in model_evals.keys():
#     model_scores_long[model] = {}
#     for patho in model_evals[model].keys():
#         model_scores_long[model][patho] = []
#         for patient in model_evals[model][patho].keys():
#             model_scores_long[model][patho].append(model_evals[model][patho][patient]["scores"]["Diagnosis"])

# #Save model_scores_long to text file
# with open(os.path.join(OUTPUT_BASE, dt_string, f"model_scores_long_{dt_string}.txt"), "w") as f:
#     f.write(f"Experiment: {experiment}\n\n")
#     for model in model_scores_long.keys():
#         f.write(f"Model: {model}\n")
#         for patho in model_scores_long[model].keys():
#             f.write(f"Pathology: {patho}\n")
#             f.write(f"Scores: {model_scores_long[model][patho]}\n\n")
#         f.write("\n\n")




#Keep only, model evals Scores

# for each other model, compare to no_rag_model
for model in MODELS:
    if model == no_rag_model:
        continue
    else:
        # modelA = np.array([model_scores[no_rag_model][DIAG][patho] for patho in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']])
        # modelB = np.array([model_scores[model][DIAG][patho] for patho in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']])

        # modelA = np.array([model_scores_long[no_rag_model][patho] for patho in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']])
        # modelA = []
        # for patho in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']:
        #     modelA.extend(model_scores_long[no_rag_model][patho])
        # modelA = np.array(modelA)
        # # modelB = np.array([model_scores_long[model][patho] for patho in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']])
        # modelB = []
        # for patho in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']:
        #     modelB.extend(model_scores_long[model][patho])
        # modelB = np.array(modelB)
        
        # #McNemar's test for paired data
        # #contingency table
        # n_11 = np.sum(np.logical_and(modelA > 0.5, modelB > 0.5))
        # n_10 = np.sum(np.logical_and(modelA > 0.5, modelB <= 0.5))
        # n_01 = np.sum(np.logical_and(modelA <= 0.5, modelB > 0.5))
        # n_00 = np.sum(np.logical_and(modelA <= 0.5, modelB <= 0.5))

        # #cochran-mantel-haenszel
        # tables = []
        # for patho in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']:
        #     table = []
        #     for patient in model_evals[model][patho].keys():
        #         if patient in model_evals[no_rag_model][patho].keys():
        #             table.append([model_evals[model][patho][patient]["scores"]["Diagnosis"], model_evals[no_rag_model][patho][patient]["scores"]["Diagnosis"]])
        #     table = np.array(table)

        #     n_11 = np.sum(np.logical_and(table[:, 0] > 0.5, table[:, 1] > 0.5))
        #     n_10 = np.sum(np.logical_and(table[:, 0] > 0.5, table[:, 1] <= 0.5))
        #     n_01 = np.sum(np.logical_and(table[:, 0] <= 0.5, table[:, 1] > 0.5))
        #     n_00 = np.sum(np.logical_and(table[:, 0] <= 0.5, table[:, 1] <= 0.5))

        #     tables.append(np.array([[n_11, n_10], [n_01, n_00]]))

        # # Create the stratified table object
        # st = StratifiedTable(tables)

        # # Run the Cochran–Mantel–Haenszel test
        # result = st.test_equal_odds()

        # # The result is a 'ContrastResult' containing statistic, p-value, etc.
        # print("CMH test statistic:", result.statistic)
        # print("p-value:", result.pvalue)
        # print("Common odds ratio estimate:", result.oddsratio)
        
        # modelA = [92.7, 62.5, 68.5, 58.9]
        modelA = [70.6]
        modelA = np.array(modelA)
        #Repeat each number 5 times
        modelA = np.repeat(modelA, 3)

        # modelB = [94.9, 93.9, 93.5, 92.2, 94.6, 
        #           77.9, 72.8, 84.8, 74.9, 76.3,
        #           75.5, 71.2, 72.8, 70.4, 76.3,
        #           58.9, 66.4, 54.3, 49.6, 50.7]
        modelB = [76.0, 76.3, 77.8]
        modelB = np.array(modelB)

        # modelA = np.array([model_scores[no_rag_model][DIAG][patho] for patho in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']])
        # modelB = np.array([model_scores[model][DIAG][patho] for patho in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']])

        # modelA = [89,62,55,58,66]
        # modelB = [100,68,50,76,73]
        # modelA = np.array(modelA)
        # modelB = np.array(modelB)

        # # * 100 to get percentages
        # modelA *= 100
        # modelB *= 100

        m = 3


        # Append array to itself m times to get m*4 data points
        # modelA = np.tile(modelA, m)
        # modelB = np.tile(modelB, m)

        # 1. Bartlett’s test for equal variance
        stat_b, p_bartlett = bartlett(modelA, modelB)
        print(f"Bartlett's test for {model} vs {no_rag_model}: statistic={stat_b:.3f}, p-value={p_bartlett:.3f}")

        # 2. Welch’s two-sided t-test
        ttest_res = ttest_ind(modelA, modelB, equal_var=False)
        print(f"Welch's t-test for {model} vs {no_rag_model}: t-stat={ttest_res.statistic:.5f}, raw p-value={ttest_res.pvalue:.5f}")
        # # Welch’s t-test for equal variance
        # stat_w, p_welch = ttest_ind(modelA, modelB)
        # print(f"Welch's test for {model} vs {no_rag_model}: statistic={stat_w:.3f}, p-value={p_welch:.3f}")

        # Levene test for equal variance
        stat_l, p_levene = levene(modelA, modelB, center='mean')
        print(f"Levene's test for {model} vs {no_rag_model}: statistic={stat_l:.3f}, p-value={p_levene:.3f}")

        # Bonferroni correction for k=5 comparisons
        p_value_corrected = min(1, ttest_res.pvalue * m)
        print(f"Bonferroni corrected p-value for {model} vs {no_rag_model}={p_value_corrected:.3f}")

        # No need to correct for multiple comparisons here, as we only do one comparison
        #Save to text file
        with open(os.path.join(OUTPUT_BASE, dt_string, f"p-values_{dt_string}.txt"), "a") as f:
            #Write the modelA and modelB arrays first
            f.write(f"Model: {model} vs {no_rag_model}\n")
            f.write(f"{model} accuracies:\n{modelA}\n")
            f.write(f"{no_rag_model} accuracies:\n{modelB}\n")
            f.write(f"Bartlett's test for {model} vs {no_rag_model}:\nstatistic={stat_b:.7f}, p-value={p_bartlett:.7f}\n")
            f.write(f"Levene's test for {model} vs {no_rag_model}:\nstatistic={stat_l:.7f}, p-value={p_levene:.7f}\n")
            f.write(f"Welch's t-test for {model} vs {no_rag_model}:\nt-stat={ttest_res.statistic:.7f}, raw p-value={ttest_res.pvalue:.7f}\n")
            # f.write(f"Bonferroni corrected p-value for {model} vs {no_rag_model}:\n{p_value_corrected:.7f}\n\n")

# --- P-Values End ---




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
model_hatches = [
    '/',   # diagonal slash
    '.',   # dots
    '-',   # horizontal dash
    'x',   # diagonal cross
    '*',   # star
    '\\',  # backslash
    '|',   # vertical bar
    '+',   # plus
    'o',   # circle
    'O',   # larger circle
    '//',  # double slash
    '..',  # dense dots
]

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
dense_model_hatches = [
    '///',  # triple slashes
    '..',   # dots
    '---',  # triple dash
    'xxx',  # triple x
    '**',   # double asterisk
    '\\\\', # double backslash
    '||',   # vertical bars
    '+++',  # plus signs
    'ooo',  # triple o
    'OOO',  # triple capital O
]

model_handles = [mpatches.Patch(facecolor='white', edgecolor='black', hatch=dense_model_hatches[j], label=model_order[j]) for j in range(len(model_order))]

# Combine the modality and model handles
legend_handles = modality_handles + model_handles

# Add the legend to the plot
plt.legend(handles=legend_handles, bbox_to_anchor=(0.95, 1.25), ncol=5, frameon=False, fontsize=15)

# Save the plot
os.makedirs(os.path.join(OUTPUT_BASE, dt_string), exist_ok=True)
plt.savefig(os.path.join(OUTPUT_BASE, dt_string, f"ImagingPercentages_{dt_string}.png"), dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()

# --- Plot Imaging Percentage End ---


# --- Plot Imaging Count ---

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
from collections import defaultdict

# ----------------------------------------------------------------------------
# STEP A: Collect ALL imaging data while tracking unique patient IDs.
# ----------------------------------------------------------------------------

data = []
unique_patients_map = defaultdict(set)  # (model, pathology) -> set of patient IDs

for model in model_results.keys():
    for patho in model_results[model].keys():
        for _id in model_results[model][patho].keys():
            # Keep track of unique patients
            unique_patients_map[(prettify_model_name[model], patho.capitalize())].add(_id)
            found_any_imaging = False

            for step in model_results[model][patho][_id]['intermediate_steps']:
                if step[0].tool == 'Imaging':
                    tool_input = step[0].tool_input
                    region = tool_input['action_input']["region"]
                    modality = tool_input['action_input']["modality"]
                    if region == "Abdomen":
                        found_any_imaging = True
                        modality = modality_map.get(modality, modality)
                        data.append({
                            "Model": prettify_model_name[model],
                            "Pathology": patho.capitalize(),
                            "Modality": modality
                        })

            # If no abdominal imaging was requested, log "None"
            if not found_any_imaging:
                data.append({
                    "Model": prettify_model_name[model],
                    "Pathology": patho.capitalize(),
                    "Modality": "None"
                })

# For MIMIC Doctors
for patho, hadm_info in zip(
    ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis'],
    [app_hadm_info_firstdiag, cholec_hadm_info_firstdiag, 
     divert_hadm_info_firstdiag, pancr_hadm_info_firstdiag]
):
    for patient in hadm_info.keys():
        # Track unique patients for MIMIC
        unique_patients_map[("MIMIC Doctors", patho.capitalize())].add(patient)
        found_any_imaging = False

        for rad in hadm_info[patient]["Radiology"]:
            if rad["Region"] == "Abdomen":
                found_any_imaging = True
                data.append({
                    "Model": "MIMIC Doctors",
                    "Pathology": patho.capitalize(),
                    "Modality": modality_map.get(rad["Modality"], rad["Modality"])
                })

        if not found_any_imaging:
            data.append({
                "Model": "MIMIC Doctors",
                "Pathology": patho.capitalize(),
                "Modality": "None"
            })

# Convert to DataFrame
df = pd.DataFrame(data, columns=['Model', 'Pathology', 'Modality'])

# ----------------------------------------------------------------------------
# STEP B: Aggregate counts and remove "None".
# ----------------------------------------------------------------------------
counts = df.groupby(['Model', 'Pathology', 'Modality']).size().reset_index(name='Counts')
counts = counts[counts['Modality'] != 'None']  # Exclude "None"

# ----------------------------------------------------------------------------
# STEP C: Build a dictionary of unique patient counts for normalization.
# ----------------------------------------------------------------------------
patient_counts_dict = {}
for (model, patho), pid_set in unique_patients_map.items():
    patient_counts_dict[(model, patho)] = len(pid_set)

# Create a new column 'Normalized' = Counts / (# patients in that pathology for that model)
counts['Normalized'] = counts.apply(
    lambda row: row['Counts'] / patient_counts_dict[(row['Model'], row['Pathology'])],
    axis=1
)

# ----------------------------------------------------------------------------
# STEP D: Plot Setup (thinner hatches & white text).
# ----------------------------------------------------------------------------
sns.set(style="whitegrid", font_scale=1.3)

# Make hatches thinner & white
mpl.rcParams['hatch.color'] = 'white'
mpl.rcParams['hatch.linewidth'] = 0.5  # Thinner hatch lines

# We'll plot the 'Normalized' column on the y-axis.
models_in_counts = counts['Model'].unique()
pathologies = counts['Pathology'].unique()
modality_order = ["CT", "Ultrasound", "MRI", "Radiograph", "Other"]

model_order = [prettify_model_name[m] for m in MODELS] + ["MIMIC Doctors"]

# Single color per model
# color_map = {
#     "GPT-3.5": "#1f77b4",
#     "GPT-4": "#ff7f0e",
#     "Vicuna": "#2ca02c",
#     "MIMIC Doctors": "#8c564b",
# }

# White hatches for each modality
modality_hatches = {
    "CT": "///",
    "Ultrasound": "...",
    "MRI": "xxx",
    "Radiograph": "\\\\",
    "Other": "+++"
}

# Bar chart parameters
n_bars = len(model_order)
bar_width = 0.15

plt.figure(figsize=(14, 6))

for j, model in enumerate(model_order):
    offsets = np.arange(len(pathologies)) + (j - n_bars / 2) * bar_width + bar_width / 2
    bottom_val = np.zeros(len(pathologies))

    for modality in modality_order:
        subset = counts[(counts['Model'] == model) & (counts['Modality'] == modality)]

        # Grab the 'Normalized' values
        heights = (
            subset.set_index('Pathology')
            .reindex(pathologies)['Normalized']
            .fillna(0)
            .tolist()
        )

        bars = plt.bar(
            offsets,
            heights,
            width=bar_width,
            bottom=bottom_val,
            color=color_map.get(model, "#999999"),
            hatch=modality_hatches.get(modality, ""),
            #Edgecolor = white for all, black if MIMIC Doctors
            # edgecolor='black' if model == "MIMIC Doctors" else 'white',
            edgecolor='white',
            alpha=1.0,
            label=modality if j == 0 else ""
        )

        # # Plot the numbers in white
        # for bar in bars:
        #     height = bar.get_height()
        #     # if height > 0.0:
        #     if height >= 0.1:
        #         y_position = bar.get_y() + height / 2
        #         plt.text(
        #             bar.get_x() + bar.get_width() / 2,
        #             y_position,
        #             f'{height:.2f}',  # or use int(...) if desired
        #             ha='center',
        #             va='center',
        #             #Color = black for all, white if MIMIC Doctors
        #             color='black' if model != "MIMIC Doctors" else 'white',
        #             weight="bold",
        #             fontsize=10
        #         )

        bottom_val += np.array(heights)

    for i, offset in enumerate(offsets):
        total_height = bottom_val[i]
        if total_height > 0:
            # Slight vertical offset (0.01) to ensure the label isn't cut off
            plt.text(
                offset,
                total_height + 0.01,      # a small gap above the bar
                f'{total_height:.2f}',    # display as float or int
                ha='center',
                va='bottom',
                color='black',           # White text
                # weight='bold',
                fontsize=12
            )

# Labels and ticks
plt.xlabel('Pathology')
plt.ylabel('Imaging Requests Count\n(Average per Patient)')
plt.xticks(np.arange(len(pathologies)), pathologies)

# Legends for models and modalities
model_handles = [
    mpatches.Patch(facecolor=color_map[m], label=m)
    for m in model_order
]
modality_handles = [
    mpatches.Patch(
        facecolor='#333333',  # Dummy color
        hatch=modality_hatches[m],
        edgecolor='white',
        label=m
    ) for m in modality_order
]

plt.legend(
    handles=model_handles + modality_handles,
    bbox_to_anchor=(0.95, 1.15),
    ncol=4,
    frameon=False,
    fontsize=12
)

# Save & show
os.makedirs(os.path.join(OUTPUT_BASE, dt_string), exist_ok=True)
plt.savefig(
    os.path.join(OUTPUT_BASE, dt_string, f"ImagingNormalized_{dt_string}.png"),
    dpi=300,
    bbox_inches='tight'
)

plt.tight_layout()
plt.show()


# --- Plot Imaging Count End ---


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
os.makedirs(os.path.join(OUTPUT_BASE, dt_string), exist_ok=True)
plt.savefig(os.path.join(OUTPUT_BASE, dt_string, f"TreatmentRequested_{dt_string}.png"), dpi=300, bbox_inches='tight')

plt.show()


# --- Plot Treatment Percentage End ---




# --- Plot Physical Examination ---
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Use the same models as in the main code
models = MODELS  # from the main code
# experiment = 'BIN'  # Assuming 'BIN' is the experiment we are using
experiment = experiments[0]
fields = ['Physical Examination', 'Late Physical Examination']

# Now, call load_scores
experiment_results, experiment_evals, experiment_scores, experiment_retrievals = load_scores([experiment], fields=fields, models=models)

# Get the scores for the experiment
model_scores = experiment_scores[experiment]
model_results = experiment_results[experiment]
model_evals = experiment_evals[experiment]

# Prepare data
data = []
for model in model_scores.keys():
    mean_physical = np.mean([model_scores[model]['Physical Examination'][patho] for patho in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']])
    mean_physical_late = np.mean([model_scores[model]['Late Physical Examination'][patho] for patho in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']])
    data.append([model, mean_physical, mean_physical_late])

df = pd.DataFrame(data, columns=['Model', 'Physical Examination', '(Late) Physical Examination'])
df['Physical Examination'] *= 100
df['(Late) Physical Examination'] *= 100

# Reshaping the dataframe
melted_df = df.melt(id_vars=['Model'], var_name='Category', value_name='Percentage' )

sns.set(style="whitegrid", font_scale=1.4)

# Creating the bar plot
plt.figure(figsize=(10, 4))
melted_df['Model'] = melted_df['Model'].apply(lambda x: prettify_model_name[x])

bar_plot = sns.barplot(x='Category', y='Percentage', hue='Model', data=melted_df, palette=color_map, saturation=intensity)

# Adding the scores above the bars
for p in bar_plot.patches:
    if p.get_height() > 0:
        bar_plot.annotate(format(p.get_height(), '.1f'), 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha = 'center', va = 'center', 
                      xytext = (0, 9), 
                      textcoords = 'offset points')

# Additional plot formatting
plt.title('')
plt.ylabel('Examination Requested (%)')
plt.xlabel('')
plt.ylim(0, 100)
plt.legend(bbox_to_anchor=(0.9, 1.2),  ncol=len(model_scores.keys()), frameon=False, fontsize=16)
# Save the plot
plt.savefig(os.path.join(OUTPUT_BASE, dt_string, f"PhysicalExaminationPercentages_{dt_string}.png"), dpi=300, bbox_inches='tight')
plt.show()

# --- Plot Physical Examination End ---




# --- Plot Laboratory Tests ---
from agents.AgentAction import AgentAction

# Evaluators are loaded using the existing load_evaluator function
evaluators = {}
for patho in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']:
    evaluators[patho] = load_evaluator(patho)

# Using same models as in the main code
models = MODELS
# experiment = 'BIN'
experiment = experiments[0]
fields = []

# Load the scores
experiment_results, experiment_evals, experiment_scores, experiment_retrievals = load_scores([experiment], fields=fields, models=models)

model_scores = experiment_scores[experiment]
model_results = experiment_results[experiment]
model_evals = experiment_evals[experiment]

# Now, we need to process required lab tests
required_lab_tests = {}
for patho in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']:
    required_lab_tests[patho] = evaluators[patho].required_lab_tests

# Load the hadm_info files
app_hadm_info_firstdiag         = load_hadm_from_file('appendicitis_hadm_info_first_diag', base_mimic=MIMIC_BASE)
cholec_hadm_info_firstdiag      = load_hadm_from_file('cholecystitis_hadm_info_first_diag', base_mimic=MIMIC_BASE)
pancr_hadm_info_firstdiag       = load_hadm_from_file('pancreatitis_hadm_info_first_diag', base_mimic=MIMIC_BASE)
divert_hadm_info_firstdiag      = load_hadm_from_file('diverticulitis_hadm_info_first_diag', base_mimic=MIMIC_BASE)

# Now, we process MIMIC Doctors
model_evals["MIMIC Doctors"] = {}
for patho, hadm_info in zip(['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis'], [app_hadm_info_firstdiag, cholec_hadm_info_firstdiag, divert_hadm_info_firstdiag, pancr_hadm_info_firstdiag]):
    model_evals["MIMIC Doctors"][patho] = {}
    for patient in hadm_info.keys():
        evaluator = evaluators[patho]
        action = AgentAction(tool="Laboratory Tests", tool_input={"action_input": list(hadm_info[patient]['Laboratory Tests'].keys())}, log="", custom_parsings=0)
        evaluator.score_laboratory_tests(action)
        eval = {
            "scores": evaluator.scores.copy(),
            "answers": evaluator.answers.copy(),
        }
        model_evals["MIMIC Doctors"][patho][patient] = eval

# Prepare data for plotting
data = []
required_lab_test_percentages = {}
for model in model_evals.keys():
    required_lab_test_percentages[model] = {}
    for patho in model_evals[model].keys():
        required_lab_test_percentages[model][patho] = {}
        for required_lab_test in required_lab_tests[patho]:
            required_lab_test_percentages[model][patho][required_lab_test] = 0
        for _id in model_evals[model][patho].keys():
            for required_lab_test in required_lab_tests[patho]:
                required_lab_test_percentages[model][patho][required_lab_test] += 1 if len(model_evals[model][patho][_id]['answers']['Correct Laboratory Tests'][required_lab_test]) else 0
        for required_lab_test in required_lab_tests[patho]:
            required_lab_test_percentages[model][patho][required_lab_test] /= len(model_evals[model][patho].keys())
            required_lab_test_percentages[model][patho][required_lab_test] *= 100
            data.append([prettify_model_name[model], patho.capitalize(), required_lab_test, required_lab_test_percentages[model][patho][required_lab_test]])

df = pd.DataFrame(data, columns=['Model', 'Pathology', 'Required Laboratory Test', 'Percentage'])

# Creating the plot
sns.set_theme(style="whitegrid", font_scale=1.6)
fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)

# Replace 'Seriousness' with 'Severity' for consistency
df['Required Laboratory Test'] = df['Required Laboratory Test'].replace('Seriousness', 'Severity')

# Pathologies and their relevant tests
pathology_tests = {
    'Appendicitis': ['Inflammation'],
    'Cholecystitis': ['Inflammation', 'Liver', 'Gallbladder'],
    'Diverticulitis': ['Inflammation'],
    'Pancreatitis': ['Inflammation', 'Pancreas', 'Severity']
}

# Plotting each pathology in its subplot
for i, (pathology, tests) in enumerate(pathology_tests.items()):
    ax = axes[i // 2, i % 2]
    pathology_df = df[(df['Pathology'] == pathology) & (df['Required Laboratory Test'].isin(tests))]
    bar_plot = sns.barplot(x='Required Laboratory Test', y='Percentage', hue='Model', data=pathology_df, ax=ax, palette=color_map, saturation=intensity)

    unique_tests = pathology_df['Required Laboratory Test'].unique()
    for j in range(len(unique_tests) - 1):
        bar_plot.axvline(x=j + 0.5, color='gray', linestyle='--', linewidth=1)

    ax.set_title(pathology)
    if i > 1:
        ax.set_xlabel('Laboratory Test Category')
    else:
        ax.set_xlabel('')
    ax.set_ylabel('Lab Test Requested (%)')
    bar_plot.get_legend().remove()

    # Print % correct above each bar
    for p in bar_plot.patches:
        if p.get_height() > 0:
            adjust = 0 if p.get_height() < 99 else 10
            color = 'black' if p.get_height() < 99 else 'white'
            _format = '.1f' if p.get_height() < 99 else '.0f'
            bar_plot.annotate(format(p.get_height(), _format),
                (p.get_x() + p.get_width() / 2., p.get_height() - adjust),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points',
                fontsize=15, color=color)

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.ylim(0, 100)

# Create a single legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=len(model_evals.keys()), frameon=False, fontsize=16)

# Save the plot
plt.savefig(os.path.join(OUTPUT_BASE, dt_string, f"LaboratoryTestPercentages_{dt_string}.png"), dpi=300, bbox_inches='tight')
plt.show()

# --- Plot Laboratory Tests End ---







# --- Plot Instruction Following Scores ---
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Use the same models as in the main code
models = MODELS

# experiments = ['BIN', 'FI_PLI']
experiments = EXPERIMENTS
fields = ['Action Parsing', 'Treatment Parsing', 'Diagnosis Parsing', 'Invalid Tools']

# Load the scores for both experiments
experiment_results, experiment_evals, experiment_scores, experiment_retrievals = load_scores(experiments, fields=fields, models=models)

# Get scores for experiment
model_scores = experiment_scores[experiments[0]]
model_results = experiment_results[experiments[0]]
model_evals = experiment_evals[experiments[0]]

# Get scores for 'BIN' experiment
# model_scores = experiment_scores["BIN"]
# model_results = experiment_results["BIN"]
# model_evals = experiment_evals["BIN"]

# Get scores for 'FI_PLI' experiment
# model_scores_fi = experiment_scores["FI_PLI"]
# model_results_fi = experiment_results["FI_PLI"]
# model_evals_fi = experiment_evals["FI_PLI"]

data = []
for model in model_scores.keys():
    mean_action_parsing = np.mean([model_scores[model]['Action Parsing'][patho] for patho in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']])
    mean_treatment_parsing = np.mean([model_scores[model]['Treatment Parsing'][patho] for patho in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']])
    mean_diagnosis_parsing = np.mean([model_scores[model]['Diagnosis Parsing'][patho] for patho in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']])
    # mean_diagnosis_parsing_fi = np.mean([model_scores_fi[model]['Diagnosis Parsing'][patho] for patho in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']])
    mean_invalid_tools = np.mean([model_scores[model]['Invalid Tools'][patho] for patho in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']])
    # data.append([model, mean_action_parsing, mean_diagnosis_parsing, mean_invalid_tools, mean_diagnosis_parsing_fi])
    data.append([model, mean_action_parsing, mean_diagnosis_parsing, mean_invalid_tools])

# df = pd.DataFrame(data, columns=['Model', 'Action Parsing', 'Diagnosis Parsing', 'Invalid Tools', 'Diagnosis Parsing FI'])
df = pd.DataFrame(data, columns=['Model', 'Action Parsing', 'Diagnosis Parsing', 'Invalid Tools'])
df["Model"] = df["Model"].apply(lambda x: prettify_model_name[x])
df['Invalid Tools'] = 1 / df['Invalid Tools']
df["Action Parsing"] = 1 / df["Action Parsing"]
df["Diagnosis Parsing"] = 1 / df["Diagnosis Parsing"]
# df["Diagnosis Parsing FI"] = 1 / df["Diagnosis Parsing FI"]

# Reshaping the dataframe
melted_df = df.melt(id_vars=['Model'], var_name='Category', value_name='Percentage' )

sns.set(style="whitegrid", font_scale=1.4)

# Creating the bar plot
plt.figure(figsize=(12, 4))

# bar_plot = sns.barplot(x='Category', y='Percentage', hue='Model', data=melted_df, palette=color_map, order=['Action Parsing', 'Invalid Tools', 'Diagnosis Parsing', 'Diagnosis Parsing FI'], saturation=intensity)
bar_plot = sns.barplot(x='Category', y='Percentage', hue='Model', data=melted_df, palette=color_map, order=['Action Parsing', 'Invalid Tools', 'Diagnosis Parsing'], saturation=intensity)

# unique_categories = ['Action Parsing', 'Invalid Tools', 'Diagnosis Parsing', 'Diagnosis Parsing FI']
unique_categories = ['Action Parsing', 'Invalid Tools', 'Diagnosis Parsing']
for j in range(len(unique_categories) - 1):
    bar_plot.axvline(x=j + 0.5, color='gray', linestyle='--', linewidth=1)

# Adding the scores above the bars
for p in bar_plot.patches:
    if p.get_height() > 0:
        bar_plot.annotate(format(p.get_height(), '.1f'),
                      (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='center',
                      xytext=(0, 9),
                      textcoords='offset points')

# Additional plot formatting
plt.title('')
plt.ylabel('Average Number of Patients\nUntil Formatting Error')
plt.xlabel('')
plt.ylim(0, 30)
# plt.xticks(labels=['BIN\nNext Action Error', 'BIN\nTool Hallucination', 'BIN\nDiagnosis Error', 'FI_PLI\nDiagnosis Error'], ticks=[0, 1, 2, 3])
# plt.xticks(labels=['BIN\nNext Action Error', 'BIN\nTool Hallucination', 'BIN\nDiagnosis Error'], ticks=[0, 1, 2])
plt.xticks(labels=[experiment+'\nNext Action Error', experiment+'\nTool Hallucination', experiment+'\nDiagnosis Error'], ticks=[0, 1, 2])
plt.legend(bbox_to_anchor=(0.85, 1.2),  ncol=len(model_scores.keys()), frameon=False, fontsize=16)

# Save the plot
plt.savefig(os.path.join(OUTPUT_BASE, dt_string, f"InstructionFollowingScores_{dt_string}.png"), dpi=300, bbox_inches='tight')
plt.show()

# --- Plot Instruction Following Scores End ---





# --- Hallucination Plot ---

# --- Sample Data Collection ---
pathologies = ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']
data = []
for model in model_scores.keys():
    # Add each pathology row
    for patho in pathologies:
        tool_hallucination = model_scores[model]['Invalid Tools'][patho]
        data.append([model, patho.capitalize(), tool_hallucination])
    
    # Compute the average (mean) across pathologies for this model
    avg_tool_hallucination = np.mean([
        model_scores[model]['Invalid Tools'][patho] for patho in pathologies
    ])
    data.append([model, 'Mean', avg_tool_hallucination])

# Create a DataFrame
df_tool_hallucination = pd.DataFrame(
    data, columns=['Model', 'Pathology', 'Tool Hallucination']
)

# (Optional) Prettify model names
df_tool_hallucination['Model'] = df_tool_hallucination['Model'].apply(
    lambda x: prettify_model_name[x]
)

# --- Style the plot similarly to your diagnostic accuracy snippet ---
sns.set(style="whitegrid", font_scale=1.4)
plt.figure(figsize=(12, 4))

# Specify the order of categories so that 'Average' is last
order = ['Appendicitis', 'Cholecystitis', 'Diverticulitis', 'Pancreatitis', 'Mean']

bar_plot = sns.barplot(
    x='Pathology',
    y='Tool Hallucination',
    hue='Model',
    data=df_tool_hallucination,
    palette=color_map,   # Reuse your custom color_map if available
    saturation=intensity,  # Reuse your intensity if available
    order=order
)

# Draw vertical dashed lines between each category (except the last)
for i in range(len(order) - 1):
    bar_plot.axvline(x=i + 0.5, color='gray', linestyle='--', linewidth=1)

# Add the scores above the bars
for p in bar_plot.patches:
    if p.get_height() > 0:
        bar_plot.annotate(
            # format(p.get_height(), '.1f'),  # Format to 1 decimal place
            # Format to 2 decimal places
            format(p.get_height(), '.2f'),
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center',
            xytext=(0, 9),
            textcoords='offset points',
            fontsize=14
        )

# Additional plot formatting
plt.title('')
plt.ylabel('Tool Hallucination Count\n(Average per Patient)')
plt.xlabel('')
plt.ylim(0, max(df_tool_hallucination['Tool Hallucination']) * 1.2)  # Give headroom
plt.legend(
    bbox_to_anchor=(0.8, 1.18), 
    ncol=len(df_tool_hallucination['Model'].unique()),
    frameon=False, 
    fontsize=15
)


# Save the plot (optional)
now = datetime.now()
plt.savefig(os.path.join(OUTPUT_BASE, dt_string, f"HallucinationPlot_{dt_string}.png"), dpi=300, bbox_inches='tight')
plt.show()

# --- Hallucination Plot End ---





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

if EVAL_MISMATCH:
    # === MISMATCH RAG PLOT ===

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    def create_mismatch_heatmap(mismatch_data, mismatch_retrievals_data, models, pathologies, output_base, dt_string):
        """
        Generate a heatmap for mismatch data.

        Args:
            mismatch_data (dict): Contains scores for actual vs predicted pathologies.
            mismatch_retrievals_data (dict): Contains retrieval data for each mismatch.
            models (list): List of models to process.
            pathologies (list): List of pathologies.
            output_base (str): Base directory for saving outputs.
            dt_string (str): Timestamp for filenames.
        """
        for model in models:
            model_mismatch_scores = mismatch_data[model]
            model_mismatch_retrievals = mismatch_retrievals_data[model]

            # Prepare data for the heatmap
            heatmap_data = []
            documents = []

            for actual_patho in pathologies:
                row = []
                subrows = []
                for predicted_patho in pathologies:
                    # Calculate total score and counts for each document
                    if actual_patho in model_mismatch_retrievals and predicted_patho in model_mismatch_retrievals[actual_patho]:
                        retrieval_data = model_mismatch_retrievals[actual_patho][predicted_patho]
                        for document in retrieval_data.keys():
                            total_chunks = sum(
                                retrieval_data[document][page][chunk]["count"]
                                for page in retrieval_data[document]
                                for chunk in retrieval_data[document][page]
                            )
                            row.append(total_chunks)
                            subrows.append((document, total_chunks))
                            documents.append(document)
                    else:
                        row.append(0)
                heatmap_data.append(row)

            # Normalize the rows by the number of documents retrieved
            for i, row in enumerate(heatmap_data):
                row_total = sum(row)
                if row_total > 0:
                    heatmap_data[i] = [x / row_total for x in row]

            # Create the heatmap
            plt.figure(figsize=(len(pathologies) * 3, len(pathologies) * 2))
            df_heatmap = pd.DataFrame(heatmap_data, index=pathologies, columns=pathologies)

            sns.heatmap(
                df_heatmap,
                annot=True,
                cmap="coolwarm",
                cbar_kws={"label": "Normalized Retrieval Count"},
                xticklabels=pathologies,
                yticklabels=pathologies,
            )

            plt.title(f"{model} - Mismatch Retrieval Heatmap")
            plt.xlabel("Predicted Pathology")
            plt.ylabel("Actual Pathology")
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save the heatmap
            os.makedirs(os.path.join(output_base, dt_string), exist_ok=True)
            heatmap_path = os.path.join(output_base, dt_string, f"{model}_mismatch_heatmap.png")
            plt.savefig(heatmap_path, dpi=300)
            plt.close()

            # Save the document-level breakdown (optional)
            doc_df = pd.DataFrame(subrows, columns=["Document", "Count"])
            doc_breakdown_path = os.path.join(output_base, dt_string, f"{model}_document_breakdown.csv")
            doc_df.to_csv(doc_breakdown_path, index=False)

            print(f"Saved heatmap to {heatmap_path}")
            print(f"Saved document breakdown to {doc_breakdown_path}")


    mismatch_data, mismatch_retrievals_data = load_mismatch_data(experiments, models)

    experiment = experiments[0]
    models = MODELS

    mismatch_data = mismatch_data[experiment]
    mismatch_retrievals_data = mismatch_retrievals_data[experiment]

    #save the mismatch data to csv
    mismatch_data_df = pd.DataFrame(mismatch_data)
    mismatch_data_df.to_csv(os.path.join(OUTPUT_BASE, dt_string, f"MismatchData_{dt_string}.csv"), index=False)



    #print keys of mismatch_data and mismatch_retrievals_data to know what to pass to the function
    print("Mismatch data keys:", mismatch_data.keys())
    print("Mismatch retrievals data keys:", mismatch_retrievals_data.keys())

    # Example usage
    # create_mismatch_heatmap(
    #     mismatch_data=mismatch_data,
    #     mismatch_retrievals_data=mismatch_retrievals_data,
    #     models=models,
    #     pathologies=["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"],
    #     output_base=OUTPUT_BASE,
    #     dt_string=dt_string,
    # )



    # === CREATE MISMATCH HEATMAP ===

    import pickle
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Pathologies under consideration
    diseases = [
        "appendicitis",
        "cholecystitis",
        "diverticulitis",
        "pancreatitis",
    ]

    # Create a heatmap for each model
    for model in models:
        # Initialize a matrix to store average scores
        # Rows = actual pathologies, Columns = predicted pathologies
        matrix = np.zeros((len(diseases), len(diseases)))

        for i, actual_patho in enumerate(diseases):
            for j, predicted_patho in enumerate(diseases):
                matrix[i, j] = mismatch_data[model][actual_patho][predicted_patho]

        # Create a heatmap using seaborn
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", xticklabels=diseases, yticklabels=diseases, cbar_kws={"label": "Average Diagnosis Score"}, ax=ax)
        ax.set_xlabel("Predicted Pathology")
        ax.set_ylabel("Actual Pathology")
        ax.set_title(f"Diagnosis Confusion Heatmap for\n{model}")

        #Make sure each cell is a square
        ax.set_aspect('equal')

        #Angle the x-axis labels
        plt.xticks(rotation=45)

        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_BASE, dt_string, f"MismatchHeatmap__{model}_{dt_string}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # === CREATE MISMATCH HEATMAP END ===


    # === CREATE RETRIEVAL MISMATCH HEATMAP ===

    #Map of document counts per pathology predicted per pathology actual per model

    map = {}
    for model in models:
        model_is_rag = False

        map[model] = {}
        for pathology in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']:
            map[model][pathology] = {"count": 0}
            for pathology2 in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']:
                map[model][pathology][pathology2] = {"count": 0}
                if pathology in mismatch_retrievals_data[model] and pathology2 in mismatch_retrievals_data[model][pathology]:
                    for document in mismatch_retrievals_data[model][pathology][pathology2]:
                        total_chunks = 0
                        for page in mismatch_retrievals_data[model][pathology][pathology2][document]:
                            for order in mismatch_retrievals_data[model][pathology][pathology2][document][page]:
                                total_chunks += mismatch_retrievals_data[model][pathology][pathology2][document][page][order]["count"]
                        map[model][pathology][pathology2][document] = total_chunks
                        map[model][pathology]["count"] += total_chunks
                        map[model][pathology][pathology2]["count"] += total_chunks

                        if total_chunks > 0:
                            model_is_rag = True

        if model_is_rag:
            #order the documents by alphabetical order
            for pathology in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']:
                for pathology2 in ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']:
                    map[model][pathology][pathology2] = {k: v for k, v in sorted(map[model][pathology][pathology2].items(), key=lambda item: item[0])}

            #Create the heatmap
            qty_docs = len([k for k in map[model]['appendicitis']['appendicitis'].keys() if k != "count"])

            # Create the matrix
            matrix = np.zeros((len(diseases)*qty_docs, len(diseases)))

            for i, actual_patho in enumerate(diseases):
                # Retrieve sorted document keys for the "actual_patho/actual_patho" pair (just to ensure consistent count)
                doc_keys_for_actual = [doc for doc in map[model][actual_patho][actual_patho].keys() if doc != "count"]
                # Make sure qty_docs matches the actual number of documents for each pathology pair if needed
                # qty_docs = len(doc_keys_for_actual)  # If needed

                for j, predicted_patho in enumerate(diseases):
                    # Now iterate over the sorted keys from `map`, excluding 'count'
                    doc_keys = [doc for doc in map[model][actual_patho][predicted_patho].keys() if doc != "count"]
                    
                    for k, document in enumerate(doc_keys):
                        matrix[i*qty_docs + k, j] = map[model][actual_patho][predicted_patho][document]
                        # Normalize by the total count of chunks for the actual pathology
                        matrix[i*qty_docs + k, j] /= map[model][actual_patho]["count"]

            # # Save map to JSON
            # import json
            # with open(os.path.join(OUTPUT_BASE, dt_string, f"Retrievals_{model}_map_{dt_string}.json"), 'w') as f:
            #     json.dump(map[model], f)

            # Get all document names from the sorted keys in the primary pair 'appendicitis/appendicitis'
            documents = [key for key in map[model]['appendicitis']['appendicitis'].keys() if key != "count"]
            # Split and shorten long document names
            documents = [doc.split("/")[-1] for doc in documents]
            documents = [doc[:25] + "..." if len(doc) > 25 else doc for doc in documents]

            # Plotting the heatmap with consistent ordering
            fig, ax = plt.subplots(figsize=(24, 24))
            sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", xticklabels=diseases,
                        yticklabels=[f"{disease}\n{doc}" for disease in diseases for doc in documents],
                        cbar_kws={"label": "Normalized Retrieval Count"}, ax=ax)
            ax.set_xlabel("Predicted Pathology")
            ax.set_ylabel("Actual Pathology")
            ax.set_title(f"Retrieval Mismatch Heatmap (Weighted by total chunks per actual pathology) for\n{model}")

            #Angle the x-axis labels
            plt.xticks(rotation=45)

            # Save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_BASE, dt_string, f"MismatchRetrievalHeatmap__{model}_{dt_string}.png"), dpi=300, bbox_inches='tight')




            # === CREATE RETRIEVAL MISMATCH HEATMAP WEIGHTED ===

            # Create the matrix
            matrix = np.zeros((len(diseases)*qty_docs, len(diseases)))

            for i, actual_patho in enumerate(diseases):
                # Retrieve sorted document keys for the "actual_patho/actual_patho" pair (just to ensure consistent count)
                doc_keys_for_actual = [doc for doc in map[model][actual_patho][actual_patho].keys() if doc != "count"]
                # Make sure qty_docs matches the actual number of documents for each pathology pair if needed
                # qty_docs = len(doc_keys_for_actual)  # If needed

                for j, predicted_patho in enumerate(diseases):
                    # Now iterate over the sorted keys from `map`, excluding 'count'
                    doc_keys = [doc for doc in map[model][actual_patho][predicted_patho].keys() if doc != "count"]
                    
                    for k, document in enumerate(doc_keys):
                        matrix[i*qty_docs + k, j] = map[model][actual_patho][predicted_patho][document]
                        # Normalize by the total count of chunks for the PREDICTED PATHOLOGY for the actual pathology
                        matrix[i*qty_docs + k, j] /= map[model][actual_patho][predicted_patho]["count"]

            # # Save map to JSON
            # import json
            # with open(os.path.join(OUTPUT_BASE, dt_string, f"Retrievals_{model}_map_{dt_string}.json"), 'w') as f:
            #     json.dump(map[model], f)

            # Get all document names from the sorted keys in the primary pair 'appendicitis/appendicitis'
            documents = [key for key in map[model]['appendicitis']['appendicitis'].keys() if key != "count"]
            # Split and shorten long document names
            documents = [doc.split("/")[-1] for doc in documents]
            documents = [doc[:25] + "..." if len(doc) > 25 else doc for doc in documents]

            # Plotting the heatmap with consistent ordering
            fig, ax = plt.subplots(figsize=(24, 24))
            sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", xticklabels=diseases,
                        yticklabels=[f"{disease}\n{doc}" for disease in diseases for doc in documents],
                        cbar_kws={"label": "Normalized Retrieval Count"}, ax=ax)
            ax.set_xlabel("Predicted Pathology")
            ax.set_ylabel("Actual Pathology")
            ax.set_title(f"Retrieval Mismatch Heatmap (Weighted by total chunks per predicted pathology per actual pathology) for\n{model}")

            #Angle the x-axis labels
            plt.xticks(rotation=45)

            # Save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_BASE, dt_string, f"MismatchRetrievalHeatmapPredictProfiles__{model}_{dt_string}.png"), dpi=300, bbox_inches='tight')

            # === CREATE RETRIEVAL MISMATCH HEATMAP END WEIGHTED ===


    # === CREATE RETRIEVAL MISMATCH HEATMAP END ===
