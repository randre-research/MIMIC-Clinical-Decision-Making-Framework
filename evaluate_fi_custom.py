from os.path import join
import pickle
import glob

from dataset.utils import load_hadm_from_file
from utils.logging import read_from_pickle_file
from run import load_evaluator

BASE_MIMIC = r"/container/data"

def calculate_average(evals, field, pathology):
    average = 0
    for patient in evals.keys():
        average += evals[patient]["scores"][field]

    average /= len(evals)
    # print(f'{pathology}: {average:0.02} (n={len(evals)})'.rjust(30))
    return average, len(evals)


def calculate_percentages(evals, field):
    for patient in evals.keys():
        evals[patient]["scores"][field] = (
            evals[patient]["scores"][field[: -len(" Percentage")]]
            / evals[patient]["max_scores"][field[: -len(" Percentage")]]
        )
    return evals


def count_unnecessary(evals, field):
    for patient in evals.keys():
        evals[patient]["scores"][field] = len(evals[patient]["answers"][field])
    return evals


# Check new evaluation strategy
base_hosp = join("mimic-iv", "hosp")

# id_difficulty = pickle.load(open("id_difficulty.pkl", "rb"))
id_difficulty = pickle.load(open(BASE_MIMIC + "/id_difficulty.pkl", "rb"))
difficulty = "first_diag"

experiment_results = {}
experiment_evals = {}
experiment_scores = {}
experiment_retrievals = {} #RAG: Per Experiment, Per Model, per Pathology, per Document, per Page, per Chunk Order, QTY of each Retrieved

for experiment in [
    "FULL_INFO_PLI_N_ONLYABNORMAL_BIN_BINABNORMAL_VANILLA_PROBS",
    # "FULL_INFO_PLI_N_BIN_BINABNORMAL",
    # "FI_H",
    # "FI_I",
    # "FI_ILP",
    # "FI_IPL",
    # "FI_L",
    # "FI_LIP",
    # "FI_LPI",
    # "FI_P",
    # "FI_PIL",
    # "FI_PLI",
    # "FI_PLI_ACUTE",
    # "FI_PLI_CONFIRM",
    # "FI_PLI_DR_NOABBR",
    # "FI_PLI_MAINDIAGNOSIS",
    # "FI_PLI_MINIMALSYSTEM",
    # "FI_PLI_NOFINAL",
    # "FI_PLI_NOMEDICAL",
    # "FI_PLI_NOPROMPT",
    # "FI_PLI_NOSUMMARY",
    # "FI_PLI_NOSYSTEM",
    # "FI_PLI_NOSYSTEMNOUSER",
    # "FI_PLI_P",
    # "FI_PLI_PRIMARYDIAGNOSIS",
    # "FI_PLI_SERIOUS",
    # "FI_PLI_ONLYABNORMAL",
]:
    print(experiment)
    model_scores = {}
    model_results = {}
    model_evals = {}
    model_retrievals = {} #RAG: Per Model, per Pathology, per Document, per Page, per Chunk Order, QTY of each Retrieved

    for model in [
        "Llama-3.2-1B-Instruct-exl2-4.0bpw",
        "Llama-3.2-1B-Instruct-exl2-4.0bpw_stella_en_400M_v5",
        "Llama-3.2-1B-Instruct-exl2-4.0bpw_stella_en_400M_v5_chunkr",

        # "Llama-3.1-70B-Instruct-exl2-4.0bpw",
        # "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_400M_v5",
        # "Llama-3.1-70B-Instruct-exl2-2.5bpw_stella_en_1.5B_v5",
        # "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_400M_v5_k12_8k",
        # "Llama-3.1-70B-Instruct-exl2-2.5bpw_stella_en_1.5B_v5_k12_8k",

        # "Llama-2-70B-chat-GPTQ",
        # "Llama2-70B-OASST-SFT-v10-GPTQ",
        # "WizardLM-70B-V1.0-GPTQ",
        # # "ClinicalCamel-70B-GPTQ",
        # # "Meditron-70B-GPTQ",
    ]:
        run = f"_{model}_*_FULL_INFO_*results.pkl"
        assert "result" in run

        all_evals = {}
        all_results = {}
        all_retrievals = {} #RAG: Per Pathology, per Document, per Page, per Chunk Order, QTY of each Retrieved
        for patho in [
            "appendicitis",
            "cholecystitis",
            "diverticulitis",
            "pancreatitis",
        ]:
            # evaluator = load_evaluator(patho)
            # Load patient data
            # hadm_info_clean = load_hadm_from_file(f"{patho}_hadm_info_first_diag")
            hadm_info_clean = load_hadm_from_file(f"{patho}_hadm_info_first_diag", base_mimic=BASE_MIMIC)
            all_evals[patho] = {}
            all_results[patho] = {}
            all_retrievals[patho] = {} #RAG: Per Document, per Page, per Chunk Order, QTY of each Retrieved

            # results_log_path = f"logs/SOTA/{experiment}/{patho}{run}"
            results_log_path = f"{BASE_MIMIC}/logs/{patho}_{model}_*_{experiment}/{patho}{run}"

            print(f"Loading {results_log_path}")
            #check if the path exists
            if len(glob.glob(results_log_path)) == 0:
                print(f"{results_log_path} not found")
            #else print how many files were found
            else:
                print(f"Found {len(glob.glob(results_log_path))} files for {results_log_path}")

            #Make sure none of the results accidentally contain the name of another model, e.g. "Llama-3.2-1B-Instruct-exl2-4.0bpw" will also match "Llama-3.2-1B-Instruct-exl2-4.0bpw_stella_en_400M_v5"
            if len(glob.glob(results_log_path)) > 1:
                #keep only the shortest path
                results_log_path = min(glob.glob(results_log_path), key=len)
                print(f"Multiple files, reducing to the shortest path {results_log_path}")

            results = []
            for r in read_from_pickle_file(glob.glob(results_log_path)[0]):
                results.append(r)
            results = {k: v for d in results for k, v in d.items()}

            for _id in id_difficulty[patho][difficulty]:
                if _id not in results:
                    print(f"Skipping {_id} | {glob.glob(results_log_path)[0]}")
                    continue
                # if "PROBS" in experiment or "SELFCONSISTENCY" in experiment:
                if "Diagnosis" in results[_id] and ("Probabilities" in results[_id] or "Retrieval" in results[_id]): #RAGFIX
                    result = "Final Diagnosis: " + results[_id]["Diagnosis"]
                    if "Probabilities" in results[_id]:
                        diag_probs = results[_id]["Probabilities"]
                    else:
                        diag_probs = None

                    if "Retrieval" in results[_id]:
                        retrieved_chunks = results[_id]["Retrieval"]
                    else:
                        retrieved_chunks = None #RAG
                else:
                    result = "Final Diagnosis: " + results[_id]
                    diag_probs = None
                    retrieved_chunks = None #RAG

                evaluator = load_evaluator(patho)

                eval = evaluator._evaluate_agent_trajectory(
                    prediction=result,
                    input="",
                    reference=(
                        hadm_info_clean[_id]["Discharge Diagnosis"],
                        hadm_info_clean[_id]["ICD Diagnosis"],
                        hadm_info_clean[_id]["Procedures ICD9"],
                        hadm_info_clean[_id]["Procedures ICD10"],
                        hadm_info_clean[_id]["Procedures Discharge"],
                    ),
                    agent_trajectory=[],
                    diagnosis_probabilities=diag_probs,
                    retrieved_chunks=retrieved_chunks, #RAG
                )
                all_evals[patho][_id] = eval
                all_results[patho][_id] = result

                # --- RAG: Count the number of retrieved chunks ---

                #Chunk properties: chunk_id, document_reference, page_number, token_size, order_in_document, content
                if retrieved_chunks is not None:
                    for chunk in retrieved_chunks:
                        if chunk["document_reference"] not in all_retrievals[patho]:
                            all_retrievals[patho][chunk["document_reference"]] = {}

                        if chunk["page_number"] not in all_retrievals[patho][chunk["document_reference"]]:
                            all_retrievals[patho][chunk["document_reference"]][chunk["page_number"]] = {}

                        if chunk["order_in_document"] not in all_retrievals[patho][chunk["document_reference"]][chunk["page_number"]]:
                            all_retrievals[patho][chunk["document_reference"]][chunk["page_number"]][chunk["order_in_document"]] = {
                                "id": chunk["chunk_id"],
                                "count": 0,
                                "content": chunk["content"],
                            }

                        #Add count
                        all_retrievals[patho][chunk["document_reference"]][chunk["page_number"]][chunk["order_in_document"]]["count"] += 1
                # --- RAG END ---

        model_evals[model] = all_evals
        model_results[model] = all_results
        model_retrievals[model] = all_retrievals #RAG
        avg_scores = {}
        avg_samples = {}

        for field in ["Diagnosis", "Gracious Diagnosis"]:
            avg_scores[field] = {}
            avg_samples[field] = {}
            for patho in [
                "appendicitis",
                "cholecystitis",
                "diverticulitis",
                "pancreatitis",
            ]:
                avg, n = calculate_average(all_evals[patho], field, patho)

                avg_scores[field][patho] = avg
                avg_samples[field][patho] = n
        model_scores[model] = avg_scores

        #Create folders when they don't exist
        import os
        #Create analysis folder
        if not os.path.exists(f"{BASE_MIMIC}/logs/analysis"):
            os.makedirs(f"{BASE_MIMIC}/logs/analysis")
        #Create experiment folder
        if not os.path.exists(f"{BASE_MIMIC}/logs/analysis/{experiment}"):
            os.makedirs(f"{BASE_MIMIC}/logs/analysis/{experiment}")

        if difficulty == "first_diag" or difficulty == "dr_eval":
            pickle.dump(
                all_evals,
                open(
                    # f"logs/SOTA/{experiment}/{model}_evals.pkl",
                    f"{BASE_MIMIC}/logs/analysis/{experiment}/{model}_evals.pkl",
                    "wb",
                ),
            )
            pickle.dump(
                all_results,
                open(
                    # f"logs/SOTA/{experiment}/{model}_results.pkl",
                    f"{BASE_MIMIC}/logs/analysis/{experiment}/{model}_results.pkl",
                    "wb",
                ),
            )
            pickle.dump(
                avg_scores,
                open(
                    # f"logs/SOTA/{experiment}/{model}_scores.pkl",
                    f"{BASE_MIMIC}/logs/analysis/{experiment}/{model}_scores.pkl",
                    "wb",
                ),
            )
            pickle.dump( #RAG
                all_retrievals,
                open(
                    # f"logs/SOTA/{experiment}/{model}_retrievals.pkl",
                    f"{BASE_MIMIC}/logs/analysis/{experiment}/{model}_retrievals.pkl",
                    "wb",
                ),
            ) #RAG
    if difficulty == "first_diag" or difficulty == "dr_eval":
        pickle.dump(
            model_evals,
            open(
                # f"logs/SOTA/{experiment}/evals.pkl",
                f"{BASE_MIMIC}/logs/analysis/{experiment}/evals.pkl",
                "wb",
            ),
        )
        pickle.dump(
            model_results,
            open(
                # f"logs/SOTA/{experiment}/results.pkl",
                f"{BASE_MIMIC}/logs/analysis/{experiment}/results.pkl",
                "wb",
            ),
        )
        pickle.dump(
            model_scores,
            open(
                # f"logs/SOTA/{experiment}/scores.pkl",
                f"{BASE_MIMIC}/logs/analysis/{experiment}/scores.pkl",
                "wb",
            ),
        )
        pickle.dump( #RAG
            model_retrievals,
            open(
                # f"logs/SOTA/{experiment}/retrievals.pkl",
                f"{BASE_MIMIC}/logs/analysis/{experiment}/retrievals.pkl",
                "wb",
            ),
        ) #RAG
    experiment_results[experiment] = model_results
    experiment_evals[experiment] = model_evals
    experiment_scores[experiment] = model_scores
    experiment_retrievals[experiment] = model_retrievals #RAG