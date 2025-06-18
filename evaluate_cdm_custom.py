from os.path import join
import pickle
import glob

from dataset.utils import load_hadm_from_file
from utils.logging import read_from_pickle_file
from run import load_evaluator

BASE_MIMIC = r"/container/data"

EVAL_MISMATCH = False
SEEDED = False
# SEEDS = [110,891,484,1126,431,435,1246,1852,283,1370,1063,1945,887,567,950,1633,1496,728,1507,1508]
SEEDS = [110,891,484,1126,431,435]

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


base_hosp = join("mimic-iv", "hosp")

# id_difficulty = pickle.load(open("id_difficulty.pkl", "rb"))
id_difficulty = pickle.load(open(BASE_MIMIC + "/id_difficulty.pkl", "rb"))
difficulty = "first_diag"
models = [
    # "Llama-3.2-1B-Instruct-exl2-4.0bpw",
    # "Llama-3.2-1B-Instruct-exl2-4.0bpw_stella_en_400M_v5",
    # "Llama-3.2-1B-Instruct-exl2-4.0bpw_stella_en_400M_v5_markdown",
    # "Llama-3.2-1B-Instruct-exl2-4.0bpw_stella_en_400M_v5_noise",
    # "Llama-3.2-1B-Instruct-exl2-4.0bpw_MedCPT_md_k4",
    # "Llama-3.2-1B-Instruct-exl2-4.0bpw_MedCPT_md_k4_no_rerank",
    # "Llama-3.2-1B-Instruct-exl2-4.0bpw_all-MiniLM-L6-v2",
    # "Llama-3.1-70B-Instruct-exl2-4.0bpw_NV-Embed-v2",
    # "Llama-3.1-70B-Instruct-exl2-4.0bpw_NV-Embed-v2-md_full",
    # "Llama-3.1-70B-Instruct-exl2-4.0bpw_NV-Embed-v2-md_k8",
    # "Llama-3.1-70B-Instruct-exl2-4.0bpw",
    # "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_400M_v5",
    # "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_400M_v5_markdown",
    # "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_400M_v5_smart_md",
    # "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_400M_v5_smart_md_full",
    # "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_400M_v5_smart_md_252k10",
    # "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_1.5B_v5",
    # "Llama-3.1-70B-Instruct-exl2-4.0bpw_NV-Embed-v2-md_ablated",
    # "Llama-3.1-70B-Instruct-exl2-4.0bpw_NV-Embed-v2-md_no_titles",
    # "Llama-3.1-70B-Instruct-exl2-4.0bpw_NV-Embed-v2-md_requery",
    # "Llama-3.1-70B-Instruct-exl2-4.0bpw_stella_en_400M_v5_requery",
    # "Llama-3.1-70B-Instruct-exl2-4.0bpw_MedCPT_badmd_requery",
    # "Llama-3.1-70B-Instruct-exl2-4.0bpw_MedCPT_md_requery",
    # "Llama-3.1-70B-Instruct-exl2-4.0bpw_MedCPT_refmd_requery",
    # "Llama-3.1-70B-Instruct-exl2-4.0bpw_MedCPT_pdf_requery",
    # "Llama-3.1-70B-Instruct-exl2-4.0bpw_NV-Embed-v2-md_requery_shortcontext",
    "Llama-3.1-70B-Instruct-exl2-4.0bpw_MedCPT_pubmed_requery",
]

if SEEDED:
    new_models = []
    for model in models:
        # new_models.append(model) #Add self to the models to evaluate
        for seed in SEEDS:
            new_models.append(f"{model}_SEED={seed}")
    models = new_models

experiment_results = {}
experiment_evals = {}
experiment_scores = {}
experiment_retrievals = {} #RAG: Per Experiment, Per Model, per Pathology, per Document, per Page, per Chunk Order, QTY of each Retrieved

for experiment in [
    # "CDM_VANILLA",
    # "CDM_NOSUMMARY"
    "BIN",
]:
    print(experiment)
    model_scores = {}
    model_evals = {}
    model_results = {}
    model_retrievals = {} #RAG: Per Model, per Pathology, per Document, per Page, per Chunk Order, QTY of each Retrieved

    for model in models:
        run = f"_ZeroShot_{model}_*_results.pkl"
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
            # Load patient data
            # hadm_info_clean = load_hadm_from_file(f"{patho}_hadm_info_clean")
            hadm_info_clean = load_hadm_from_file(f"{patho}_hadm_info_first_diag", base_mimic=BASE_MIMIC)
            all_evals[patho] = {}
            all_results[patho] = {}
            all_retrievals[patho] = {} #RAG: Per Document, per Page, per Chunk Order, QTY of each Retrieved

            # results_log_path = f"logs/SOTA/{experiment}/{patho}{run}"
            # results_log_path = f"{BASE_MIMIC}/logs/{patho}_{model}_*_{experiment}/{patho}{run}"
            if experiment == "CDM_VANILLA":
                # results_log_path = f"{BASE_MIMIC}/logs/{patho}_ZeroShot_{model}_*/{patho}{run}"
                results_log_path = f"{BASE_MIMIC}/logs/{patho}_ZeroShot_{model}_*[0-9]/{patho}{run}"
            else:
                results_log_path = f"{BASE_MIMIC}/logs/{patho}_ZeroShot_{model}_*_{experiment}/{patho}{run}"

            #if the model name contains "SEED" add "_seeded" after /logs/
            if "_SEED=" in model:
                results_log_path = results_log_path.replace("/logs/", "/logs/_seeded/")

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

                result = results[_id]

                #RAGFIX
                if "retrieval" in result:
                    retrieved_chunks = result["retrieval"]
                else:
                    retrieved_chunks = None

                evaluator = load_evaluator(
                    patho
                )  # Reload every time to ensure no state is carried over
                eval = evaluator._evaluate_agent_trajectory(
                    prediction=result["output"],
                    input=result["input"],
                    reference=(
                        hadm_info_clean[_id]["Discharge Diagnosis"],
                        hadm_info_clean[_id]["ICD Diagnosis"],
                        hadm_info_clean[_id]["Procedures ICD9"],
                        hadm_info_clean[_id]["Procedures ICD10"],
                        hadm_info_clean[_id]["Procedures Discharge"],
                    ),
                    agent_trajectory=result["intermediate_steps"],
                    retrieved_chunks=retrieved_chunks, #RAG
                )
                all_evals[patho][_id] = eval
                all_results[patho][_id] = result

                # --- RAG: Count the number of retrieved chunks ---

                #Chunk properties: chunk_id, document_reference, page_number, token_size, order_in_document, content
                #Chunk medcpt json properties: chunk_id, document_id, source, format, tags
                if retrieved_chunks is not None:
                    for step in retrieved_chunks:
                        for chunk in retrieved_chunks[step]:
                            # document_id
                            #get chunk format
                            chunk_format = chunk.get("format", "unknown")  # Default to unknown if not present
                            print(f"Processing chunk format: {chunk_format}")
                            # print("keys in chunk:", chunk.keys())
                            #keys in chunk: dict_keys(['chunk_id', 'document_reference', 'page_number', 'token_size', 'order_in_document', 'content', 'score'])
                            print("document_reference:", chunk.get("document_reference", "N/A"))

                            if chunk_format == "medcpt":
                                print(f"Processing MedCPT chunk: {chunk}")
                                doc_ref = chunk["document_id"]
                                doc_name = chunk["source"]
                                tags = chunk.get("tags", "").lower()  # Convert tags to lowercase for consistency
                                #since there are too many documents in medcpt pubmed set, instead of using document_reference, we use the pathologies they refer to
                                #but since we have no page number, we use the page_number for the document name
                                pathologies = []
                                if "appendicitis" in tags:
                                    pathologies.append("appendicitis")
                                if "cholecystitis" in tags:
                                    pathologies.append("cholecystitis")
                                if "diverticulitis" in tags:
                                    pathologies.append("diverticulitis")
                                if "pancreatitis" in tags:
                                    pathologies.append("pancreatitis")
                                if len(pathologies) == 0:
                                    pathologies = ["other"]

                                for pathology_tag in pathologies:
                                    if doc_ref not in all_retrievals[patho]:
                                        all_retrievals[patho][pathology_tag] = {}

                                    if doc_name not in all_retrievals[patho][pathology_tag]:
                                        all_retrievals[patho][pathology_tag][doc_name] = {}

                                    #use chunk_id instead of order_in_document for medcpt
                                    if chunk.get("chunk_id", "0") not in all_retrievals[patho][pathology_tag][doc_name]:
                                        all_retrievals[patho][pathology_tag][doc_name][chunk.get("chunk_id", "0")] = {
                                            "id": chunk.get("chunk_id", "0"),
                                            "count": 0,
                                            "content": chunk.get("content", ""),  # Use get to avoid KeyError if content is missing
                                        }
                                    
                                    #Add count
                                    all_retrievals[patho][pathology_tag][doc_name][chunk.get("chunk_id", "0")]["count"] += 1
                            else:
                                if chunk["document_reference"] not in all_retrievals[patho]:
                                    all_retrievals[patho][chunk["document_reference"]] = {}

                                if chunk["page_number"] not in all_retrievals[patho][chunk["document_reference"]]:
                                    all_retrievals[patho][chunk["document_reference"]][chunk["page_number"]] = {}

                                if chunk["order_in_document"] not in all_retrievals[patho][chunk["document_reference"]][chunk["page_number"]]:
                                    all_retrievals[patho][chunk["document_reference"]][chunk["page_number"]][chunk["order_in_document"]] = {
                                        "id": chunk["chunk_id"],
                                        "count": 0,
                                        "content": chunk.get("content", ""),  # Use get to avoid KeyError if content is missing
                                    }

                                #Add count
                                all_retrievals[patho][chunk["document_reference"]][chunk["page_number"]][chunk["order_in_document"]]["count"] += 1
                # --- RAG END ---

        model_evals[model] = all_evals
        model_results[model] = all_results
        model_retrievals[model] = all_retrievals #RAG
        avg_scores = {}
        avg_samples = {}

        for field in [
            "Diagnosis",
            "Gracious Diagnosis",
            "Physical Examination",
            "Late Physical Examination",
            "Action Parsing",
            "Treatment Parsing",
            "Diagnosis Parsing",
            "Rounds",
            "Invalid Tools",
            "Unnecessary Laboratory Tests",
            "Unnecessary Imaging",
        ]:
            avg_scores[field] = {}
            avg_samples[field] = {}
            for patho in [
                "appendicitis",
                "cholecystitis",
                "diverticulitis",
                "pancreatitis",
            ]:
                if field in ["Unnecessary Laboratory Tests", "Unnecessary Imaging"]:
                    all_evals[patho] = count_unnecessary(all_evals[patho], field)

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

        if difficulty == "first_diag":
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
    if difficulty == "first_diag":
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


if EVAL_MISMATCH:
    # =========================================
    # START OF DISEASE MISMATCH + RETRIEVAL EVALUATION SNIPPET
    # =========================================

    diseases = [
        "appendicitis",
        "cholecystitis",
        "diverticulitis",
        "pancreatitis",
    ]

    # This dictionary will hold diagnosis mismatch scores:
    # mismatch_data[model][actual_patho][predicted_patho] = list_of_scores_for_each_patient
    mismatch_data = {}

    # This dictionary will hold retrieval logs for each mismatch scenario:
    # mismatch_retrievals_data[model][actual_patho][predicted_patho][document][page_number][chunk_order] = {"count": X, "content": "..."}
    mismatch_retrievals_data = {}

    from tqdm import tqdm

    for model in tqdm(models, desc="Processing models"):
        mismatch_data[model] = {}
        mismatch_retrievals_data[model] = {}
        for actual_patho in tqdm(diseases, desc=f"Processing diseases for model {model}", leave=False):
            # Re-load the hadm_info for the actual pathology and difficulty if not in scope:
            hadm_info_clean = load_hadm_from_file(f"{actual_patho}_hadm_info_{difficulty}", base_mimic=BASE_MIMIC)

            mismatch_data[model][actual_patho] = {}
            mismatch_retrievals_data[model][actual_patho] = {}

            # Initialize predicted_patho entries
            for predicted_patho in diseases:
                mismatch_data[model][actual_patho][predicted_patho] = []
                mismatch_retrievals_data[model][actual_patho][predicted_patho] = {}

            # Make sure we have results for this model and pathology
            if model in model_results and actual_patho in model_results[model]:
                for _id, result in tqdm(model_results[model][actual_patho].items(), desc=f"Evaluating predictions for {model}-{actual_patho}", leave=False):
                    prediction = result["output"]
                    input_data = result["input"]
                    agent_trajectory = result["intermediate_steps"]

                    reference = (
                        hadm_info_clean[_id]["Discharge Diagnosis"],
                        hadm_info_clean[_id]["ICD Diagnosis"],
                        hadm_info_clean[_id]["Procedures ICD9"],
                        hadm_info_clean[_id]["Procedures ICD10"],
                        hadm_info_clean[_id]["Procedures Discharge"],
                    )

                    # Original retrieval logs for this patient
                    # This was gathered from the initial evaluation run
                    # They won't change when re-scoring with different predicted_pathologies
                    retrieved_chunks = result.get("retrieval", None)

                    # Evaluate the predicted diagnosis using each evaluator (predicted_pathology)
                    for predicted_patho in diseases:
                        evaluator = load_evaluator(predicted_patho)
                        eval_mismatch = evaluator._evaluate_agent_trajectory(
                            prediction=prediction,
                            input=input_data,
                            reference=reference,
                            agent_trajectory=agent_trajectory
                        )

                        # Record the diagnosis score
                        mismatch_data[model][actual_patho][predicted_patho].append(eval_mismatch["scores"]["Diagnosis"])

                        # Record the retrieval logs under this predicted_patho scenario as well
                        if eval_mismatch["scores"]["Diagnosis"] > 0:
                            # Only store retrieval logs if the diagnosis is correct
                            if retrieved_chunks is not None:
                                for step in retrieved_chunks:
                                    for chunk in retrieved_chunks[step]:
                                        doc_ref = chunk["document_reference"]
                                        page_num = chunk["page_number"]
                                        order_in_doc = chunk["order_in_document"]

                                        # Initialize nested dictionaries if not present
                                        if doc_ref not in mismatch_retrievals_data[model][actual_patho][predicted_patho]:
                                            mismatch_retrievals_data[model][actual_patho][predicted_patho][doc_ref] = {}
                                        if page_num not in mismatch_retrievals_data[model][actual_patho][predicted_patho][doc_ref]:
                                            mismatch_retrievals_data[model][actual_patho][predicted_patho][doc_ref][page_num] = {}
                                        if order_in_doc not in mismatch_retrievals_data[model][actual_patho][predicted_patho][doc_ref][page_num]:
                                            mismatch_retrievals_data[model][actual_patho][predicted_patho][doc_ref][page_num][order_in_doc] = {
                                                "id": chunk["chunk_id"],
                                                "count": 0,
                                                "content": chunk["content"],
                                            }

                                        # Add to the count
                                        mismatch_retrievals_data[model][actual_patho][predicted_patho][doc_ref][page_num][order_in_doc]["count"] += 1


    # Make the score of each predicted patho an average
    # for model in models:
    #     for actual_patho in diseases:
    #         for predicted_patho in diseases:
    #             mismatch_data[model][actual_patho][predicted_patho] = sum(mismatch_data[model][actual_patho][predicted_patho]) / len(mismatch_data[model][actual_patho][predicted_patho])

    # Save the mismatch data
    pickle.dump(
        mismatch_data,
        open(
            f"{BASE_MIMIC}/logs/analysis/{experiment}/mismatch_scores.pkl",
            "wb",
        ),
    )

    # Save the mismatch retrieval data
    pickle.dump(
        mismatch_retrievals_data,
        open(
            f"{BASE_MIMIC}/logs/analysis/{experiment}/mismatch_retrievals.pkl",
            "wb",
        ),
    )

    # =========================================
    # END OF DISEASE MISMATCH + RETRIEVAL EVALUATION SNIPPET
    # =========================================
