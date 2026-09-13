# Copyright Reexpress AI, Inc. All rights reserved.

# Fixed to mlx-community/gemma-4-31b-it-4bit (see the module header). Embeddings are 3 x 5376 = 16128 dims.
# max-pool over the sequence :: mean-pool over the sequence :: hidden-state of the final token (that estimates Yes | No)

import json
import sys
import numpy as np
import logging
import argparse
import time
from pathlib import Path
import codecs

import data_utils
import mcp_utils_llm_api_gemma_4_31b_it_mlx as agreement

logger = logging.getLogger(__name__)


REEXPRESS_ID_KEY = "id"
REEXPRESS_LABEL_KEY = "label"
REEXPRESS_DOCUMENT_KEY = "document"
REEXPRESS_ATTRIBUTES_KEY = "attributes"
REEXPRESS_EMBEDDING_KEY = "embedding"

EXPECTED_EMBEDDING_SIZE = 16128  # 3 x 5376 = 16128


FACTCHECK_DATA = "factcheck"
SENTIMENT_DATA = "sentiment"


def print_summary(header_label, list_to_process, total=None):
    if total is not None and total > 0:
        print(
            f"{header_label} \tmean: {np.mean(list_to_process) if len(list_to_process) > 0 else 0}, "
            f"\tout of {len(list_to_process)} "
            f"\t({(len(list_to_process)/total) * 100}%) of {total}")
    else:
        print(
            f"{header_label} \tmean: {np.mean(list_to_process) if len(list_to_process) > 0 else 0}, "
            f"\tout of {len(list_to_process)}")


# For consistency, we just pull this from the JSON from the factcheck_mixtral_8x7b data.
# def get_prompt(document, dataset):
#     if dataset == FACTCHECK_DATA:
#         return f"Here is a statement that may contain errors. <statement> {document} </statement> Is the statement true? Answer Yes if the statement is true. Answer No if the statement is false. Start your response with Yes or No."
#     elif dataset == SENTIMENT_DATA:
#         return f"Here is a movie review. <review> {document} </review> Is the sentiment of the movie review positive? Answer Yes if the sentiment is positive. Answer No if the sentiment is negative. Start your response with Yes or No."
#     else:
#         assert False


def get_existing_ids(filepath_with_name):
    existing_ids = set()
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            existing_ids.add(json_obj[REEXPRESS_ID_KEY])
    return existing_ids


def construct_embedding_streaming(options, model):
    count_incomplete_responses = 0
    output_file = options.output_file
    if Path(output_file).exists():
        existing_ids = get_existing_ids(output_file)
    else:
        existing_ids = set()

    acc = []
    acc_by_class = {}
    acc_by_predicted_class = {}
    for class_i in range(options.class_size):
        acc_by_class[class_i] = []
        acc_by_predicted_class[class_i] = []

    instance_i = -1
    with codecs.open(options.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            instance_i += 1
            if instance_i % 50000 == 0:
                print(f"Currently processing instance {instance_i}")
            # The output id (possibly prefixed) is what is stored in the output file, so it must be used for the
            # resume check as well.
            if options.add_shuffled_prefix:
                output_id = f"shuffled_{json_obj[REEXPRESS_ID_KEY]}"
            else:
                output_id = json_obj[REEXPRESS_ID_KEY]
            if output_id in existing_ids:
                continue

            prompt = json_obj["prompt"]
            try:
                embedding, llm_classification = \
                    agreement.get_agreement_model_embedding(lm=model, document_text=prompt, device=None,
                                                            include_max_pool=True,
                                                            include_yes_no_logits=False)
            except Exception:
                logger.exception("Agreement model embedding failed.")
                print(f"LINE_{instance_i}: {json_obj[REEXPRESS_ID_KEY]}")
                # In principle, this case should never occur with this model, so we exit to investigate further.
                sys.exit(1)
            assert len(embedding) == EXPECTED_EMBEDDING_SIZE, \
                f"Unexpected embedding size {len(embedding)} (expected {EXPECTED_EMBEDDING_SIZE})"
            llm_classification = int(llm_classification)  # module returns a bool; store as 0/1 like the label

            new_json_obj = {}
            new_json_obj[REEXPRESS_ID_KEY] = output_id
            new_json_obj[REEXPRESS_LABEL_KEY] = json_obj[REEXPRESS_LABEL_KEY]
            new_json_obj[REEXPRESS_DOCUMENT_KEY] = json_obj[REEXPRESS_DOCUMENT_KEY]
            new_json_obj[REEXPRESS_EMBEDDING_KEY] = embedding
            # no attributes in this case and a simplified set of additional metadata
            new_json_obj["prompt"] = prompt
            new_json_obj["llm_classification"] = llm_classification
            acc.append(llm_classification == new_json_obj[REEXPRESS_LABEL_KEY])
            acc_by_class[new_json_obj[REEXPRESS_LABEL_KEY]].append(
                llm_classification == new_json_obj[REEXPRESS_LABEL_KEY])
            acc_by_predicted_class[llm_classification].append(
                llm_classification == new_json_obj[REEXPRESS_LABEL_KEY])

            data_utils.save_by_appending_json_lines(output_file, [new_json_obj])
            existing_ids.add(output_id)

    print(f"Count of documents with embedding set to 0's: {count_incomplete_responses}")
    model_label = agreement.DEFAULT_MODEL_PATH
    print_summary(f"{model_label} accuracy", acc, total=len(acc))
    print(f"Class-conditional accuracy (i.e., stratified by TRUE class):")
    for class_i in range(options.class_size):
        print_summary(f"{model_label} accuracy true class {class_i}",
                      acc_by_class[class_i], total=len(acc))
    print(f"Prediction-conditional accuracy (i.e., stratified by PREDICTED class):")
    for class_i in range(options.class_size):
        print_summary(f"{model_label} accuracy predicted class {class_i}",
                      acc_by_predicted_class[class_i], total=len(acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="-----[Add embedding data to JSON objects]-----")
    parser.add_argument("--input_file", default="", help="")
    parser.add_argument("--class_size", default=2, type=int, help="class_size")
    parser.add_argument("--dataset", default="", help="")
    parser.add_argument("--add_shuffled_prefix", action="store_true")
    parser.add_argument("--output_file", default="", help="")

    options = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # The sentiment data from Schmaltz 2026 ("Similarity-Distance-Magnitude Activations") can be processed
    # in a similar way, but for simplicity, we only consider
    # the Azaria and Mitchell (2023) data in the tutorial for Gemma 4 31B
    # (specifically, "mlx-community/gemma-4-31b-it-4bit").
    assert options.dataset in [FACTCHECK_DATA]

    start_time = time.time()
    model = agreement._load(agreement.DEFAULT_MODEL_PATH)
    construct_embedding_streaming(options, model)
    cumulative_time = time.time() - start_time
    print(f"Cumulative running time: {cumulative_time}")
