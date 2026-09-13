#########################################################################################################
##################### Demo data
#########################################################################################################

# This data is used in some of the tutorials for Reexpress two and the reexpress_sdm python package.
# The embeddings are from mlx-community/gemma-4-31b-it-4bit, and are constructed from the final-layer
# hidden states:
# max-pool over the sequence :: mean-pool over the sequence :: hidden-state of the final token (that estimates Yes | No)
# Embeddings are 3 x 5376 = 16128 dims.

# Note that Reexpress two and the reexpress_sdm python package are independent of the choice of
# input embeddings; this is just one example. Feel free to substitute another model, or use another
# means of extracting the hidden-states at the desired layer/resolution. Here we use MLX since
# Reexpress two runs on macOS, but note that the reexpress_sdm python package has a torch backend that can
# run on mps, cpu, or cuda (and possibly other supported torch devices).

# This has been Tested on an M2 Ultra 76 core 128 GB Mac Studio.

# Substitute your local file paths, where applicable, in the below.

#########################################################################################################
##################### Install dependencies
#########################################################################################################

conda create -n re_python_mlx_v1 python=3.12

conda activate re_python_mlx_v1

cd documentation/tutorials/data/code  # choose applicable path to the repo directory

pip install mlx-lm==0.31.3

#########################################################################################################
##################### Download data
#########################################################################################################

# change paths as desired

mkdir /Users/a/Documents/projects/sdm_paper_extension/data/classification
cd /Users/a/Documents/projects/sdm_paper_extension/data/classification

# Download data (here, we'll just use the copy with embeddings from Mixtral-8x7b, which we'll discard)
wget https://github.com/ReexpressAI/sdm_activations/releases/download/v1.0.0/factcheck_mixtral_8x7b.zip

#########################################################################################################
##################### Process Factcheck data
##################### Note that in this case the embedding also includes a max-pool.
#########################################################################################################

conda activate re_python_mlx_v1

cd documentation/tutorials/data/code  # choose applicable path to the repo directory

export HF_HOME=/Users/a/Documents/projects/hf_models/models_cache # choose applicable HF cache directory for the Gemma model

DATA_DIR="/Users/a/Documents/projects/sdm_paper_extension/data/classification/factcheck_mixtral_8x7b"

MODEL_LABEL="gemma_4_31b_it_4bit"
OUTPUT_DIR="/Users/a/Documents/projects/sdm_paper_extension/data/classification/factcheck_${MODEL_LABEL}"
mkdir -p ${OUTPUT_DIR}

SOURCE_MODEL_NAME="mixtral_8x7b"

# Unlike the data in https://github.com/ReexpressAI/sdm_activations/releases/download/v1.0.0, we make the shuffled data have a unique id relative to the unshuffled versions, since Reexpress two does not allow duplicate document id's in a single project.

for INPUT_FILE_NAME in "ood_eval" "calibration" "train" "ood_eval.ood_random_shuffle"; do
echo ${INPUT_FILE_NAME}
echo ${OUTPUT_DIR}/${INPUT_FILE_NAME}.${MODEL_LABEL}.logs.txt
EXTRA_ARGS=()
if [[ "${INPUT_FILE_NAME}" == "ood_eval.ood_random_shuffle" ]]; then
    EXTRA_ARGS=(--add_shuffled_prefix)
fi
python -u add_gemma_4_31b_it_4bit_embeddings.py \
--input_file=${DATA_DIR}/${INPUT_FILE_NAME}.${SOURCE_MODEL_NAME}.jsonl \
--class_size=2 \
--dataset="factcheck" \
--output_file=${OUTPUT_DIR}/${INPUT_FILE_NAME}.${MODEL_LABEL}.jsonl \
"${EXTRA_ARGS[@]}" > ${OUTPUT_DIR}/${INPUT_FILE_NAME}.${MODEL_LABEL}.logs.txt 2>&1
done
