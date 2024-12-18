export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_LAUNCH_BLOCKING=1

# ------------------------------- 1. zero-shot setting -------------------------------
# ------------------------------- 1.1 open-source models ------------------------------
model_name=Mistral-7B-Instruct-v0.3
DATASETS=('dl19')
# DATASETS=('dl19' 'dl20' 'covid' 'dbpedia' 'nfcorpus' 'robust04' 'scifact' 'signal' 'news' 'touche')
for window_size in 20 100
do
    python run_rank_llm.py \
        --model_path llm/$model_name \
        --window_size $window_size \
        --step_size 10 \
        --retrieval_num 100 \
        --datasets ${DATASETS[@]} \
        --retrieval_method bm25 \
        --shuffle_candidates False \
        --prompt_mode rank_GPT \
        --context_size 32768 \
        --variable_passages \
        --vllm_batched True \
        --batch_size 10000 \
        --output "${model_name}.txt" \
        --num_gpus 4 \
        --max_passage_length 100 \

done
# ------------------------------- 1.2 close-source models ------------------------------
# MODEL=gpt-4o-mini
MODEL=gpt-4o-2024-08-06
DATASETS=('dl19')
# DATASETS=('dl19' 'dl20' 'covid' 'dbpedia' 'nfcorpus' 'robust04' 'scifact' 'signal' 'news' 'touche')
for window_size in 20 100
do
    python run_rank_llm.py \
        --model_path=${MODEL} \
        --window_size $window_size \
        --step_size 10 \
        --retrieval_num 100 \
        --datasets ${DATASETS[@]} \
        --retrieval_method bm25 \
        --shuffle_candidates False \
        --prompt_mode rank_GPT \
        --context_size 32768 \
        --variable_passages \
        --batch_size 10000 \
        --output ${MODEL}.txt \
        --max_passage_length 100 \
        
done

# ------------------------------- 2. superivsed fine-tuning setting (RankMistral100) -------------------------------
window_size=100
model_name=RankMistral100
DATASETS=('dl19')
# DATASETS=('dl19' 'dl20' 'covid' 'dbpedia' 'nfcorpus' 'robust04' 'scifact' 'signal' 'news' 'touche')
python run_rank_llm.py \
    --model_path trained_models/$model_name \
    --window_size $window_size \
    --step_size 10 \
    --retrieval_num 100 \
    --datasets ${DATASETS[@]} \
    --retrieval_method bm25 \
    --shuffle_candidates False \
    --prompt_mode rank_GPT \
    --context_size 32768 \
    --variable_passages \
    --vllm_batched True \
    --batch_size 10000 \
    --output "${model_name}.txt" \
    --num_gpus 4 \
    --max_passage_length 100 \
