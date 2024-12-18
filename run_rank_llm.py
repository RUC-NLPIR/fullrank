import argparse
import os
import sys
import copy
import json
import time
import datetime
from typing import Any, Dict, List, Union, Optional, Sequence
from data import Query, Request, Candidate
import torch
from enum import Enum
from rerank.api_keys import get_openai_api_key
from rerank.rank_gpt import SafeOpenai
from rerank.rank_listwise_os_llm import RankListwiseOSLLM
from rerank.rankllm import PromptMode, RankLLM
from rerank.reranker import Reranker
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import FaissSearcher
from pyserini.search._base import get_topics, get_qrels
from utils import OutputFormat, get_output_writer, get_qrels_dl22, get_topics_dl22
from dataclasses import dataclass, field
from trec_eval import Eval
from index_and_topics import THE_TOPICS, THE_INDEX, THE_QRELS
from enum import Enum
from transformers import HfArgumentParser
from tqdm import tqdm
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# parent = os.path.dirname(SCRIPT_DIR)
# parent = os.path.dirname(parent)
# sys.path.append(parent)

os.environ["PYSERINI_CACHE"] = "cache"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

class RetrievalMode(Enum):
    DATASET = "dataset"
    CUSTOM = "custom"

    def __str__(self):
        return self.value

class RetrievalMethod(Enum):
    UNSPECIFIED = "unspecified"
    BM25 = "bm25"
    BM25_RM3 = "bm25_rm3"
    SPLADE_P_P_ENSEMBLE_DISTIL = "SPLADE++_EnsembleDistil_ONNX"
    D_BERT_KD_TASB = "distilbert_tas_b"
    OPEN_AI_ADA2 = "openai-ada2"
    REP_LLAMA = "rep-llama"
    CUSTOM_INDEX = "custom_index"

    def __str__(self):
        return self.value

@dataclass
class Arguments:
    # retrieval arguments
    datasets: List[str] = field(metadata={'help': 'List of test datasets.'})
    output: str = field(metadata={'help': 'Path to output file.'})
    output_format: Optional[str] = field(default='trec', metadata={'help': 'Output format.'})
    retrieval_method: RetrievalMethod = field(default=RetrievalMethod.BM25, metadata={'help': 'Method of retrieval.', 'choices': list(RetrievalMethod)})
    retrieval_num: int = field(default=100, metadata={'help': 'retrieval number'})
    rerank_topk: int = field(default=None, metadata={'help': 'only need to rerank top-k candidates'})
    threads: int = field(default=30, metadata={'help': 'Number of threads.'})
    batchsize_retrieval: int = field(default=32, metadata={'help': 'batchsize for dense retrieval'})
    remove_query: Optional[bool] = field(default=True, metadata={'help': 'Remove query from output.'})
    first_stage_model: str = field(default='bm25', metadata={'help': 'the model used for prividing the first stage ranking results'})
    save_first_stage_run: Optional[bool] = field(default=True, metadata={'help': 'Save first-stage run.'})
    remove_duplicates: Optional[bool] = field(default=False, metadata={'help': 'Remove duplicates from output.'})
    shuffle_candidates: bool = field(default=False, metadata={'help': 'Whether to shuffle the candidates before reranking.'})

    # llm arguments
    model_path: str = field(default=f'llm/rank_vicuna_7b_v1', metadata={'help': 'Path to the model. If `use_azure_ai`, pass your deployment name.'})
    use_azure_openai: bool = field(default=False, metadata={'help': 'If True, use Azure OpenAI. Requires env var to be set: `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_API_BASE`'})
    context_size: int = field(default=4096, metadata={'help': 'context size used for model.'})
    num_gpus: int = field(default=1, metadata={'help': 'the number of GPUs to use'})
    cache_dir: Optional[str] = field(default='../../cache', metadata={'help': 'Path to cache directory.'})
    llm_dtype: str = field(default='bf16', metadata={'help': 'Data type of llm.'})
    batch_size: int = field(default=32, metadata={'help': 'inference batchsize'})
    variable_passages: bool = field(default=False, metadata={'help': 'Whether the model can account for a variable number of passages in input.'})
    num_passes: int = field(default=1, metadata={'help': 'Number of passes to run the model.'})
    window_size: int = field(default=20, metadata={'help': 'Window size for the sliding window approach.'})
    max_passage_length: int = field(default=80, metadata={'help': 'maximum words for passages'})
    step_size: int = field(default=10, metadata={'help': 'Step size for the sliding window approach.'})
    vllm_batched: bool = field(default=False, metadata={'help': 'Whether to run the model in batches.'})
    prompt_mode: PromptMode = field(default=PromptMode.RANK_GPT, metadata={'required': True, 'choices': list(PromptMode)})
    system_message: str = field(default='You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.', metadata={'help': 'the system message used in prompts'})
    notes: str = field(default='', metadata={'help': 'notes for code running'})



def write_run(output_writer, results, args):
    with output_writer:
        for request in results:
            qid = request.query.qid
            hits = request.candidates
            if args.remove_duplicates:
                seen_docids = set()
                dedup_hits = []
                for hit in hits:
                    if hit.docid.strip() in seen_docids:
                        continue
                    seen_docids.add(hit.docid.strip())
                    dedup_hits.append(hit)
                hits = dedup_hits

            # For some test collections, a query is doc from the corpus (e.g., arguana in BEIR).
            # We want to remove the query from the results.
            if args.remove_query:
                hits = [hit for hit in hits if hit.docid != qid]

            # write results
            output_writer.write(qid, hits)

def evaluate_results(args, dataset, out_path, qrels, time_cost, current_pass, total_input_tokens, total_output_tokens):
    # all_metrics = Eval(out_path, THE_QRELS[args.dataset])
    all_metrics = Eval(out_path, qrels)
    print(f'###################### {dataset} ######################')
    print(all_metrics)
    print(f'time_cost: {time_cost}')
    result = {'model_path': args.model_path,
              'datetime': str(datetime.datetime.now()),
              'retrieval_num': args.retrieval_num,
              'current_pass': current_pass,
              'window_size': args.window_size,
              'step_size': args.step_size,
              'shuffle_candidates': args.shuffle_candidates,
              'time_cost': time_cost,
              'total_input_tokens': total_input_tokens,
              'total_output_tokens': total_output_tokens,
              'notes': args.notes,
              **all_metrics}
    os.makedirs('results/', exist_ok=True)
    result_path = f'results/{dataset}.json'
    if os.path.exists(result_path) == False:
        with open(result_path, 'w') as f: 
            json.dump([], f, indent=4)
    with open(result_path, 'r') as f:
        json_data = json.load(f)
        json_data.append(result)
    with open(result_path, 'w') as f: 
        json.dump(json_data, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser(Arguments)
    _args, *_ = parser.parse_args_into_dataclasses()
    args = argparse.Namespace(**vars(_args))

    ############################## retrieval for each dataset ##############################
    dataset_qrels = {}
    dataset_topics = {}
    dataset_results = {}
    for dataset in args.datasets:
        index_path = THE_INDEX[dataset]
        topics_path = THE_TOPICS[dataset]
        qrels_path = THE_QRELS[dataset]

        searcher = LuceneSearcher.from_prebuilt_index(index_path)
        topics = {str(qid): content['title'] for qid, content in get_topics(topics_path).items()} if dataset != 'dl22' else get_topics_dl22() # dl22 is not supported by pyserini 0.20.0
        qrels = {str(qid): {str(docid): int(score) for docid, score in docs.items()} 
                for qid, docs in get_qrels(qrels_path).items()} if dataset != 'dl22' else get_qrels_dl22()
        dataset_qrels[dataset] = qrels
        dataset_topics[dataset] = topics
        batch_topic_ids = []
        batch_topics = []
        for topic_id in list(topics.keys()):
            if topic_id in qrels:
                batch_topic_ids.append(topic_id)
                batch_topics.append(topics[topic_id])
        if args.first_stage_model in ['bm25']:
            first_state_run_path = f'runs/{dataset}/{args.retrieval_method}_top{args.retrieval_num}.txt'
        else:
            first_state_run_path = f'runs/{dataset}/{args.first_stage_model}.txt'
        if os.path.exists(first_state_run_path):
            print(f'Loading first stage run from {first_state_run_path}.')
            results = []
            with open(first_state_run_path, 'r') as f:
                current_qid = None
                current_ranking = []
                for line in f:
                    qid, _, docid, _, score, _ = line.strip().split()
                    if qid != current_qid:
                        if current_qid is not None:
                            current_query = Query(qid=current_qid, text=topics[current_qid])
                            results.append(Request(query=current_query, candidates=current_ranking[:args.retrieval_num]))
                        current_ranking = []
                        current_qid = qid
                    current_ranking.append(Candidate(docid=docid, score=float(score), doc=json.loads(searcher.doc(docid).raw())))
                # results.append((current_qid, current_ranking[:args.retrieval_num]))
                current_query = Query(qid=current_qid, text=topics[current_qid])
                results.append(Request(query=current_query, candidates=current_ranking[:args.retrieval_num]))
        else:
            print(f'First stage run on {dataset}...')
            _results = searcher.batch_search(batch_topics, batch_topic_ids, k=args.retrieval_num, threads=args.threads)
            results = []
            for topic_id in batch_topic_ids:
                candidates = [Candidate(docid=result.docid, score=result.score, doc=json.loads(searcher.doc(result.docid).raw())) for result in _results[topic_id]]
                results.append(Request(query=Query(qid=topic_id, text=topics[topic_id]), candidates=candidates))

            if args.save_first_stage_run:
                output_writer = get_output_writer(first_state_run_path, OutputFormat(args.output_format), 'w',
                                                max_hits=args.retrieval_num, tag=args.retrieval_method, topics=topics, )
                write_run(output_writer, results, args)
        dataset_results[dataset] = results
    ############################# load LLM reranker #############################
    if args.model_path in ['gpt-4o-mini-2024-07-18', 'gpt-4o-mini', 'gpt-4o-2024-08-06', 'gpt-4o-2024-05-13', 'gpt-4o']:
        agentclass = SafeOpenai
        agent = agentclass(
            args=args,
            model=args.model_path,
            context_size=args.context_size,
            prompt_mode=args.prompt_mode,
            window_size=args.window_size,
            keys=get_openai_api_key(model_name=args.model_path),
        )
    else:
        print(f"Loading {args.model_path} ...")
        agent = RankListwiseOSLLM(
            args=args,
            model=args.model_path,
            context_size=args.context_size,
            prompt_mode=args.prompt_mode,
            num_gpus=args.num_gpus,
            variable_passages=args.variable_passages,
            window_size=args.window_size,
            system_message=args.system_message, # need to change
            vllm_batched=args.vllm_batched,
            max_passage_length=args.max_passage_length,
        )
    reranker = Reranker(agent)
    ###################################### Reranking ######################################
    for dataset in args.datasets:
        print(f"########################## Reranking on {dataset} ##########################")
        qrels = dataset_qrels[dataset]
        topics = dataset_topics[dataset]
        results = dataset_results[dataset]
        total_time_cost = 0
        for pass_ct in range(args.num_passes):
            print(f"Pass {pass_ct + 1} of {args.num_passes}:")
            # in case of the number of candidate passages < 100 (for sliding windows strategy)
            results_with_100_passages = []
            results_less_100_passages = []
            for result in results:
                if len(result.candidates) < args.retrieval_num:
                    results_less_100_passages.append([result])
                else:
                    results_with_100_passages.append(result)
            # results_grouped = [results_with_100_passages] + results_less_100_passages
            results_grouped = [results_with_100_passages[i:i+args.batch_size] for i in range(0, len(results_with_100_passages), args.batch_size)] + results_less_100_passages
            reranked_results = []
            # for batch in tqdm(results_grouped, desc=f'{dataset}'):
            for batch in results_grouped:
                reranked_batch, time_cost = reranker.rerank_batch(
                                    batch,
                                    rank_end=args.retrieval_num if args.rerank_topk is None else args.rerank_topk,
                                    window_size=min(args.window_size, len(batch[0].candidates)),
                                    shuffle_candidates=args.shuffle_candidates,
                                    step=args.step_size,
                                    vllm_batched=args.vllm_batched,
                                )
                reranked_results.extend(reranked_batch)
                total_time_cost += time_cost
            # save results and evaluate
            out_path = os.path.join(f'runs/{dataset}', args.output[:-4] + f'_top{args.retrieval_num}_passnum={args.num_passes}.txt')
            output_writer = get_output_writer(out_path, OutputFormat(args.output_format), 'w',
                                                max_hits=args.retrieval_num, tag=args.retrieval_method, topics=topics, )
            write_run(output_writer, reranked_results, args)
            total_input_tokens = sum(summary.input_token_count for result in reranked_results for summary in result.ranking_exec_summary)
            total_output_tokens = sum(summary.output_token_count for result in reranked_results for summary in result.ranking_exec_summary)
            evaluate_results(args, dataset, out_path, qrels, time_cost = total_time_cost, current_pass=pass_ct+1, total_input_tokens=total_input_tokens, total_output_tokens=total_output_tokens)
            if args.num_passes > 1:
                results = [
                    Request(copy.deepcopy(r.query), copy.deepcopy(r.candidates))
                    for r in reranked_results
                ]
        print(f"Reranking with {args.num_passes} passes complete!")
    # agent.close()