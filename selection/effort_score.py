import os

from typing import List
from sampling import GraphDensitySampler

import torch
import pickle
from transformers import AutoModelForCausalLM,AutoTokenizer,AutoModelForQuestionAnswering
from peft import PeftConfig
from datasets import load_dataset
import torch
from effort_baseline_utils import Effort_Trainer,save_feature,load_feature,save_time_cost
import itertools
import time


from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PeftModel
)
from transformers import LlamaTokenizer  # noqa: F402
import math
import random
from datasets import load_dataset,load_from_disk
import json
import argparse
from tqdm import trange
import numpy as np
from transformers import pipeline

def get_moderate_index(data_score,rate=0.1):
    # 2023_ICLR_Moderate-DS/README.md
    low = 0.5 - rate / 2
    high = 0.5 + rate / 2
    sorted_idx = data_score.argsort()
    low_idx = round(data_score.shape[0] * low)
    high_idx = round(data_score.shape[0] * high)
    
    ids = np.concatenate((sorted_idx[:low_idx], sorted_idx[high_idx:]))
    return ids

def get_ccs_index(data_score,rate=0.1):
    # from Coverage-centric-coreset-selection/generate_importance_score.py
    stratas = 50
    score = data_score
    total_num = int(len(data_score)*rate)

    min_score = torch.min(score)
    max_score = torch.max(score) * 1.0001
    step = (max_score - min_score) / stratas

    def bin_range(k):
        return min_score + k * step, min_score + (k + 1) * step

    strata_num = []
    ##### calculate number for each strata #####
    for i in range(stratas):
        start, end = bin_range(i)
        num = torch.logical_and(score >= start, score < end).sum()
        strata_num.append(num)

    strata_num = torch.tensor(strata_num)

    def bin_allocate(num, bins):
        sorted_index = torch.argsort(bins)
        sort_bins = bins[sorted_index]

        num_bin = bins.shape[0]

        rest_exp_num = num
        budgets = []
        for i in range(num_bin):
            rest_bins = num_bin - i
            avg = rest_exp_num // rest_bins
            cur_num = min(sort_bins[i].item(), avg)
            budgets.append(cur_num)
            rest_exp_num -= cur_num


        rst = torch.zeros((num_bin,)).type(torch.int)
        rst[sorted_index] = torch.tensor(budgets).type(torch.int)

        return rst

    budgets = bin_allocate(total_num, strata_num)

    ##### sampling in each strata #####
    selected_index = []
    sample_index = torch.arange(data_score.shape[0])

    for i in range(stratas):
        start, end = bin_range(i)
        mask = torch.logical_and(score >= start, score < end)
        pool = sample_index[mask]
        rand_index = torch.randperm(pool.shape[0])
        selected_index += [idx.item() for idx in pool[rand_index][:budgets[i]]]

    return selected_index

def get_feature(base_model,dataset):
    feature_extractor = pipeline("feature-extraction", framework="pt", model=base_model)

    if not os.path.exists(dataset):
        data = load_dataset(dataset)
    else:
        data = load_from_disk(dataset)
    train_data=data.train_test_split(test_size=0.1, seed=42)['train']#factory split

    feature_list=[]
    target_list=[]
    for i in trange(len(train_data)):
        _tdata=train_data[i]
        # Extract features
        # labels=torch.Tensor(_tdata['labels'])
        mask = [label != -100 for label in _tdata['labels']]  # Create a boolean mask
        target = [label for label, m in zip(_tdata['labels'], mask) if m]
        # mask = labels != -100
        # target = labels[mask].cpu().numpy().tolist()
        feature = feature_extractor(feature_extractor.tokenizer.decode(_tdata['input_ids']), return_tensors="pt")[0]#.numpy().mean(axis=0)
        mask.append(False)
        feature=feature[mask]
        # if feature_list==None:
        #     feature_list=feature.cpu()
        #     target_list=target
        # else:
        feature_list.append(feature.cpu())#torch.cat((feature_list,feature.cpu()),dim=0)
        target_list.append(target)#.extend(target)
    return feature_list,target_list

def get_Moderate_score(raw_feature_list, raw_target_list):
    feature_list=torch.cat(raw_feature_list,dim=0)
    feature_list=feature_list.detach().numpy()
    target_list=np.array(list(itertools.chain(*raw_target_list)))
    classes_list=np.unique(target_list, axis=0)
    num_classes = len(classes_list)
    prot = np.zeros((num_classes, feature_list.shape[-1]))
    for i in trange(num_classes):
        prot[i] = np.median(feature_list[(target_list == classes_list[i]).nonzero(), :].squeeze(), axis=0, keepdims=False)
    prots_for_each_example = np.zeros(shape=(len(raw_feature_list), prot.shape[-1]))
    for i in trange(len(raw_feature_list)):
        for j in range(len(raw_target_list[i])):
            _index=[np.where(classes_list==_token)[0][0] for _token in raw_target_list[i]]
            prots_for_each_example[i, :] = np.sum(prot[_index,:])
    score_list=np.linalg.norm(raw_feature_list - prots_for_each_example, axis=1)
    return score_list

def get_D2_index(raw_features,data_score,rate=0.1,mis_ratio=0.4):
    def mislabel_mask(data_score, mis_num, mis_descending):
        mis_score = data_score
        mis_score_sorted_index = mis_score.argsort(descending=mis_descending)
        easy_index = mis_score_sorted_index[mis_num:]
        return easy_index
    
    total_num = len(score_list)
    coreset_num = int(rate * total_num)

    mis_num = int(mis_ratio * total_num)
    score_index = mislabel_mask(data_score,
                mis_num=mis_num,
                mis_descending=True)

    # np.save('./tmp_score_index_aum_%s.npy' % rate, score_index)# TODO: comment
    features=np.array([_feature.numpy().mean(axis=0) for _feature in raw_features])
    print(1)

    sampling_method = GraphDensitySampler(X=features,
            y=None,
            gamma=0.1,
            seed=42,
            importance_scores=data_score)
    coreset_index = sampling_method.select_batch_(coreset_num)
    # coreset_index = score_index[coreset_index]
    # graph_scores = sampling_method.starting_density
    return coreset_index

def get_baseline_score(base_model='YOUR_DIR/test_code/wmt-gemma-7b',# 'llama-160m'#'Llama-2-7b-hf'#'test_quote-llama27'#'test_quote-llamam160'#'opt-125m' # 'opt-2.7b'
    dataset='wmt',#'Abirate/english_quotes',
    tokenizer_path='YOUR_DIR/models/gemma-7b',
    finetune=False,
    method=['Effort']):
    # base_model = 'YOUR_DIR/models/Llama-2-7b-hf'
    train_data_path = dataset
    cutoff_len = 200
    lora_r = 81
    lora_alpha = 16
    lora_dropout = 0
    lora_target_modules = ["q_proj","v_proj",]
    # other hyperparams
    group_by_length = False
    resume_from_checkpoint = None
    gradient_accumulation_steps = 1
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    os.environ["WANDB_DISABLED"] = "true"
    if finetune:
        config = PeftConfig.from_pretrained(base_model)
        if 'QA' in base_model:
            raw_model = AutoModelForQuestionAnswering.from_pretrained(config.base_model_name_or_path, device_map="auto")
        else:
            raw_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto")
        model = PeftModel.from_pretrained(raw_model, base_model)
        model = prepare_model_for_int8_training(model,use_gradient_checkpointing=False)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,padding_side="left")
    else:
        if 'QA' in base_model:
            model = AutoModelForQuestionAnswering.from_pretrained(# AutoModelForCausalLM.from_pretrained(#Modified_LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=True,
                # torch_dtype=torch.float32,
                device_map=device_map,
            )
            
        else:
            model = AutoModelForCausalLM.from_pretrained(# AutoModelForCausalLM.from_pretrained(#Modified_LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map=device_map,
            )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,padding_side="left")

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="QUESTION_ANS" if 'QA' in base_model else 'CAUSAL_LM',
    )

    model = prepare_model_for_int8_training(model,use_gradient_checkpointing=False)
    model = get_peft_model(model, config)

    tokenizer.pad_token_id = (
        0  )
    # tokenizer.pad_token = tokenizer.eos_token # will lead to nan gradient. expired
    tokenizer.padding_side = "left" 
    
    import transformers
    
    if not os.path.exists(train_data_path):
        data = load_dataset(train_data_path)
    else:
        data = load_from_disk(train_data_path)

    train_data=data.train_test_split(test_size=0.1, seed=42)['train']#factory split

    model.is_parallelizable = True
    model.model_parallel = True

    trainer = Effort_Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=train_data,
        args=transformers.TrainingArguments(
            output_dir='./effort_output',
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=1,
            learning_rate=0,
            fp16=False,
            logging_strategy="no",
            optim="adamw_torch",
            save_strategy="no",
            ddp_find_unused_parameters=None,
            group_by_length=group_by_length,
            report_to=None,
            gradient_checkpointing=False
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        # data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    if 'Llama-2-7b' in tokenizer_path:
        model_type='llama27'
    elif 'Llama-2-13b' in tokenizer_path:
        model_type='llama213'
    elif 'gemma-2' in tokenizer_path:
        model_type='gemma'
    elif 'gemma-7' in tokenizer_path:
        model_type='gemma7'
    elif 'Mistral-' in tokenizer_path:
        if 'Nemo' in tokenizer_path:
            model_type='llama213'
        else:
            model_type='llama27'
    # freeze all layers except for the last layer of Lora
    trainer.freeze_layers(model_type)
    # TODO: LLama2-7b (opt 2.7b): layers.31.self_attn.v_proj.lora_B.default.weight
    # llama: LLama-160m (opt 125m): layers.11.self_attn.v_proj.lora_B.default.weight

    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    model = torch.compile(model)

    # calculate effort score for each sample
    trainer.set_attribte()
    trainer.get_sample_grad = True
    trainer.base_dir=base_model
    trainer.task_name=args.task
    trainer.time_cost={k:0 for k in method}
    trainer.model.base_model.model.get_sample_loss = True
    score_dict = trainer.get_grad(resume_from_checkpoint=resume_from_checkpoint,method=method)

    save_time_cost(trainer.time_cost,base_model,args.task)
    if 'D2' in method or 'Moderate' in method:
        save_feature(score_dict,base_model,args.task)
        return {}
    score_dict['score_norm']={}
    for key in score_dict['score'].keys():
        effort=score_dict['score'][key]
        score_dict['score_norm'][key]=(effort-torch.min(effort))/(torch.max(effort)-torch.min(effort))
    # if method=='None':
    #     all_gradients = torch.cat([_[0].cpu().unsqueeze(0) for _ in gradients], dim=0)
    #     effort = torch.norm(all_gradients, dim=1)
    # else:
    #     effort=gradients
    # effort_norm=(effort-torch.min(effort))/(torch.max(effort)-torch.min(effort))
    # return effort_norm,effort
    return score_dict

if __name__=="__main__":# 
    parser = argparse.ArgumentParser("", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-md", "--base_model", default='YOUR_DIR/wmt_gemma2-factory-sft-select', type=str, help="Input file")
    # YOUR_DIR/test_code/gemma_result/2_samsum/samsum_gemma7-factory-sft(dealrec)
    parser.add_argument("-td", "--tokenizer", default='YOUR_DIR/models/gemma-2b', type=str, help="Input file")
    parser.add_argument("-bs", "--dataset", default='YOUR_DIR/LLaMA-Factory/data/wmt_dataset-gemma-2b', type=str, help="")
    parser.add_argument("-tk", "--task", default='wmt', type=str, help="")
    parser.add_argument("-ft", "--finetune", default='True', type=str, help="True or False")
    parser.add_argument("-sr", "--selection_rate", default=[0.1,0.2,0.3,0.5,0.8], type=float, help="")
    parser.add_argument("-bl", "--baseline", default="['Effort']", type=str, help="None/EL2N/CCS")
    args=parser.parse_args()
    args.baseline=eval(args.baseline)
    base_model=args.base_model
    tokenizer=args.tokenizer
    dataset=args.dataset
    finetune=eval(args.finetune)
    if not finetune:
        base_model=args.tokenizer
    # True
    save_dir=os.path.join(base_model,f'selected-{args.task}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    

    scores_path=os.path.join(base_model,f'scores-{args.task}.pkl')
    feature_path=os.path.join(base_model,f'feature_{args.task}.pt')
    if ('D2' in args.baseline or 'Moderate' in args.baseline) and not os.path.exists(feature_path):# only get feature and target
        _=get_baseline_score(base_model=base_model,dataset=dataset,tokenizer_path=tokenizer,finetune=finetune,method=args.baseline)
        with open(scores_path, 'rb') as f:
            result = pickle.load(f)
    else:
        if not os.path.exists(scores_path):
            result=get_baseline_score(base_model=base_model,dataset=dataset,tokenizer_path=tokenizer,finetune=finetune,method=args.baseline)
            # result is a dict containing: score_norm,score. each one is a dict whose keys are args.baseline
            with open(scores_path, 'wb') as f:
                pickle.dump(result, f)
        else:
            with open(scores_path, 'rb') as f:
                result = pickle.load(f)
    
    if ('D2' in args.baseline and 'D2' not in result['score'].keys()) or ('Moderate' in args.baseline and 'Moderate' not in result['score'].keys()):
        if not os.path.exists(scores_path):
            score_dict={}
            score_dict['score']={}
            score_dict['score_norm']={}
        else:
            with open(scores_path, 'rb') as f:
                score_dict = pickle.load(f)
        feature_path=os.path.join(base_model,f'feature_{args.task}.pt')
        target_path=os.path.join(base_model,f'target_{args.task}.pt')
        if not os.path.exists(feature_path):
            print(f'Error! No feature file in {feature_path}')
        else:
            features,target=load_feature(base_model,args.task)

        if 'D2' in args.baseline:
            score_list=score_dict['score']['EL2N']
            for rate in args.selection_rate:
                start_time=time.time()
                if rate>=0.5:
                    mis_ratio=0
                else:
                    mis_ratio=0.4
                index_list=get_D2_index(features,score_list,rate=rate,mis_ratio=mis_ratio)
                save_path=os.path.join(save_dir,"D2-{}.pt".format(rate))
                if not os.path.exists(save_path):
                    torch.save(index_list, save_path)
                time_cost=time.time()-start_time
                save_time_cost(time_cost,base_model,args.task,"D2-{}".format(rate))
        with open(scores_path, 'wb') as f:
            pickle.dump(result, f)



    # for rate in args.selection_rate:
    #     hard_prune=float(rate)
    #     # n_fewshot=max(512,int(len(effort_norm)*hard_prune))
    #     if 'EL2N' in args.baseline:
    #         start_time=time.time()
    #         scores_sorted, indices = torch.sort(result['score_norm']['EL2N'], descending=True)
    #         n_prune = math.floor(hard_prune * len(scores_sorted))
    #         coreset = list(indices[:n_prune].numpy())
    #         save_path=os.path.join(save_dir,"EL2N-{}.pt".format(hard_prune))#effort_data_{len(coreset)}.pt
    #         if not os.path.exists(save_path):
    #             torch.save(coreset, save_path)
    #         time_cost=time.time()-start_time
    #         save_time_cost(time_cost,base_model,args.task,"EL2N-{}".format(rate))

    #         start_time=time.time()
    #         scores_sorted, indices = torch.sort(result['score_norm']['Effort'], descending=True)
    #         n_prune = math.floor(hard_prune * len(scores_sorted))
    #         coreset = list(indices[:n_prune].numpy())
    #         save_path=os.path.join(save_dir,"GraNd-{}.pt".format(hard_prune))#effort_data_{len(coreset)}.pt
    #         if not os.path.exists(save_path):
    #             torch.save(coreset, save_path)
    #         time_cost=time.time()-start_time
    #         save_time_cost(time_cost,base_model,args.task,"Effort-{}".format(rate))
    #     if 'CCS-EL2N' in args.baseline:
    #         start_time=time.time()
    #         coreset=get_ccs_index(result['score_norm']['EL2N'],rate=hard_prune)
    #         save_path=os.path.join(save_dir,"CCS-EL2N-{}.pt".format(hard_prune))#effort_data_{len(coreset)}.pt
    #         if not os.path.exists(save_path):
    #             torch.save(coreset, save_path)
    #         time_cost=time.time()-start_time
    #         save_time_cost(time_cost,base_model,args.task,"CCS-EL2N-{}".format(rate))
    #     if 'Moderate' in args.baseline:
    #         start_time=time.time()
    #         coreset=get_moderate_index(result['score_norm']['Moderate'],rate=hard_prune)
    #         save_path=os.path.join(save_dir,"Moderate-{}.pt".format(hard_prune))#effort_data_{len(coreset)}.pt
    #         if not os.path.exists(save_path):
    #             torch.save(coreset, save_path)
    #         time_cost=time.time()-start_time
    #         save_time_cost(time_cost,base_model,args.task,"Moderate-{}".format(rate))
    #     if 'Entropy' in args.baseline:
    #         start_time=time.time()
    #         scores_sorted, indices = torch.sort(result['score_norm']['Entropy'], descending=True)
    #         n_prune = math.floor(hard_prune * len(scores_sorted))
    #         coreset = list(indices[:n_prune].numpy())
    #         save_path=os.path.join(save_dir,"Entropy-{}.pt".format(hard_prune))#effort_data_{len(coreset)}.pt
    #         if not os.path.exists(save_path):
    #             torch.save(coreset, save_path)
    #         time_cost=time.time()-start_time
    #         save_time_cost(time_cost,base_model,args.task,"Entropy-{}".format(rate))
    # print(1)
    
