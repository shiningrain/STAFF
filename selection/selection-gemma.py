import os
import argparse
import torch
import pickle
import math
import numpy as np
import copy
import random


def get_core_set(overall,effort_7b,method='H',rate=0.1, verify_sample_rate=None):

    hard_prune=rate
    k=50
    n_fewshot=int(len(overall)*hard_prune)
    scores_sorted, indices = torch.sort(overall, descending=True)

    n_prune = math.floor(0.1 * len(scores_sorted))

    scores_sorted = scores_sorted[n_prune:]
    indices = indices[n_prune:]
    s_max = torch.max(scores_sorted)
    s_min = torch.min(scores_sorted)
    interval = (s_max - s_min) / k
    s_split = [min(s_min + (interval * _), s_max)for _ in range(1, k+1)]

    score_split = [[] for _ in range(k)]
    for idxx, s in enumerate(scores_sorted):
        for idx, ref in enumerate(s_split):
            if s.item() <= ref:
                score_split[idx].append({indices[idxx].item():s.item()})
                break
    
    verify_set=[]
    budget_list=[]
    m = n_fewshot
    verify_score_split=copy.deepcopy(score_split)
    while len(verify_score_split):
        # select the group with fewest samples
        group = sorted(verify_score_split, key=lambda x:len(x))
        group = [strat for strat in group if len(strat)]
        # random select and add to the fewshot indices list
        fewest = group[0]
        sorted_data = sorted(fewest, key=lambda x: list(x.values())[0],reverse=True)

        if verify_sample_rate==None:
            _tmp_budget=10
        else:
            _tmp_budget=int(len(group[0])*0.1)
        verify_budget = min(len(group[0]), _tmp_budget)
        budget = min(len(group[0]), math.floor(m/len(group)))
        verify_idx = random.sample([list(_.keys())[0] for _ in sorted_data], verify_budget)
        selected_idx = random.sample([list(_.keys())[0] for _ in fewest], budget)
        budget_list.append(budget)
        if len(group[0])<=_tmp_budget:
            verify_set.append([])
        else:
            verify_set.append(verify_idx)
        verify_score_split = group[1:]
        m = m - len(selected_idx)

    # reweight budget list based on LLM feedback.
    overall=overall.tolist()
    effort_7b_score_list=[]
    for i in range(len(verify_set)):
        if verify_set[i]==[]:
            effort_7b_score_list.append(None)
        else:
            effort_7b_score_list.append(sum([effort_7b[_value] for _value in verify_set[i]])/sum([overall[_value] for _value in verify_set[i]]))


    coreset = []
    m = n_fewshot
    count=0
    while len(score_split):
        # select the group with fewest samples

        group = sorted(score_split, key=lambda x:len(x))
        group = [strat for strat in group if len(strat)]
        # random select and add to the fewshot indices list
        fewest = group[0]
        sorted_data = sorted(fewest, key=lambda x: list(x.values())[0],reverse=True)

        assert len(effort_7b_score_list)==len(group)

        if effort_7b_score_list[0]==None:
            budget = min(len(group[0]), math.floor(m/len(group)))#math.floor(scale_list[i]*m/len(group))
        else:
            budget = min(len(group[0]), math.floor(m*effort_7b_score_list[0]/sum(effort_7b_score_list)))
        if 'R' in method:
            selected_idx = random.sample([list(_.keys())[0] for _ in sorted_data], budget)
        else:
            selected_idx = [list(_.keys())[0] for _ in sorted_data[:budget]]
        coreset.extend(selected_idx)
        count+=1
        # remove the fewest group
        score_split = group[1:]
        effort_7b_score_list=effort_7b_score_list[1:]
        m = m - len(selected_idx)
    if m>0:
        print(m)
        filtered_list = [item for item in indices.tolist() if item not in coreset]
        sampled_data = random.sample(filtered_list, m)
        coreset.extend(sampled_data)
    elif m<0:
        coreset=coreset[:m]
    return coreset

if __name__=="__main__":
    parser = argparse.ArgumentParser("", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-sd", "--save_dir", default='./selected_data/gemma/wmt_selected', type=str, help="")
    parser.add_argument("-sm", "--small_model_dir", default='YOUR_DIR/wmt_gemma2-factory-sft', type=str, help="")
    parser.add_argument("-lm", "--large_model_dir", default='YOUR_DIR/models/gemma-7b', type=str, help="")
    parser.add_argument("-tk", "--task", default='wmt', type=str, help="")
    args=parser.parse_args()


    save_dir=args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rate_list=[0.1,0.2,0.3,0.5,0.8]

    effort_2b_path=os.path.join(args.small_model_dir,f'scores-{args.task}.pkl')
    effort_7b_path=os.path.join(args.large_model_dir,f'scores-{args.task}.pkl')

    with open(effort_2b_path, 'rb') as f:
        effort_2b = pickle.load(f)
    effort_2b_norm=effort_2b['score_norm']['Effort']
    if os.path.exists(effort_7b_path):
        with open(effort_7b_path, 'rb') as f:
            effort_7b = pickle.load(f)
        effort_7b_score=effort_7b['score_norm']['Effort'].tolist()
    else:
        # verification in runtime #TODO
        print('Please calculate effort score first')
        os._exit(0)


    for rate in rate_list:
        save_path=os.path.join(save_dir,'score-{}.pt'.format(rate))
        if not os.path.exists(save_path):
            index_list=get_core_set(effort_2b_norm, effort_7b_score,rate=rate)
            torch.save(index_list, save_path)
