# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : new_gen.py
# Time       ：6/4/2024 3:28 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description：several finetuned models to generate medical knowledge graph
"""

import transformers
import torch
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import collections
from utils import load_pickle
config = {
    'KG_DATADIR': '/home/czhaobo/HyperHealth/data/ready/',

}
torch.cuda.set_device(7) # 设定cuda
torch.set_default_tensor_type(torch.cuda.HalfTensor)


class GPT2(object):
    """finetune with medical llm"""
    def __init__(self, choice=None):
        if choice=='bio':
            from modelscope import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("AI-ModelScope/BioMistral-7B")
            self.model = AutoModel.from_pretrained("AI-ModelScope/BioMistral-7B")
        elif choice == 'mmd':
            self.tokenizer = transformers.AutoTokenizer.from_pretrained("Henrychur/MMedLM2", trust_remote_code=True)
            self.model = transformers.AutoModelForCausalLM.from_pretrained("Henrychur/MMedLM2", torch_dtype=torch.float16,
                                                         trust_remote_code=True)
        else:
            self.tokenizer = transformers.LlamaTokenizer.from_pretrained('axiong/PMC_LLaMA_13B')#.to('cuda:6')
            self.model = transformers.LlamaForCausalLM.from_pretrained('axiong/PMC_LLaMA_13B').to('cuda:7')
        print('Initialize Done!')
    def chat_upgrade(self, prompt):
        batch = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False
        )
        with torch.no_grad():
            completion = self.model.generate(
                inputs=batch["input_ids"].to('cuda:7'),
                max_length=500,
                do_sample=True,
                top_k=10,
            )
        response = self.tokenizer.decode(completion[0]).strip()
        return response

def ehr_kg_prompting(category):
    """
    生成EHR知识图谱
    :return:
    """
    if category == 'condition':
        # example = """
        # Example:
        # prompt: systemic lupus erythematosus
        # updates: [[systemic lupus erythematosus, is an, autoimmune condition], [systemic lupus erythematosus, may cause, nephritis], [anti-nuclear antigen, is a test for, systemic lupus erythematosus], [systemic lupus erythematosus, is treated with, steroids], [methylprednisolone, is a, steroid]]
        # """
        example = """
        Example:
        prompt: Diabetes        
        updates: [
        [Diabetes, Increases_Risk_of, Cardiovascular_Complications],
        [Insulin, Common_Treatment_for, Diabetes],
        [Metformin, Recommended_for, Type_2_Diabetes],
        ]
        """
    elif category == 'procedure':
        example = """
        Example:
        prompt: endoscopy
        updates: [[endoscopy, is a, medical procedure], [endoscopy, used for, diagnosis], [endoscopic biopsy, is a type of, endoscopy], [endoscopic biopsy, can detect, ulcers]]
        """

    elif category == 'drug':
        example = """
        Example:
        prompt: iobenzamicacid
        updates: [[iobenzamicacid, is a, drug], [iobenzamicacid, may have, side effects], [side effects, can include, nausea], [iobenzamicacid, used as, X-ray contrast agent], [iobenzamicacid, formula, C16H13I3N2O3]]
        """
    return example

def save_triple_multi(params):
    """only test for chat ; openai version 0.28.1"""
    retry_count = 100
    retry_interval = 1
    code_map_entity, category, flag = params
    index, entity_name = code_map_entity
    term, category = entity_name, category
    for _ in range(retry_count):
        try:
            if flag == 'ehr_kg':
                example = ehr_kg_prompting(category)
                response = gpt.chat_upgrade(
                    f'''
                    Given a medical prompt, extrapolate as many relationships as possible of it and provide a list of updates.  
                    The relationships should be helpful for healthcare prediction (e.g., drug recommendation, mortality prediction, readmission prediction …) 
                    Each update should be exactly in format of [ENTITY 1, RELATIONSHIP, ENTITY 2]. The relationship is directed, so the order matters. 
                    Do this in both breadth and depth. Expand [ENTITY 1, RELATIONSHIP, ENTITY 2] until the list size reaches 50.

                    prompt: {term}
                    updates: 
                    '''
                ) #  {example} 添加到prompt中反而会又不好的影响

                data = {'index': index, 'entity_name': term, 'triple': response}
            return index, data

        except TimeoutError:
            print("任务执行超时：", index)
            print('重新请求....')
            retry_count += 1
            retry_interval *= 2  # 指数退避策略，每次重试后加倍重试间隔时间
            time.sleep(retry_interval)

        # except Exception as e:
        #     print("任务执行出错：", e)
        #     print('重新请求....')
        #     retry_count += 1
        #     retry_interval *= 2  # 指数退避策略，每次重试后加倍重试间隔时间
        #     time.sleep(retry_interval)

    return index, 'api请求失败'


def save_triple(code_map, category, flag='ehr_kg'):
    if flag == 'ehr_kg':
        df = pd.DataFrame(columns=['index', 'entity_name', 'triple'])

        for index, entity_name in code_map.items():
            term, category = entity_name, category
            print("Now processing: ", term)
            example = ehr_kg_prompting(category)
            response = gpt.chat_upgrade(
                # f'''
                # Given a prompt (a medical condition/procedure/drug), extrapolate as many relationships as possible of it and provide a list of updates.
                # The relationships should be helpful for healthcare prediction (e.g., drug recommendation, mortality prediction, readmission prediction …)
                # Each update should be exactly in format of [ENTITY 1, RELATIONSHIP, ENTITY 2]. The relationship is directed, so the order matters.
                # Both ENTITY 1 and ENTITY 2 should be noun or proper nouns.
                # Any element in [ENTITY 1, RELATIONSHIP, ENTITY 2] should be conclusive, make it as short as possible.
                # Do this in both breadth and depth. Expand [ENTITY 1, RELATIONSHIP, ENTITY 2] until the size reaches 100.
                f'''
                    Given a medical prompt, extrapolate as many relationships as possible of it and provide a list of updates.  
                    The relationships should be helpful for healthcare prediction (e.g., drug recommendation, mortality prediction, readmission prediction …) 
                    Each update should be exactly in format of [ENTITY 1, RELATIONSHIP, ENTITY 2]. The relationship is directed, so the order matters. 
                    Do this in both breadth and depth. Expand [ENTITY 1, RELATIONSHIP, ENTITY 2] until the list size reaches 50.

                    prompt: {term}
                    updates: 
                '''
            )
            data = {'index': index, 'entity_name': term, 'triple': response}
            df = df.append(data, ignore_index=True)
    df['triple'] = df['triple'].apply(lambda x: extract_triplets(x))
    return df

def multi_pro(code_map, category, flag='ehr_kg'):
    """多线程处理"""
    print("Now we are processing!")
    with ThreadPoolExecutor(max_workers=3) as executor:
        # asycn
        futures = [executor.submit(save_triple_multi, (code_map_entity, category, flag)) for code_map_entity in
                   code_map.items()]
        query2res = collections.defaultdict(int)  # 因为异步等待结果，返回的顺序是不定的，所以记录一下进程和输入数据的对应
        # 异步等待结果（返回顺序和原数据顺序可能不一致） ，直接predict函数里返回结果？
        for job in as_completed(futures):
            query, res = job.result(timeout=None)  # 默认timeout=None，不限时间等待结果
            query2res[query] = res
            time.sleep(1)
    if flag == 'ehr_kg':
        df = pd.DataFrame(columns=['index', 'entity_name', 'triple'])
        # df['triple'] = df['triple'].str.split('updates').str.get(2)
    else:
        df = pd.DataFrame(columns=['index', 'entity_name', 'explain'])

    for i, data in query2res.items():
        df = df.append(data, ignore_index=True)
    df['triple'] = df['triple'].apply(lambda x: extract_triplets(x))


    return df


def extract_triplets(string):
    triplets = []
    start_index = string.find("[")
    while start_index != -1:
        end_index = string.find("]", start_index)
        if end_index != -1:
            triplet = string[start_index + 1:end_index]
            triplet_list = triplet.split(", ")
            if len(triplet_list) == 3:
                stripped_triplet = [item.strip() for item in triplet_list]
                triplets.append(stripped_triplet)
            start_index = string.find("[", end_index)
        else:
            break
    return triplets


if __name__ == '__main__':
    file = load_pickle(config['KG_DATADIR'] + 'name.pkl')
    feature_keys = ['conditions', 'procedures', 'drugs']
    for feature_key in feature_keys:
        print(len(file[feature_key]))
    gpt = GPT2(choice='mmd')


    # first test a sample
    name_map_med = file['conditions']
    data = list(name_map_med.items())
    name_map_med = dict(data[:1])
    df_drug = save_triple(name_map_med, category='condition', flag='ehr_kg') # drug condition, procedure

    print(df_drug['triple'].iloc[0])
    # conditions
    name_map_med = file['conditions']
    data = list(name_map_med.items())
    start_time = time.time()
    print(start_time)
    name_map_med_s1 = dict(data[:3000])
    print(time.time()-start_time)
    name_map_med_s2 = dict(data[3000:6000])
    name_map_med_s3 = dict(data[6000:9000])
    name_map_med_s4 = dict(data[9000:12000])
    name_map_med_s5 = dict(data[12000:15000])
    name_map_med_s6 = dict(data[15000:18000])
    name_map_med_s7 = dict(data[18000:])
    print(len(name_map_med_s1), len(name_map_med_s2), len(name_map_med_s3))

    # df_drug = save_triple(name_map_med_s1, category='condition', flag='ehr_kg') # drug condition, procedure
    # df_drug.to_csv('../data/readys/con1_gpt3.csv', index=False)
    # print("slice 1 done")
    # df_drug = save_triple(name_map_med_s2, category='condition', flag='ehr_kg') # drug condition, procedure
    # df_drug.to_csv('../data/readys/con2_gpt3.csv', index=False)
    # print("slice 2 done")
    df_drug = save_triple(name_map_med_s3, category='condition', flag='ehr_kg') # drug condition, procedure
    df_drug.to_csv('../data/readys/con3_gpt3.csv', index=False)
    print("slice 3 done")
    # df_drug = save_triple(name_map_med_s4, category='condition', flag='ehr_kg')  # drug condition, procedure
    # df_drug.to_csv('../data/readys/con4_gpt3.csv', index=False)
    # print("slice 4 done")
    # df_drug = save_triple(name_map_med_s5, category='condition', flag='ehr_kg') # drug condition, procedure
    # df_drug.to_csv('../data/readys/con5_gpt3.csv', index=False)
    # print("slice 5 done")
    # df_drug = save_triple(name_map_med_s6, category='condition', flag='ehr_kg') # drug condition, procedure
    # df_drug.to_csv('../data/readys/con6_gpt3.csv', index=False)
    # print("slice 6 done")
    # df_drug = save_triple(name_map_med_s7, category='condition', flag='ehr_kg') # drug condition, procedure
    # df_drug.to_csv('../data/readys/con7_gpt3.csv', index=False)
    # print("slice 7 done")
    # print("All done")

    # # procedures
    # name_map_med = file['procedures']
    # data = list(name_map_med.items())
    # name_map_med_s1 = dict(data[:3000])
    # name_map_med_s2 = dict(data[3000:6000])
    # name_map_med_s3 = dict(data[6000:9000])
    # name_map_med_s4 = dict(data[9000:])

    # df_drug = save_triple(name_map_med_s1, category='procedure', flag='ehr_kg') # drug condition, procedure
    # df_drug.to_csv('../data/readys/proc1_gpt3.csv', index=False)
    # print("slice 1 done")
    # df_drug = save_triple(name_map_med_s2, category='procedure', flag='ehr_kg') # drug condition, procedure
    # df_drug.to_csv('../data/readys/proc2_gpt3.csv', index=False)
    # print("slice 2 done")
    # df_drug = save_triple(name_map_med_s3, category='procedure', flag='ehr_kg') # drug condition, procedure
    # df_drug.to_csv('../data/readys/proc3_gpt3.csv', index=False)
    # print("slice 3 done")
    # df_drug = save_triple(name_map_med_s4, category='procedure', flag='ehr_kg')  # drug condition, procedure
    # df_drug.to_csv('../data/readys/proc4_gpt3.csv', index=False)
    # print("slice 4 done")

    # drugs
    # name_map_med = file['drugs']
    # data = list(name_map_med.items())
    # name_map_med = dict(data)
    # df_drug = save_triple(name_map_med, category='drug', flag='ehr_kg') # drug condition, procedure
    # df_drug.to_csv('../data/readys/drug_gpt3.csv', index=False)
