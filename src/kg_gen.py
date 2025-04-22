# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : kg_gen.py
# Time       ：8/3/2024 9:13 am
# Author     ：Chuang Zhao
# version    ：python 
# Description：xxx
"""
import openai
import transformers
import torch
import time
import dashscope
import collections
from http import HTTPStatus
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI # 1.3.7
from utils import get_atc_name, get_node_name, get_stand_system, get_aux_icd
from pyhealth.tokenizer import Tokenizer
from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset
from pyhealth.tasks import (drug_recommendation_mimic3_fn, drug_recommendation_mimic4_fn, length_of_stay_prediction_mimic3_fn,length_of_stay_prediction_mimic4_fn,
                            mortality_prediction_mimic3_fn,mortality_prediction_mimic4_fn, readmission_prediction_mimic3_fn, readmission_prediction_mimic4_fn)


import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from umlsparser.UMLSParser import gpt_to_triple, triple_save, tokenizer
from tqdm import tqdm
from sklearn.manifold import TSNE
from config import config
from utils import load_pickle, save_pickle, near
from transformers import AutoTokenizer, AutoModel

def extract_embs(entity, dataset=None):
    """可以在编码之后"""
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").to('cuda:7')
    all_names = list(entity.values())
    print(all_names[:2])
    bs = 256  # batch size during inference
    all_embs = []
    for i in tqdm(np.arange(0, len(all_names), bs)):
        try:
            toks = tokenizer.batch_encode_plus(all_names[i:i + bs],
                                               padding="max_length",
                                               max_length=25,
                                               truncation=True,
                                               return_tensors="pt")
        except:
            # replace nan with 'Unknown'
            updated_list = ['Unknown' if (isinstance(v, float) and math.isnan(v)) or v != v else v for v in
                            all_names[i:i + bs]]

            toks = tokenizer.batch_encode_plus(updated_list,
                                               padding="max_length",
                                               max_length=25,
                                               truncation=True,
                                               return_tensors="pt")

        toks_cuda = {}
        for k, v in toks.items():
            toks_cuda[k] = v.to('cuda:7')
        cls_rep = model(**toks_cuda)[0][:, 0, :]  # use CLS representation as the embedding
        all_embs.append(cls_rep.cpu().detach().numpy())

    all_embs = np.concatenate(all_embs, axis=0)
    # pad = np.zeros((1, config['KG_DIM'])) # 别家了
    # all_embs = np.concatenate((all_embs, pad), axis=0)  # 加到末尾, 这个思考一下
    # save_pickle(all_embs, config['KG_DATADIR'] + dataset + '/entity_emb.pkl')
    print("Extract done!")


def query_nn_neighbor(query, name_file, emb_file, k=5, trun_ratio=0.01):
    """
    :param query: string
    :param file:
    :return:
    """
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").cuda()
    index_name_dict, all_reps_emb = load_pickle(name_file), load_pickle(emb_file)
    print("ALL name shape:", all_reps_emb.shape)
    trun_num = round(all_reps_emb.shape[0] * trun_ratio)
    index_name, all_reps_emb = np.array(list(index_name_dict.keys())[:trun_num]), all_reps_emb[:trun_num]
    print("Truncated name shape:", all_reps_emb.shape)


    query_tokens = tokenizer.batch_encode_plus(query,
                                       padding="max_length",
                                       max_length=25,
                                       truncation=True,
                                       return_tensors="pt")
    # print(query_tokens)
    query_tokens = {k: v.cuda() for k, v in query_tokens.items()}
    query_cls_rep = model(**query_tokens)[0][:, 0, :] # use cls as repr

    index = near(query_cls_rep, all_reps_emb, index_name, k=k)
    print('Nearest Neighbors Name:', np.vectorize(index_name_dict.get)(index))



def visual_kg_emb(name_file, emb_file, trun_ratio=0.01, k=5, nk=5):
    """visualize KG distribution"""
    index_name_dict, all_reps_emb = load_pickle(name_file), load_pickle(emb_file)
    trun_num = round(all_reps_emb.shape[0] * trun_ratio)
    index_name, all_reps_emb = np.array(list(index_name_dict.keys())[:trun_num]), all_reps_emb[:trun_num]

    indices = list(range(all_reps_emb.shape[0]))
    embedding_matrix = np.array(all_reps_emb)

    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embedding_matrix)

    plt.figure(figsize=(10, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)

    anchors = np.random.choice(len(indices), size=k, replace=False)  # 随机选择5个锚点
    print(anchors)
    near_index = near(embedding_matrix[anchors], embedding_matrix, index_name, k=nk+1, real_index=False)
    # near_name = np.vectorize(index_name_dict.get)(near_index)

    for num, anchor in enumerate(anchors):
        # 标记锚点
        plt.scatter(embeddings_2d[anchor, 0], embeddings_2d[anchor, 1], color='red')
        plt.text(embeddings_2d[anchor, 0], embeddings_2d[anchor, 1], index_name[anchor], color='black')

        # 标记近邻
        for neighbor in near_index[num][1:]: # 排除自己
            plt.scatter(embeddings_2d[neighbor, 0], embeddings_2d[neighbor, 1], color='blue')
            plt.text(embeddings_2d[neighbor, 0], embeddings_2d[neighbor, 1], index_name[neighbor], color='green')

    plt.show()




def kg_preprocess(dataset_name):
    # triple_save(dataset_name) # 合并GPT和external KG
    # tokenizer(dataset_name) # 将name三元组进行ID编码
    # if add_ehr: 这个放在main中比较好，重新处理KG
    #     add_edges(train_dataset, kg, ehr_id_dic)
    # 获取emb
    file = load_pickle(config['KG_DATADIR'] + dataset_name + '/ehr_name_map.pkl')
    extract_embs(file, dataset=dataset_name)
    print("KG Preprocess Done!")



###################################### other LLM

class GPT(object):
    def __init__(self, api_key=None, model_engine=None):
        if api_key is None:
            api_key = "sk-xxxxxx"
        openai.api_key = api_key
        # 调用 ChatGPT 接口
        if model_engine is None:
            self.model_engine = "text-davinci-003"
        else:
            self.model_engine = model_engine

        # """for openai 1.3.7"""
        os.environ["OPENAI_API_KEY"] = 'xxxxx'
        self.client = OpenAI()
        """llama2-13b-chat-v2"""
        dashscope.api_key = 'xxxx'

    def chat(self, prompt):
        """for openai 0.28.1"""
        completion = openai.Completion.create(
            engine=self.model_engine,
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.7,
        )
        response = completion.choices[0].text
        return response

    def chat_upgrade(self, prompt):
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a medical expert, skilled in medical knowledge."},  # 设定系统角色
                {"role": "user", "content": prompt},
            ]  #
        )
        response = completion.choices[0].message.content.strip()
        return response

    def chat_llama(self, prompt):
        messages = [{'role': 'user', 'content': prompt}]
        response = dashscope.Generation.call(
            model='llama2-13b-chat-v2',
            messages=messages,
            result_format='message',  # set the result to be "message" format.
            max_tokens=2048
        )
        response = response.output.choices[0].message.content.strip()
        return response


class GPT2(object):
    """finetune with medical llm"""
    def __init__(self, choice=None):
        self.tokenizer = transformers.LlamaTokenizer.from_pretrained('axiong/PMC_LLaMA_13B')
        self.model = transformers.LlamaForCausalLM.from_pretrained('axiong/PMC_LLaMA_13B')
        if choice:
            from modelscope import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("AI-ModelScope/BioMistral-7B")
            self.model = AutoModel.from_pretrained("AI-ModelScope/BioMistral-7B")

    def chat_upgrade(self, prompt):
        batch = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False
        )
        with torch.no_grad():
            completion = self.model.generate(
                inputs=batch["input_ids"],
                max_length=500, # 原来是200
                do_sample=True,
                top_k=100
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
        prompt: Diabetes        
        Triple set: [
        (Diabetes, Increases_Risk_of, Cardiovascular_Complications),
        (Insulin, Common_Treatment_for, Diabetes),
        (Metformin, Recommended_for, Type_2_Diabetes)
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


def get_node_prompting(category):
    """
    生成节点属性
    :return:
    """
    if category == 'condition':
        template = "[Medical Condition Name] is a [brief definition]. It is commonly caused by [primary causes] and characterized by symptoms such as [key symptoms]. Diagnosis typically involves [standard diagnostic methods]. Treatment may include [common treatment approaches] and can be supported by [prevention strategies, if applicable]. The prognosis of [Medical Condition Name] is [general outlook], varying based on individual cases and management effectiveness."
        example = """
        prompt: Asthma
        explanation: Asthma is a chronic respiratory condition characterized by airway inflammation and narrowing. It is commonly caused by allergens, genetic factors, and environmental triggers, characterized by symptoms such as wheezing, coughing, and breathlessness. Diagnosis typically involves tests like spirometry and peak flow measurement. Treatment may include the use of inhalers and corticosteroids and can be supported by avoiding triggers and controlling allergies. The prognosis of Asthma varies individually but is generally manageable with proper treatment."""
    elif category == 'procedure':
        template = "[Medical Procedure Name] is a procedure performed to [primary purpose]. During the procedure, [brief description of the procedure]. Patients usually need to [preparation needed] before undergoing it. The common risks or side effects include [risks], and the typical recovery process involves [recovery process]. The expected outcome of [Medical Procedure Name] is [expected results]."
        example = """
        prompt: Colonoscopy
        explanation: Colonoscopy is a procedure performed to diagnose intestinal issues. During the procedure, an endoscope is used for a detailed examination of the colon. Patients usually need to follow a bowel cleansing regimen before undergoing it. The common risks or side effects include bleeding and potential perforation, and the typical recovery process involves a short period of rest post-procedure. The expected outcome of a Colonoscopy is to identify any abnormalities in the colon, informing further treatment decisions."""
    elif category == 'drug':
        template = "[Drug Name] is a [drug class] used primarily for [primary uses]. It works by [mechanism of action]. The typical dosage range is [dosage], and common side effects may include [side effects]. Patients taking [Drug Name] should be aware of significant interactions with [interactions] and adhere to precautions such as [precautions]."
        example = """
        prompt: Metformin
        explanation: Metformin is an oral hypoglycemic medication used primarily for managing Type 2 diabetes. It works by reducing hepatic glucose production and increasing insulin sensitivity. The typical dosage range varies, but can go up to 2000 mg per day. Common side effects may include gastrointestinal upset and a rare risk of lactic acidosis. Patients taking Metformin should be aware of significant interactions with alcohol and the impact on kidney function, and adhere to precautions such as regular monitoring of blood sugar levels and renal function."""
    return template, example


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
                Given a prompt (a medical condition, procedure, or drug), please give a external knowledge graph about it. You have to follow the rules below. 

                Must be helpful to healthcare prediction tasks like drug recommendation, mortality prediction, readmission prediction.
                Be as broad and realistic as possible, and you can explore relationships using breadth and depth searches.
                Synonyms are expressed in only one form, such as 'lead to' and 'result in'
                No duplicate triples.

                Please give a list of triple sets like below, 50 triple sets are required:

                {example}

                prompt: {term}
                updates: 
                '''
            )
            data = {'index': index, 'entity_name': term, 'triple': response}
            df = df.append(data, ignore_index=True)
    else:
        df = pd.DataFrame(columns=['index', 'entity_name', 'explain'])

        for index, entity_name in code_map.items():
            term, category = entity_name, category
            template, example = get_node_prompting(category)
            response = gpt.chat_upgrade(
                f'''
                Given a prompt (a medical condition, procedure, or drug), please explain it briefly follow the template below:
                {template}

                Example:{example}

                prompt: {term}
                explanation: 
                '''
            )
            data = {'index': index, 'entity_name': term, 'triple': response}
            df = df.append(data, ignore_index=True)
    return df


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
                    Given a prompt (a medical condition/procedure/drug), extrapolate as many relationships as possible of it and provide a list of updates.  
                    The relationships should be helpful for healthcare prediction (e.g., drug recommendation, mortality prediction, readmission prediction …) 
                    Each update should be exactly in format of [ENTITY 1, RELATIONSHIP, ENTITY 2]. The relationship is directed, so the order matters. 
                    Both ENTITY 1 and ENTITY 2 should be noun or proper nouns. 
                    Any element in [ENTITY 1, RELATIONSHIP, ENTITY 2] should be conclusive, make it as short as possible. 
                    Do this in both breadth and depth. Expand [ENTITY 1, RELATIONSHIP, ENTITY 2] until the size reaches 50.

                    {example}

                    prompt: {term}
                    updates: 
                    '''
                )
                data = {'index': index, 'entity_name': term, 'triple': response}
            else:
                template, example = get_node_prompting(category)
                response = gpt.chat_upgrade(
                    f'''
                    Given a prompt (a medical condition, procedure, or drug), please explain it briefly follow the template below:
                    {template}

                    Example:{example}

                    prompt: {term}
                    explanation:
                    '''
                )
                data = {'index': index, 'entity_name': term, 'explain': response}
                # df = df.append(data, ignore_index=True)
            return index, data

        # except openai.error.RateLimitError as e: # v1 版本后不需要了
        #     print("超出openai api 调用频率：", e)
        #     print('重新请求....')
        #     retry_count += 1
        #     retry_interval *= 2  # 指数退避策略，每次重试后加倍重试间隔时间
        #     time.sleep(retry_interval)

        except TimeoutError:
            print("任务执行超时：", index)
            print('重新请求....')
            retry_count += 1
            retry_interval *= 2  # 指数退避策略，每次重试后加倍重试间隔时间
            time.sleep(retry_interval)

        except Exception as e:
            print("任务执行出错：", e)
            print('重新请求....')
            retry_count += 1
            retry_interval *= 2  # 指数退避策略，每次重试后加倍重试间隔时间
            time.sleep(retry_interval)

    return index, 'api请求失败'


def multi_pro(code_map, category, flag='ehr_kg'):
    """多线程处理"""
    print("Now we are processing!2")
    with ThreadPoolExecutor(max_workers=3) as executor:
        # asycn
        futures = [executor.submit(save_triple_multi, (code_map_entity, category, flag)) for code_map_entity in
                   code_map.items()]
        query2res = collections.defaultdict(int)  # 因为异步等待结果，返回的顺序是不定的，所以记录一下进程和输入数据的对应
        # 异步等待结果（返回顺序和原数据顺序可能不一致） ，直接predict函数里返回结果？
        for job in as_completed(futures):
            query, res = job.result(timeout=None)  # 默认timeout=None，不限时间等待结果
            query2res[query] = res

            time.sleep(1)  # 为了避免超过OpenAI API的速率限制，每次预测之间间隔1秒
            print('XXXXXX')
    if flag == 'ehr_kg':
        df = pd.DataFrame(columns=['index', 'entity_name', 'triple'])
    else:
        df = pd.DataFrame(columns=['index', 'entity_name', 'explain'])

    for i, data in query2res.items():
        df = df.append(data, ignore_index=True)
    return df



def filter_codes(dataset1, dataset2, code_type1, code_type2):
    names_set = {}
    voc_set = {}
    icd10_set = {}
    feature_keys = ["conditions", "procedures", "drugs"]
    special_tokens = ["<pad>", "<unk>"]
    for feature_key in feature_keys:
        aux_code = get_aux_icd(feature_key) # atc似乎不用

        names = {}
        icd10_code = []
        tokenizer1 = Tokenizer(
            tokens= dataset1.get_all_tokens(key=feature_key),
            special_tokens=special_tokens,
        )
        tokenizer2 = Tokenizer(
            tokens= dataset2.get_all_tokens(key=feature_key),
            special_tokens=special_tokens,
        )
        tokens1 = list(tokenizer1.vocabulary.idx2token.values()) # 这里的token其实是ehr id
        for i in tokens1:
            if i in special_tokens:
                continue
            try:
                name = code_type1[feature_key].lookup(i)
            except:
                name = code_type1['conditions'].lookup(i) # 3601, 3605两个
            names[i] = name

        # tokens1 = [code_type1[feature_key].standardize(token) for token in tokens1]
        tokens2 = list(tokenizer2.vocabulary.idx2token.values())
        for i in tokens2:
            if i in special_tokens:
                continue
            try:
                name = code_type1[feature_key].lookup(i)
            except:
                try:
                    name = code_type2[feature_key].lookup(i) # ICD10
                except:
                    try:
                        name = aux_code[i] # 有些没有icd-cm,老版本，45个, 没有icd-proc是proc
                        icd10_code.append(i)
                    except:
                        continue # 确实得忽略一部分code了

            names[i] = name

        # tokens2 = [code_type2[feature_key].standardize(token) for token in tokens2] # 用10的先过滤一遍
        tokens = set(tokens1) | set(tokens2)

        print("Len of tokens ", len(tokens1), len(tokens2), len(tokens)) # 19686
        voc_set[feature_key] = tokens
        names_set[feature_key] = names
        icd10_set[feature_key] = icd10_code

    return voc_set, names_set, icd10_set # 如果nameset能成，就不需要voc set了


def using_code(feature_keys):
    """GPT很贵，还是节约一点，只采集需要的code"""
    # name_map_diag = get_node_name('ICD9CM')
    # name_map_proc = get_node_name('ICD9PROC')
    # name_map_med = get_atc_name(4)
    # name2ehr_diag = dict(zip(name_map_diag.values(), name_map_diag.keys()))
    # name2ehr_proc = dict(zip(name_map_proc.values(), name_map_proc.keys()))
    # name2ehr_med = dict(zip(name_map_med.values(), name_map_med.keys()))
    # ehr = {"conditions":name2ehr_diag, "procedures": name2ehr_proc, "drugs":name2ehr_med}

    base_dataset1 = MIMIC3Dataset(
        root="/home/czhaobo/KnowHealth/data/physionet.org/files/mimiciii/1.4",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        dev=False,
        refresh_cache=False,
    )
    base_dataset1.stat()
    sample_dataset1 = base_dataset1.set_task(drug_recommendation_mimic3_fn)
    sample_dataset1_1 = base_dataset1.set_task(length_of_stay_prediction_mimic3_fn)
    sample_dataset1_2 = base_dataset1.set_task(mortality_prediction_mimic3_fn)
    sample_dataset1_3 = base_dataset1.set_task(readmission_prediction_mimic3_fn)

    base_dataset2 = MIMIC4Dataset(
        root="/home/czhaobo/KnowHealth/data/physionet.org/files/mimiciv/2.0/hosp",  # 2.2 不大行
        tables=["diagnoses_icd", "procedures_icd", "prescriptions", "labevents"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        dev=False,
        refresh_cache=False, # 第一次用True
    )
    base_dataset2.stat()
    sample_dataset2 = base_dataset2.set_task(drug_recommendation_mimic4_fn)
    sample_dataset2_1 = base_dataset2.set_task(length_of_stay_prediction_mimic4_fn)
    sample_dataset2_2 = base_dataset2.set_task(mortality_prediction_mimic4_fn)
    sample_dataset2_3 = base_dataset2.set_task(readmission_prediction_mimic4_fn)

    diag_sys, proc_sys, med_sys = get_stand_system('MIMIC-III')
    diag_sys2, proc_sys2, med_sys2 = get_stand_system('MIMIC-IV')
    code_type1 = {'conditions':diag_sys, 'procedures':proc_sys, 'drugs': med_sys}
    code_type2 = {'conditions':diag_sys2, 'procedures':proc_sys2, 'drugs': med_sys2}


    rec_voc, rec_names_set, rec_icd10_set = filter_codes(sample_dataset1, sample_dataset2, code_type1, code_type2) 
    los_voc, los_names_set, los_icd10_set = filter_codes(sample_dataset1_1, sample_dataset2_1, code_type1, code_type2)
    mor_voc, mor_names_set, mor_icd10_set = filter_codes(sample_dataset1_2, sample_dataset2_2, code_type1, code_type2)
    red_voc, red_names_set, red_icd10_set = filter_codes(sample_dataset1_3, sample_dataset2_3, code_type1, code_type2)

    print("==============Formal filter================")
    final_vocs = {}
    final_names = {}
    final_icd10s = {}
    for feature_key in feature_keys:
        print("Now we are processing ", feature_key)
        print("Len of key, rec, los, mor, red, ", len(rec_voc[feature_key]), len(los_voc[feature_key]),
              len(mor_voc[feature_key]), len(red_voc[feature_key]))
        final_voc = set(rec_voc[feature_key]) | set(los_voc[feature_key]) | set(mor_voc[feature_key]) | set(
            red_voc[feature_key])
        final_vocs[feature_key] = final_voc
        print("Len of combination, ", len(final_voc)) # 如果差不多我就用原始的表了，我去

        final_name = {}
        final_name.update(rec_names_set[feature_key])
        final_name.update(los_names_set[feature_key])
        final_name.update(mor_names_set[feature_key])
        final_name.update(red_names_set[feature_key])

        final_names[feature_key] = final_name
        final_icd = set(rec_icd10_set) | set(los_icd10_set) | set(mor_icd10_set) | set(red_icd10_set)
        final_icd10s[feature_key] = final_icd
        # ehr_name = {}
        # name2ehr = ehr[feature_key]
        # for name in final_voc:
        #     ehr_name[name2ehr[name]] = name
        # final_vocs[feature_key] = ehr_name
    save_pickle(final_vocs, config['KG_DATADIR'] + 'voc.pkl')
    save_pickle(final_names, config['KG_DATADIR'] + 'name.pkl')
    save_pickle(final_names, config['KG_DATADIR'] + 'icd10.pkl')

    return final_vocs



if __name__ == '__main__':
    # gpt = GPT()

    ## 对其进行过滤
    # feature_keys = ["conditions", "procedures", "drugs"]
    # final_vocs = using_code(feature_keys) # 有少量ICD编码缺失，非常少

    # load name
    # file = load_pickle(config['KG_DATADIR'] + 'name.pkl')
    # # for feature_key in feature_keys:
    # #     print(len(file[feature_key]))
    #
    # # 加载代码GPT请求
    # name_map_med = file['conditions']
    # data = list(name_map_med.items())
    # name_map_med_s1 = dict(data[:3000])
    # name_map_med_s2 = dict(data[3000:6000])
    # name_map_med_s3 = dict(data[6000:9000])
    # name_map_med_s4 = dict(data[9000:12000])
    # name_map_med_s5 = dict(data[12000:15000])
    # name_map_med_s6 = dict(data[15000:18000])
    # name_map_med_s7 = dict(data[18000:])
    # print(len(name_map_med_s1), len(name_map_med_s2), len(name_map_med_s3))
    # # print(name_map_med_s1,name_map_med_s2,name_map_med_s3)
    # # print(name_map_med)
    # # print(name_map_med)
    # df_drug = multi_pro(name_map_med_s1, category='condition', flag='ehr_kg') # drug condition, procedure
    # df_drug.to_csv('../data/ready/con1_gpt3.csv', index=False)
    # print("slice 1 done")
    # time.sleep(20)
    # df_drug = multi_pro(name_map_med_s2, category='condition', flag='ehr_kg') # drug condition, procedure
    # df_drug.to_csv('../data/ready/con2_gpt3.csv', index=False)
    # print("slice 2 done")
    # time.sleep(20)
    # df_drug = multi_pro(name_map_med_s3, category='condition', flag='ehr_kg') # drug condition, procedure
    # df_drug.to_csv('../data/ready/con3_gpt3.csv', index=False)
    # print("slice 3 done")
    # time.sleep(20)
    # df_drug = multi_pro(name_map_med_s4, category='condition', flag='ehr_kg')  # drug condition, procedure
    # df_drug.to_csv('../data/ready/con4_gpt3.csv', index=False)
    # time.sleep(20)
    # print("slice 4 done")
    #
    # df_drug = multi_pro(name_map_med_s5, category='condition', flag='ehr_kg') # drug condition, procedure
    # df_drug.to_csv('../data/ready/con5_gpt3.csv', index=False)
    # print("slice 5 done")
    # time.sleep(20)
    # df_drug = multi_pro(name_map_med_s6, category='condition', flag='ehr_kg') # drug condition, procedure
    # df_drug.to_csv('../data/ready/con6_gpt3.csv', index=False)
    # print("slice 6 done")
    # time.sleep(20)
    # df_drug = multi_pro(name_map_med_s7, category='condition', flag='ehr_kg') # drug condition, procedure
    # df_drug.to_csv('../data/ready/con7_gpt3.csv', index=False)
    # print("slice 7 done")
    # print("All done")

    # dataset = None
    # kg = None
    # add_edges(dataset, kg)
    
    # KG Preprocess
    # kg_preprocess('MIII')
    # print('=======MIII KG Generated=======')
    # kg_preprocess('MIV')
    # print('=======MIV KG Generated=======')

    # # query NN
    # emb_file = config['KG_DATADIR'] + 'MIII' + '/entity_emb.pkl'
    # name_file = config['KG_DATADIR'] + 'MIII' + '/ehr_name_map.pkl'
    # query_nn_neighbor(['Sulfonamides', 'Potassium lactate'], name_file, emb_file, k=6) # 看看是否真的能找到最近的邻居

    # visualize KG emb
    # emb_file = config['KG_DATADIR'] + 'MIII' + '/entity_emb.pkl'
    # name_file = config['KG_DATADIR'] + 'MIII' + '/ehr_name_map.pkl'
    # visual_kg_emb(name_file, emb_file, trun_ratio=0.001, k=5, nk=5)
    #

    # 下面是一些分析
    # import os
    # # 使用第一张与第三张GPU卡
    # os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    # terms = ['Sulfonamides', 'Diphenylpropylamine derivative analgesics', 'Potassium lactate',
    #          'Pipenzolate and psycholeptics']
    # example = ehr_kg_prompting('drug')
    # gpt = GPT2()
    # for term in terms:
    #     response = gpt.chat_upgrade(
    #         f'''
    #         Given a prompt (a medical condition/procedure/drug), extrapolate as many relationships as possible of it and provide a list of updates.
    #         The relationships should be helpful for healthcare prediction (e.g., drug recommendation, mortality prediction, readmission prediction …)
    #         Each update should be exactly in format of [ENTITY 1, RELATIONSHIP, ENTITY 2]. The relationship is directed, so the order matters.
    #         Both ENTITY 1 and ENTITY 2 should be noun or proper nouns.
    #         Any element in [ENTITY 1, RELATIONSHIP, ENTITY 2] should be conclusive, make it as short as possible.
    #         Do this in both breadth and depth. Expand [ENTITY 1, RELATIONSHIP, ENTITY 2] until the size reaches 100.
    #
    #         Please give a list of triple sets like below, 50 triple sets are required:
    #
    #         prompt: {term}
    #         updates:
    #         '''
    #     )
    #     print(response)
    #
    #     print('================================')
    #
