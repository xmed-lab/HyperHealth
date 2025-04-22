<h1 align="center"> Beyond Sequential Patterns: Rethinking Healthcare
Predictions with Contextual Insights </h1>

## About Our Work

Update: 2025/04/22: We have created a repository for the paper titled *Beyond Sequential Patterns: Rethinking Healthcare Predictions with Contextual Insights*, accepted *TOIS2025*. In this repository, we offer the original sample datasets, preprocessing scripts, and algorithm files to showcase the reproducibility of our work.

![image-20240717084138363](https://s2.loli.net/2024/07/17/Uxn8jWDCMezN1EK.png)

![image-20240717084254049](https://s2.loli.net/2024/07/17/SrhvG7a1DTXBu2z.png)

## Requirements

- openai==1.3.5
- torch==1.13.1+cu117
- dgl==1.1.2
- pyhealth==1.1.4
- seaborn==0.13.0

## Data Sets

Owing to the copyright stipulations associated with the dataset, we are unable to provide direct upload access. However, it can be readily obtained by downloading directly from the official website: [MIMIC-III](https://physionet.org/content/mimiciii/1.4/), [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/). 

The structure of the data set should be like,

```powershell
data
|_ drugrec-gpt
|  |_ MIII
|  |_ _processed
|  |_ _ _datasets_pre_stand.pkl
|  |_ MIV
|  |_ _ _datasets_pre_stand.pkl
|_ los
|_ mortality-gpt
|_ readmission-gpt
|_ ready-gpt
|_ _MIII
|  |_ _processed
|  |_ triples_id.pkl
|  |_ entity_emb.pkl
|  |_ re_id.pkl
```

## RUN

```powershell
# process the data
python kg_gen.py
# run the file
python main_los.py
```

## Acknowledge & Contact

You can contact czhaobo@connect.ust.hk for more information and help.