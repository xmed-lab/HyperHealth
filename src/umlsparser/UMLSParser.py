import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
from typing import Dict

from tqdm import tqdm

from umlsparser.model.Concept import Concept
from umlsparser.model.SemanticType import SemanticType
from src.config import config
from src.utils import save_pickle
from utils import *

UMLS_sources_by_language = {
    'ENG': ['MSH', 'CSP', 'NCI', 'PDQ', 'NCI_NCI-GLOSS', 'CHV', 'NCI_CRCH', 'NCI_CareLex', 'UWDA', 'FMA',
            'NCI_CDISC-GLOSS', 'NCI_NICHD', 'NCI_CTCAE', 'HPO', 'MEDLINEPLUS', 'NCI_CDISC', 'NCI_FDA', 'NCI_GAIA',
            'HL7V3.0', 'PSY', 'SPN', 'AIR', 'GO', 'CCC', 'SNOMEDCT_US', 'UMD', 'NIC', 'ALT', 'NCI_EDQM-HC', 'JABL',
            'NUCCPT', 'LNC', 'ICF-CY', 'NCI_BRIDG', 'ICF', 'NDFRT', 'NANDA-I', 'PNDS', 'NOC', 'OMS', 'NCI_CTEP-SDC',
            'NCI_DICOM', 'NCI_KEGG', 'NCI_BioC', 'MCM', 'AOT', 'NCI_CTCAE_5', 'NCI_CTCAE_3', 'MDR', 'NCI_INC'],
    'SPA': ['MDRSPA', 'SCTSPA', 'MSHSPA'],
    'FRE': ['MDRFRE', 'MSHFRE'],
    'JPN': ['MDRJPN'],
    'CZE': ['MDRCZE', 'MSHCZE'],
    'ITA': ['MDRITA'],
    'GER': ['MDRGER'],
    'POR': ['MDRPOR', 'MSHPOR'],
    'DUT': ['MDRDUT'],
    'HUN': ['MDRHUN'],
    'NOR': ['MSHNOR'],
    'HRV': ['MSHSCR']  # not sure
}

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# https://www.ncbi.nlm.nih.gov/books/NBK9685/ data sample parse
def byLineReader(filename):
    with open(filename, "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            yield line
            line = f.readline()
    return

class UMLSParser:

    def __init__(self, path: str, language_filter: list = []):
        """
        :param path: Basepath to UMLS data files
        :param languages: List of languages with three-letter style language codes (if empty, no filtering will be applied)
        """
        logger.info("Initialising UMLSParser for basepath {}".format(path))
        if language_filter:
            logger.info("Language filtering for {}".format(",".join(language_filter)))
        else:
            logger.info("No language filtering applied.")
        self.paths = {
            'MRCONSO': os.path.join(path, 'META', 'MRCONSO.RRF'),
            'MRDEF': os.path.join(path, 'META', 'MRDEF.RRF'),
            'MRSTY': os.path.join(path, 'META', 'MRSTY.RRF'),
            'SRDEF': os.path.join(path, 'NET', 'SRDEF'),
            'MRREL': os.path.join(path, 'old_version', 'MRREL.txt')
        }
        self.language_filter = language_filter
        self.concepts = {}
        self.semantic_types = {}
        self.__parse_mrconso__()
        self.__parse_mrdef__()
        self.__parse_mrsty__()
        self.__parse_srdef__()

        self.load_rel()

    def __get_or_add_concept__(self, cui: str) -> Concept:
        concept = self.concepts.get(cui, Concept(cui))
        self.concepts[cui] = concept
        return concept

    def __get_or_add_semantic_type__(self, tui: str) -> SemanticType:
        semantic_type = self.semantic_types.get(tui, SemanticType(tui))
        self.semantic_types[tui] = semantic_type
        return semantic_type

    def __parse_mrconso__(self):
        for line in tqdm(open(self.paths['MRCONSO']), desc='Parsing UMLS concepts (MRCONSO.RRF)'):
            line = line.split('|')
            data = {
                'CUI': line[0],
                'LAT': line[1],  # language of term
                'TS': line[2],  # term status
                'LUI': line[3],
                'STT': line[4],
                'SUI': line[5],
                'ISPREF': line[6],
                'AUI': line[7],
                'SAUI': line[8],
                'SCUI': line[9],
                'SDUI': line[10],
                'SAB': line[11],  # source abbreviation
                'TTY': line[12],
                'CODE': line[13],  # Unique Identifier or code for string in source
                'STR': line[14],  # description string
                'SRL': line[15],
                'SUPPRESS': line[16],
                'CVF': line[17]
            }
            if len(self.language_filter) != 0 and data.get('LAT') not in self.language_filter:
                continue
            concept = self.__get_or_add_concept__(data.get('CUI'))
            concept.__add_mrconso_data__(data)
        logger.info('Found {} unique CUIs'.format(len(self.concepts.keys())))

    def __parse_mrdef__(self):
        source_filter = []
        for language in self.language_filter:
            for source in UMLS_sources_by_language.get(language):
                source_filter.append(source)

        for line in tqdm(open(self.paths['MRDEF']), desc='Parsing UMLS definitions (MRDEF.RRF)'):
            line = line.split('|')
            data = {
                'CUI': line[0],
                'AUI': line[1],
                'ATUI': line[2],
                'SATUI': line[3],
                'SAB': line[4],  # source
                'DEF': line[5],  # definition
                'SUPPRESS': line[6],
                'CVF': line[7]
            }
            if len(self.language_filter) != 0 and data.get('SAB') not in source_filter:
                continue
            concept = self.__get_or_add_concept__(data.get('CUI'))
            concept.__add_mrdef_data__(data)

    def __parse_mrsty__(self):
        for line in tqdm(open(self.paths['MRSTY']), desc='Parsing UMLS semantic types (MRSTY.RRF)'):
            line = line.split('|')
            data = {
                'CUI': line[0],
                'TUI': line[1],
                'STN': line[2],  # empty in MRSTY.RRF ?
                'STY': line[3],  # empty in MRSTY.RRF ?
                'ATUI': line[4],  # empty in MRSTY.RRF ?
                'CVF': line[5]  # empty in MRSTY.RRF ?
            }
            concept = self.__get_or_add_concept__(data.get('CUI'))
            concept.__add_mrsty_data__(data)

    def __parse_srdef__(self):
        for line in tqdm(open(self.paths['SRDEF']), desc='Parsing UMLS semantic net definitions (SRDEF)'):
            line = line.split('|')
            data = {
                'RT': line[0],  # Semantic Type (STY) or Relation (RL)
                'UI': line[1],  # Identifier
                'STY_RL': line[2],  # Name of STY / RL
                'STN_RTN': line[3],  # Tree Number of STY / RL
                'DEF': line[4],  # Definition of STY / RL
                'EX': line[5],  # Examples of Metathesaurus concepts
                'UN': line[6],  # Usage note for STY assignment
                'NH': line[7],  # STY and descendants allow the non-human flag
                'ABR': line[8],  # Abbreviation of STY / RL
                'RIN': line[9]  # Inverse of the RL
            }
            semantic_type = self.__get_or_add_semantic_type__(data['UI'])
            semantic_type.__add_srdef_data__(data)
        logger.info('Found {} unique TUIs'.format(len(self.semantic_types.keys())))

    def get_concepts(self) -> Dict[str, Concept]:
        """
        :return: A dictionary of all detected UMLS concepts with CUI being the key.
        """
        return self.concepts

    def get_semantic_types(self) -> Dict[str, SemanticType]:
        """
        :return: A dictionary of all detected UMLS semantic types with TUI being the key.
        """
        return self.semantic_types

    def get_languages(self):
        return self.language_filter

    def load_rel(self):
        """load relation https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.related_concepts_file_mrrel_rrf/"""
        reader = byLineReader(os.path.join(self.paths['MRREL'])) # MRREL + self.type
        self.rel = set()
        for line in tqdm(reader, ascii=True, desc='Parsing UMLS relation (MRREL)'):
            l = line.strip().split("|")
            cui0 = l[0] # 实体 cui
            re = l[3] # relation
            cui1 = l[4]
            rel = l[7] # more specific relatoin
            if (cui0 in self.concepts.keys()) and (cui1 in self.concepts.keys()):
                str_rel = "\t".join([cui0, cui1, re, rel]) # \t
                # print('test rel', str_rel) # rel C0000103	C0046421 RN	mapped_to
                if not str_rel in self.rel and cui0 != cui1:
                    self.rel.update([str_rel])
        self.rel = list(self.rel)
        self.rel = [i.split('\t') for i in self.rel]
        # print(self.rel[0].split('\t'))

        print("rel count:", len(self.rel)) # 1500多万关系



def recode(umls, gpt_df, ehr_name_id, ehr_id_name, dataset=None):
    """
    change all triple sets using ehr code & unique code (if others, 暂时将gpt生成的和umls看作两个系统)
    """
    # 1. umls rel
    # umls = UMLSParser('/home/czhaobo/HyperHealth/data/umls-extract')
    print("Initial")

    data = pd.DataFrame(umls.rel, columns=['cui1', 'cui2', 'rel1','rel2'])
    data['rel'] = data['rel1'] + " " + data['rel2']
    # unique_concept = set(data['cui1'].unique()) | set(data['cui2'].unique())
    print("First Step End ")

    # 2. umls
    all_ehr_concepts = {} # [EHR: CID]
    atc_lis = [] # ATC code
    concepts_names = {} # all names of concept [CID:name, ATC:name] if exist atc
    if dataset == 'MIII':
        med_codes = ['ATC', 'ICD9CM', 'ICD9PROC']
    elif dataset == 'MIV':
        med_codes = ['ATC', 'ICD10CM', 'ICD10PROC'] # 10有可能要考虑proc

    # for all datasets
    i = 0
    overlap_num = 0
    for cui, concept in umls.get_concepts().items():
        # print(cui, concept) # 记录了其各种乱七把澡的名字
        med_code = set(med_codes) & set(concept.get_source_ids().keys())
        if med_code: # 有一个就行
            med_code = med_code.pop() # 'ATC'感觉要单独处理,因为过滤了；
            medids = concept.get_source_ids().get(med_code)  # 对应concept的对应EHR系统的id编码
            medids = list(medids)[0] # A02BC01
            if medids not in all_ehr_concepts.keys(): # nearly no 1-to-many
                if med_code == 'ATC':
                    atc_lis.append(medids)
                all_ehr_concepts[medids] = cui  # [EHR: CID]

                if medids in ehr_id_name.keys(): # [EHR:name]
                    concepts_names[medids] = ehr_id_name[medids] # concept.get_preferred_names_for_language('ENG')[0] # {ATCID: amosinlin, CUI:toubao}
                    overlap_num += 1
                elif med_code == 'ATC' and medids[:config['ATCLEVEL']+1] in ehr_id_name.keys(): # [ATC:name]
                    concepts_names[medids] = ehr_id_name[medids[:config['ATCLEVEL']+1]]
                    overlap_num += 1
                else: # [CID:name]
                    concepts_names[medids] = concept.get_preferred_names_for_language('ENG')[0]
            else:
                concepts_names[cui] = concept.get_preferred_names_for_language('ENG')[0]
        else:
            try:
                concepts_names[cui] = concept.get_preferred_names_for_language('ENG')[0] # more than unique cui in rel as it use all data
            except:
                concepts_names[cui] = 'Unknown English ' + str(i)
                i = i+1
    print("All {} concepts donnot have eng names".format(i))
    print("We have {} overlap entity in UMLS KG.".format(overlap_num))

    ehr_id_name.update(concepts_names) # 更新独立的UMLS实体
    print('Maybe not safe for replace as ICD10CM and ICD10PROC maybe overlap', len(ehr_id_name))
    ehr_id_name = {key.replace('.', ''): value for key, value in ehr_id_name.items()} # [{ATC, name}]
    print('Maybe not safe for replace', len(ehr_id_name)) # standard之后可能会导致总实体数量减少

    ehr_name_id = dict(zip(ehr_id_name.values(), ehr_id_name.keys()))

    # # 有EHR标识就替换为EHR标识
    concepts_ehr = dict(zip(all_ehr_concepts.values(), all_ehr_concepts.keys())) #[CID, EHR]
    concepts_ehr = {key: value.replace('.', '') for key, value in concepts_ehr.items()} # [{CID, EHR-nostand}]

    data['cui1'] = data['cui1'].map(concepts_ehr).fillna(data['cui1']) # 速度优化
    data['cui2'] = data['cui2'].map(concepts_ehr).fillna(data['cui2'])
    # data['cui1'] = data['cui1'].apply(lambda x: concepts_ehr[x] if x in concepts_ehr.keys() else x) # ehr rel ehr
    # data['cui2'] = data['cui2'].apply(lambda x: concepts_ehr[x] if x in concepts_ehr.keys() else x)

    # 对ATC进行替换
    def optimize_cui(cui_column):
        # 判断cui值是否在atc列表中,
        is_in_atc = cui_column.isin(atc_lis)
        # 根据条件选择适当的处理方式
        return cui_column.where(~is_in_atc, cui_column.str[:config['ATCLEVEL'] + 1])

    # 应用优化函数到相关列
    data['cui1'] = optimize_cui(data['cui1'])
    data['cui2'] = optimize_cui(data['cui2'])

    # data['cui1'] = data['cui1'].apply(lambda x: x[:config['ATCLEVEL']+1] if x in atc_lis else x)
    # data['cui2'] = data['cui2'].apply(lambda x: x[:config['ATCLEVEL']+1] if x in atc_lis else x)


    print("Second Step End ")
    # check 逻辑
    unique_data = set(data['cui1'].unique()) | set(data['cui2'].unique())
    print("We have {} unique entity in UMLS KG. Nonoverlap with ehr_id_name {}".format(len(unique_data), len(unique_data - set(ehr_id_name.keys()))))

    # 3. gpt
    unique_gptconcept_name = set(gpt_df['cui1'].unique()) | set(gpt_df['cui2'].unique())
    print("We have {} unique entity in GPT KG.".format(len(unique_gptconcept_name)))
    gpt_name_ehr = {} # {amosiline: ATCID, 头孢:G0001}
    overlap_num = 0
    for index, name in enumerate(unique_gptconcept_name):
        if name not in ehr_id_name.keys():
            gpt_name_ehr[name] = 'GPT000' + str(index)
        else:
            overlap_num += 1

    print("We have {} overlap entity in GPT KG.".format(overlap_num))

    # gpt_df['cui1'] = gpt_df['cui1'].apply(lambda x: gpt_name_ehr[x] if x in gpt_name_ehr.keys() else x) # {G001, EHR/CID}
    # gpt_df['cui2'] = gpt_df['cui2'].apply(lambda x: gpt_name_ehr[x] if x in gpt_name_ehr.keys() else x)

    gpt_df['cui1'] = gpt_df['cui1'].map(gpt_name_ehr).fillna(gpt_df['cui1'])
    gpt_df['cui2'] = gpt_df['cui2'].map(gpt_name_ehr).fillna(gpt_df['cui2'])



    ehr_gpt_name = dict(zip(gpt_name_ehr.values(), gpt_name_ehr.keys()))
    ehr_id_name.update(ehr_gpt_name)
    print("Third Step End ")

    # 4. save
    data = pd.concat([gpt_df[['cui1', 'cui2', 'rel']], data[['cui1', 'cui2', 'rel']]])

    save_path = os.path.join(config['KG_DATADIR'], dataset)

    data.to_pickle(save_path + '/triples_name.pkl')
    save_pickle(ehr_id_name, save_path + '/ehr_name_map.pkl')
    print(data.head(5))
    print("Now we have merge all datasets, and we have {} entity and {} relation !".format(len(ehr_id_name), data.shape[0]))
    print("All Step End!")

    return ehr_id_name, data


def gpt_to_triple(dataset):
    """先把gpt的triple sets也转换为编码"""
    # read all file
    drug = pd.read_csv(config['KG_DATADIR'] + 'drug_gpt3.csv')
    cond_1 = pd.read_csv(config['KG_DATADIR'] + 'con1_gpt3.csv')
    cond_2 = pd.read_csv(config['KG_DATADIR'] + 'con2_gpt3.csv')
    cond_3 = pd.read_csv(config['KG_DATADIR'] + 'con3_gpt3.csv')
    cond_4 = pd.read_csv(config['KG_DATADIR'] + 'con4_gpt3.csv')
    cond_5 = pd.read_csv(config['KG_DATADIR'] + 'con5_gpt3.csv')
    cond_6 = pd.read_csv(config['KG_DATADIR'] + 'con6_gpt3.csv')
    cond_7 = pd.read_csv(config['KG_DATADIR'] + 'con7_gpt3.csv')
    proc_1 = pd.read_csv(config['KG_DATADIR'] + 'proc1_gpt3.csv')
    proc_2 = pd.read_csv(config['KG_DATADIR'] + 'proc2_gpt3.csv')
    proc_3 = pd.read_csv(config['KG_DATADIR'] + 'proc3_gpt3.csv')
    proc_4 = pd.read_csv(config['KG_DATADIR'] + 'proc4_gpt3.csv')

    drug_triple = drug
    proc_triple = pd.concat([proc_1, proc_2, proc_3, proc_4])
    cond_triple = pd.concat([cond_1, cond_2, cond_3, cond_4, cond_5, cond_6, cond_7])
    print("GPT data load sucess, ", drug_triple.shape, proc_triple.shape, cond_triple.shape)
    tmp_df = pd.concat([drug_triple, proc_triple, cond_triple])

    # 检查
    print('GPT', tmp_df.head(5)) # ehr_index, ehr_name, triple
    unique = set(tmp_df['index'].values.tolist())
    ehr_id2name = {}
    if dataset == 'MIII':
        name_map_diag = get_node_name('ICD9CM')  # 这里需要同时替换所有的ID, 为ICD9和10分别处理一份吧，别慌。 居然有些ID没有
        name_map_proc = get_node_name('ICD9PROC')
    else:
        name_map_diag = get_node_name('ICD10CM')  # 这里需要同时替换所有的ID, 为ICD9和10分别处理一份吧，别慌。 居然有些ID没有
        name_map_proc = get_node_name('ICD10PROC')

    name_map_med = get_atc_name(config['ATCLEVEL']) # 前4位置
    ehr_id2name.update(name_map_diag)
    ehr_id2name.update(name_map_proc)
    ehr_id2name.update(name_map_med)
    ehr_id2name = {key.replace('.', ''): value for key, value in ehr_id2name.items()}  # [{ATC, name}]
    print("Len of overlap -gpt download", len(unique & set(ehr_id2name.keys()))) # 查看gpt查询的缺失情况

    # 1. replace name to ehr_index
    tmp_df['triple_new'] = tmp_df.apply(lambda row: row['triple'].replace(row["entity_name"], str(row['index'])), axis=1)
    tmp_df['triple_new'] = tmp_df['triple_new'].apply(lambda x: parse_str(x))    # 因为request名过长，无法识别为准确的entity，gpt将其分拆
    tmp_df['processed_triple'] = tmp_df['triple_new'].apply(lambda x: process_row(x))     # 搞一个模糊匹配,不然很多挺难识别的
    tmp_df['filter_triple'] = tmp_df.apply(fuzz_match, axis=1)
    print(tmp_df['processed_triple'].iloc[1])
    print(tmp_df['filter_triple'].iloc[1])

    filter_triple = tmp_df['filter_triple'].values.tolist() # [ehr_index, rel, gpt_name]
    new_df = [j for i in filter_triple for j in i]
    print(len(new_df))

    # 2. save
    gpt_df = pd.DataFrame(new_df, columns=['cui1', 'rel', 'cui2'])
    gpt_df = gpt_df.drop_duplicates()
    gpt_df['cui1'] = gpt_df['cui1'].str.replace('[', '', regex=False)
    gpt_df['cui1'] = gpt_df['cui1'].str.replace(']', '', regex=False)
    gpt_df['cui2'] = gpt_df['cui2'].str.replace('[', '', regex=False)
    gpt_df['cui2'] = gpt_df['cui2'].str.replace(']', '', regex=False)

    unique_cui1 = gpt_df['cui1'].unique()
    unique_cui2 = gpt_df['cui2'].unique()
    combined_unique_values = set(unique_cui1).union(set(unique_cui2))

    print("Len of overlap 2 - triple", len(combined_unique_values & set(ehr_id2name.keys())))

    print("Unique entity {}, Unique rel {}!".format(len(combined_unique_values), len(gpt_df['rel'].unique())))
    print(gpt_df.head(5))
    return gpt_df


def triple_save(dataset):
    """处理得到所需要的triple sets"""
    umls = UMLSParser('/home/czhaobo/KnowHealth/data/umls-extract')
    ehr_id2name, ehr_name2id = {}, {}
    if dataset == "MIII":
        name_map_diag = get_node_name('ICD9CM')  # 这里需要同时替换所有的ID, 为ICD9和10分别处理一份吧，别慌。 居然有些ID没有
        name_map_proc = get_node_name('ICD9PROC')
        name_map_med = get_atc_name(config['ATCLEVEL'])
        ehr_id2name.update(name_map_diag)
        ehr_id2name.update(name_map_proc)
        ehr_id2name.update(name_map_med)
        ehr_name2id = dict(zip(ehr_id2name.values(), ehr_name2id.keys()))
        gpt_df = gpt_to_triple(dataset)
    else:
        name_map_diag = get_node_name('ICD9CM')  # 这里需要同时替换所有的ID, 为ICD9和10分别处理一份吧，别慌。 居然有些ID没有
        name_map_proc = get_node_name('ICD9PROC')
        name_map_med = get_atc_name(config['ATCLEVEL'])
        name_map_diag2 = get_node_name('ICD10CM')  # 只有600多重叠，很少，即使有不对的也可以当噪声
        name_map_proc2 = get_node_name('ICD10PROC')
        name_map_med2 = get_atc_name(config['ATCLEVEL'])
        ehr_id2name.update(name_map_diag)
        ehr_id2name.update(name_map_diag2)
        ehr_id2name.update(name_map_proc)
        ehr_id2name.update(name_map_proc2)
        ehr_id2name.update(name_map_med)
        ehr_id2name.update(name_map_med2)
        ehr_name2id = dict(zip(ehr_id2name.values(), ehr_name2id.keys()))
        gpt_df = gpt_to_triple(dataset)

    print('Len individual ', len(name_map_diag) + len(name_map_proc) + len(name_map_med))
    print('Len all ', len(ehr_id2name)) # 是否在ehr一开始就有重叠【MIMIC-III是没有的；MIMIC-IV是有的】

    recode(umls, gpt_df, ehr_name2id, ehr_id2name, dataset)
    print("All Over, ", dataset)
    return

def tokenizer(dataset):
    if dataset == 'MIII':
        file_name = config['KG_DATADIR'] + 'MIII/triples_name.pkl'
        index_file = config['KG_DATADIR'] + 'MIII/ehr_name_map.pkl'
    else:
        file_name = config['KG_DATADIR'] + 'MIV/triples_name.pkl'
        index_file = config['KG_DATADIR'] + 'MIV/ehr_name_map.pkl'
    file = load_pickle(file_name)
    index_file = load_pickle(index_file)
    ehr_id = dict(zip(list(index_file.keys()), list(range(len(index_file)))))

    rel_unique = file['rel'].unique().tolist()
    rel_id = dict(zip(rel_unique, list(range(len(rel_unique)))))

    file['cui1'] = file['cui1'].apply(lambda x: ehr_id[x]) # ehr_id ()
    file['cui2'] = file['cui2'].apply(lambda x: ehr_id[x])
    file['rel'] = file['rel'].apply(lambda x: rel_id[x])

    # save
    save_path = os.path.join(config['KG_DATADIR'], dataset)

    file.to_pickle(save_path + '/triples_id.pkl')
    print("Save triples_id.pkl")
    save_pickle(ehr_id, save_path + '/ehr_id_map.pkl')
    save_pickle(rel_id, save_path + '/rel_id_map.pkl')

    print("All Transfer End!")
    return


if __name__ == '__main__':
    pass









