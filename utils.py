# coding: utf-8
import math
import json
import datetime
import numpy as np
import pandas as pd

def processing_mimic3(file_adm, file_dxx, file_txx, file_drug, file_drg, output_file):
    m_adm = pd.read_csv(file_adm, dtype={'HOSPITAL_EXPIRE_FLAG': object}, low_memory=False)
    m_dxx = pd.read_csv(file_dxx, dtype={'ICD9_CODE': object}, low_memory=False)
    m_txx = pd.read_csv(file_txx, dtype={'ICD9_CODE': object}, low_memory=False)
    m_drg = pd.read_csv(file_drg, dtype={'DRG_CODE': object}, low_memory=False)
    m_drug = pd.read_csv(file_drug, dtype={'NDC': object}, low_memory=False)

    # get total unique patients
    unique_pats = m_dxx.SUBJECT_ID.unique()

    patients = []  # store all preprocessed patients' data
    print ('{} total number of patients:'.format(len(unique_pats)))
    for idx, sub_id in enumerate(unique_pats, start=1):
        patient = dict()
        patient['pid'] = str(sub_id)
        pat_dxx = m_dxx[m_dxx.SUBJECT_ID == sub_id]  # get a specific patient's all data in dxx file
        uni_hadm = pat_dxx.HADM_ID.unique()  # get all unique admissions
        grouped = pat_dxx.groupby(['HADM_ID'])
        visits = []
        for hadm in uni_hadm:
            act = dict()
            adm = m_adm[(m_adm.SUBJECT_ID == sub_id) & (m_adm.HADM_ID == hadm)]
            admsn_dt = datetime.datetime.strptime(adm.ADMITTIME.values[0], "%Y-%m-%d %H:%M:%S")
            disch_dt = datetime.datetime.strptime(adm.DISCHTIME.values[0], "%Y-%m-%d %H:%M:%S")
            death_flag = adm.HOSPITAL_EXPIRE_FLAG.values[0]

            delta = disch_dt - admsn_dt
            act['admsn_dt'] = admsn_dt.strftime("%Y%m%d")
            act['day_cnt'] = str(delta.days + 1)

            codes = grouped.get_group(hadm)  # get all diagnosis codes in the adm
            DXs = []
            for index, row in codes.iterrows():
                dx = row['ICD9_CODE']
                # if dx is not NaN
                if dx == dx:
                    DXs.append(dx)

            TXs = []
            pat_txx = m_txx[(m_txx.SUBJECT_ID == sub_id) & (m_txx.HADM_ID == hadm)]
            tx_codes = pat_txx.ICD9_CODE.values  # get all procedure codes in the adm
            for code in tx_codes:
                if code == code:
                    TXs.append(code)

            drugs = []
            pat_drugs = m_drug[(m_drug.SUBJECT_ID == sub_id) & (m_drug.HADM_ID == hadm)]
            drug_codes = pat_drugs.NDC.values  # get all drug codes in the adm
            for code in drug_codes:
                if code == code and code != '0':
                    drugs.append(code)

            drgs = []
            pat_drgs = m_drg[(m_drg.SUBJECT_ID == sub_id) & (m_drg.HADM_ID == hadm)]
            drg_codes = pat_drgs.DRG_CODE.values  # get all drug codes in the adm
            for code in drg_codes:
                if code == code:
                    drgs.append(code)

            act['DXs'] = DXs
            act['CPTs'] = TXs
            act['DRGs'] = drgs
            act['Drugs'] = drugs
            act['Death'] = death_flag
            visits.append(act)
        patient['visits'] = visits
        patients.append(patient)
        if math.log(idx, 2).is_integer():
            print ('{} patients are processed!'.format(idx))

    with open(output_file, 'w') as outfile:
        json.dump(patients, outfile)

import torch
from torch import nn
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaForMaskedLM
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaModel

class RobertaForICD9(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(config.hidden_size*3, self.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        
        cls = sequence_output[:, 0, :]
        
        pooled_output, _ = torch.max(sequence_output, 1)
        pooled_output = torch.relu(pooled_output)
        
        pooled_output_mean = torch.mean(sequence_output, 1)
        
        pooled_output = torch.cat((pooled_output, pooled_output_mean, cls), 1)
        
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)
        
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits( logits.view(-1), labels.view(-1) )
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

def get_score(gt_labels, pred, k=5):
    scores = []
    for gt_row, row in zip(gt_labels, pred):
        order = np.argsort(row)[::-1]

        pred = set(order[:k])

        gt = set(np.where(gt_row==1)[0])

        if len(gt)>0:
            numenator = len(gt.intersection(pred))
            score = numenator/min(k, len(gt))
            scores.append(score)
    return np.mean(scores)