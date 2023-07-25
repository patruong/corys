#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:33:32 2023

@author: ptruong
"""

import torch
from tape import ProteinBertModel, TAPETokenizer

model = ProteinBertModel.from_pretrained('bert-base')
tokenizer = TAPETokenizer(vocab='iupac')  # iupac is the vocab for TAPE models, use unirep for the UniRep model

def encode_seqeuence_BERT(sequence, model, tokenizer):
    # Pfam Family: Hexapep, Clan: CL0536
    # sequence = 'GCTVEDRCLIGMGAILLNGCVIGSGSLVAAGALITQ'
    
    token_ids = torch.tensor([tokenizer.encode(sequence)])
    output = model(token_ids)
    sequence_output = output[0]
    pooled_output = output[1]

    # NOTE: pooled_output is *not* trained for the transformer, do not use
    # w/o fine-tuning. A better option for now is to simply take a mean of
    # the sequence output

    return sequence_output.mean().item() # this is the mean of the sequence output as mentioned above...











