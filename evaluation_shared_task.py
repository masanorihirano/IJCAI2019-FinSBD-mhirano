#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 16:00:17 2019

@author: Abderrahim
"""
import json
from sklearn.metrics import classification_report

    
def label_seq(data):
    """
    Input: 
        data: dictionary of text, begin_sentences, end_sentences
    Output:
        a sequence of labels where each token from the text is assiciated with a label:
            regular token --> O
            begin sentences token --> BS
            end sentences token --> ES
    """
    text = data["text"].split(" ")
    
    True_Begin_sentences = data["begin_sentence"]
    True_End_sentences = data["end_sentence"]
    
    labels_train = ["O"] * len(text)    
    for index in True_Begin_sentences:
        labels_train[index] ="BS"
    for index in True_End_sentences:
        labels_train[index] ="ES"
    return labels_train


def evaluate_result(data_true,data_pred):
    """
    Report the score of Begin sentence and end sentences and regular labels
    NB : Only F1 score of BS and ES will be taken under account in the evaluation (first two lines of the report)
    data_true: a json file of the ground truth (rain_fr.json for instance)
    data_pred is a json file in the same format as the json data files
    """
    labels_true = label_seq(data_true)       
    labels_pred = label_seq(data_pred)

    
    target_names= ["O","BS","ES"]
    tag2idx = {t: i for i, t in enumerate(target_names)}
    
    y_true = [tag2idx[i] for i in labels_true]
    y_pred = [tag2idx[i] for i in labels_pred]
    print(classification_report(y_true, y_pred, target_names=target_names))

    
