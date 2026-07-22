key: rulebert
title: RuleBERT: Teaching Soft Rules to Pre-Trained Language Models
authors: Mohammed Saeed, Naser Ahmadi, Preslav Nakov, Paolo Papotti
venue/year: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP 2021), pp. 1460-1476
source url: https://aclanthology.org/2021.emnlp-main.110/ (abstract text sourced from https://arxiv.org/abs/2109.13006, same paper)
tier: context
quality: abstract-only
verbatim: yes
fetch date: 2026-07-22
discrepancies: None vs expected citation. ACL Anthology page did not expose visible abstract text in the fetched HTML, so the verbatim abstract below was taken from the corresponding arXiv abstract page (title matches exactly).

## Abstract

While pre-trained language models (PLMs) are the go-to solution to tackle many natural language processing problems, they are still very limited in their ability to capture and to use common-sense knowledge. In fact, even if information is available in the form of approximate (soft) logical rules, it is not clear how to transfer it to a PLM in order to improve its performance for deductive reasoning tasks. Here, we aim to bridge this gap by teaching PLMs how to reason with soft Horn rules. We introduce a classification task where, given facts and soft rules, the PLM should return a prediction with a probability for a given hypothesis. We release the first dataset for this task, and we propose a revised loss function that enables the PLM to learn how to predict precise probabilities for the task. Our evaluation results show that the resulting fine-tuned models achieve very high performance, even on logical rules that were unseen at training. Moreover, we demonstrate that logical notions expressed by the rules are transferred to the fine-tuned model, yielding state-of-the-art results on external datasets.
