key: proofwriter
title: ProofWriter: Generating Implications, Proofs, and Abductive Statements over Natural Language
authors: Oyvind Tafjord, Bhavana Dalvi Mishra, Peter Clark
venue/year: Findings of ACL-IJCNLP 2021 (arXiv preprint v1: Dec 2020)
source url: https://aclanthology.org/2021.findings-acl.317/ (abstract text sourced from https://arxiv.org/abs/2012.13048, same paper)
tier: context
quality: abstract-only
verbatim: yes
fetch date: 2026-07-22
discrepancies: ACL Anthology listing renders author "Bhavana Dalvi" (matches expected citation); arXiv metadata lists "Mishra, Bhavana Dalvi" (same author, alternate name form). ACL Anthology page did not expose visible abstract text in the fetched HTML, so the verbatim abstract below was taken from the arXiv abstract page for the identical paper (title and content match).

## Abstract

Transformers have been shown to emulate logical deduction over natural language theories (logical rules expressed in natural language), reliably assigning true/false labels to candidate implications. However, their ability to generate implications of a theory has not yet been demonstrated, and methods for reconstructing proofs of answers are imperfect. In this work we show that a generative model, called ProofWriter, can reliably generate both implications of a theory and the natural language proof(s) that support them. In particular, iterating a 1-step implication generator results in proofs that are highly reliable, and represent actual model decisions (rather than post-hoc rationalizations). On the RuleTaker dataset, the accuracy of ProofWriter's proofs exceed previous methods by +9% absolute, and in a way that generalizes to proof depths unseen in training and on out-of-domain problems. We also show that generative techniques can perform a type of abduction with high precision: Given a theory and an unprovable conclusion, identify a missing fact that allows the conclusion to be proved, along with a proof. These results significantly improve the viability of neural methods for systematically reasoning over natural language.
