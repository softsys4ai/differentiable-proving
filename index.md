---
layout: default
---

Solving symbolic mathematics has always been of in the arena of human
ingenuity that needs compositional reasoning and recurrence. However, recent studies have shown that large-scale language models such as **transformers are universal** and surprisingly can be trained as a sequence-to-sequence
task to solve complex mathematical equations. These large transformer
models need **humongous amounts of training data** to generalize to unseen
symbolic mathematics problems. In this work, we present a sample efficient way of solving the symbolic tasks by 
1. Pretraining the transformer
model with language translation and, 
2. Fine-tuning the pretrained transformer model to solve the downstream task of symbolic mathematics. 

We achieve comparable accuracy on the integration task with our pretrained
model while using around **1.5** orders of magnitude less number of training samples with respect to the state-of-the-art [deep learning for symbolic
mathematics](https://arxiv.org/abs/1912.01412). The test accuracy on differential equation tasks is **consider-
ably lower** comparing with integration as they need higher order recursions
that are not present in language translations. We pretrain our model with
different pairs of language translations. Our results show **language bias**
in solving symbolic mathematics tasks. Finally, we study the robustness
of the fine-tuned model on symbolic math tasks against distribution shift,
and our approach generalizes better in distribution shift scenarios for the
function integration.