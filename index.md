---
layout: default
---

Solving **symbolic mathematics** has always been of in the arena of human
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
![Octocat](assets/SymMath.png)

# Empirical Evaluation
In orther to examine the transfer from language translation to solving
symbolic math equations and attempt to understand better why this happens and which factors enable this transfer, we desighn the following research questions. we refer to [Lample & Charton (2019)](https://arxiv.org/abs/1912.01412)’s model results with the keyword LC in our tables and visualizations.
## i. **Does this pretrained model help us to use less data for training?**
As studied in Lample & Charton (2019), to train transformer architecture on the symbolic
math data, we need a vast amount of training data for each task to achieve the highest
accuracies (in the order of 40 million to 80 million training samples for each task.).  We can see in the following table that our model outperformed
in the integration task, with a considerable gap from the LC model. But it cannot properly
perform on the differential equation task, especially the second-order differential equations. 

|                   | Our Model | LC's Model |
|:-----------------:|:---------:|:----------:|
| Integration (FWD) | 87.4    | 79.4     |
| Integration (BWD) | 92.2    | 83.4    |
| Integration (IBP) | 86.2    | 87.4     |
| ODE 1           | 62.2    | 71.8     |
| ODE 2           | 17.9    | 39.9     |

The following figure extends this exploration by running the same experiment for different orders of magnitude
of training data (i.e., 10K, 100K, and 1M).  Our fine-tuned model has higher
accuracy in comparison to LC in all tasks and with different training sample sizes, except
that in the differential equations the accuracy growth of our model suddenly gets lower than
the LC model when using the 1 million samples for training.

![Octocat](assets/acc.png)

