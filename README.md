# differentiable-proving
Codes for the Symbolic Math project. 
### Frozen Pre-Trained Transformer (FPT)
References: 
- [Pretrained Transformers as Universal Computation Engines.](https://github.com/kzl/universal-computation/blob/master/demo.ipynb)
- [Deep Learning For Symbolic Mathematics](https://github.com/facebookresearch/SymbolicMathematics)

Run `training.py` and `evaluator.py` (Python code) or `sym_math_FPT.ipynb` (Jupiter notebook)
### Encoder-Decoder Transformer 
Reference: 
- [Fine-tuning a model on a translation task (Hugging Face)](https://github.com/huggingface/notebooks/blob/master/examples/translation.ipynb)

Run `sym_math_end_dec.py` (Python code) or `sym_math_enc_dec.ipynb` (Jupiter notebook) (in Facebook folder above)
## Dependencies

- Python 3
- [NumPy](http://www.numpy.org/)
- [SymPy](https://www.sympy.org/)
- [PyTorch](http://pytorch.org/) 
- [Transformers](https://github.com/huggingface/transformers)

## Datasets 

| Dataset                       | #train     | Link                                                                            |
| ------------------------------|:----------:|:-------------------------------------------------------------------------------:|
| Integration (FWD)             |    45M     | [Link](https://dl.fbaipublicfiles.com/SymbolicMathematics/data/prim_fwd.tar.gz) |
| Integration (BWD)             |    88M     | [Link](https://dl.fbaipublicfiles.com/SymbolicMathematics/data/prim_bwd.tar.gz) |
| Integration (IBP)             |    23M     | [Link](https://dl.fbaipublicfiles.com/SymbolicMathematics/data/prim_ibp.tar.gz) |
| Differential equations (ODE1) |    65M     | [Link](https://dl.fbaipublicfiles.com/SymbolicMathematics/data/ode1.tar.gz)     |
| Differential equations (ODE2) |    32M     | [Link](https://dl.fbaipublicfiles.com/SymbolicMathematics/data/ode2.tar.gz)     |
