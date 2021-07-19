# differentiable-proving

Codes for the Symbolic Math project.

### Encoder-Decoder Transformer

Reference:

- [Fine-tuning a model on a translation task (Hugging Face)](https://github.com/huggingface/notebooks/blob/master/examples/translation.ipynb)

Run `training.py` and `evaluator.py` (Python code)

## Dependencies

- Python 3
- [NumPy](http://www.numpy.org/)
- [SymPy](https://www.sympy.org/)
- [PyTorch](http://pytorch.org/)
- [Transformers](https://github.com/huggingface/transformers)

## Datasets

| Dataset                       | #train |                                      Link                                       |
| ----------------------------- | :----: | :-----------------------------------------------------------------------------: |
| Integration (FWD)             |  45M   | [Link](https://dl.fbaipublicfiles.com/SymbolicMathematics/data/prim_fwd.tar.gz) |
| Integration (BWD)             |  88M   | [Link](https://dl.fbaipublicfiles.com/SymbolicMathematics/data/prim_bwd.tar.gz) |
| Integration (IBP)             |  23M   | [Link](https://dl.fbaipublicfiles.com/SymbolicMathematics/data/prim_ibp.tar.gz) |
| Differential equations (ODE1) |  65M   |   [Link](https://dl.fbaipublicfiles.com/SymbolicMathematics/data/ode1.tar.gz)   |
| Differential equations (ODE2) |  32M   |   [Link](https://dl.fbaipublicfiles.com/SymbolicMathematics/data/ode2.tar.gz)   |
