## PyTorch Language Modeling Toolkit (for Fast Weight Programmers)

This repository contains the official code used for language modeling experiments in the paper(s):
* [Linear Transformers are Secretly Fast Weight Programmers (ICML 2021)](https://arxiv.org/abs/2102.11174)
* [Going Beyond Linear Transformers with Recurrent Fast Weight Programmers](https://arxiv.org/abs/2106.06295)
* ...

More generally, this can be used as a language modeling toolkit in PyTorch to experiment with:
* [Standard Transformers](https://arxiv.org/abs/1808.04444)
* [Transformer-XL](https://arxiv.org/abs/1901.02860)
* **Fast Weight Programmers** with different **update rules** and **linear attention functions**:
    * Update rules: "sum" and our "delta" rule (as proposed in our paper; Sec 4.2)
    * Linear attention functions: "ELU-based" linear attention, "FAVOR+", "deterministic parameter-free projection (DPFP)"
    
    e.g. some combinations result in well known models:
    * [Linear Transformers](https://arxiv.org/abs/2006.16236) = "sum" update rule + "ELU-based" linear attention
    * [Performers](https://arxiv.org/abs/2009.14794) = "sum" update rule + "FAVOR+"

## Fast Weight Implementations
This repositiory contains two implementations of fast weights.
* Custom cuda kernel (see [utils/fast_fast_weight](https://github.com/IDSIA/lmtool-fwms/tree/master/src/utils/fast_fast_weight) and [utils/cuda_fast_weight_layer.py](https://github.com/IDSIA/lmtool-fwms/blob/master/src/utils/cuda_fast_weight_layer.py))
* Custom `torch.autograd.Function` (see [utils/fast_weight.py](https://github.com/IDSIA/lmtool-fwms/blob/master/src/utils/fast_weight.py))

While we only used the cuda implementation for all our final experiments (faster/much better GPU utilization),
`torch.autograd.Function` version can be useful for a quick prototyping with new extensions.

## Requirements
This toolkit requires PyTorch `torch` and Ninja `ninja` (to compile the cuda kernels).

The experiments for the paper were conducted with Python 3.6 and PyTorch 1.4.0 (note on Aug 24, 2023: the code also works with Python 3.11 and PyTorch 2.0.1+cu117).

More recent versions of PyTorch are not yet well supported by this toolkit which still uses `torch.nn.DataParallel` for multi-GPU training.
If you really need to use a more recent version of PyTorch, check the [documentation](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)
to use `torch.nn.parallel.DistributedDataParallel` instead. We will hopefully fix this soon, but we cannot tell exactly when.

The toolkit supports [Weights & Biases](https://docs.wandb.ai/) for monitoring jobs. If you use it, also install `wandb`.

## Acknowledgements
This reposity contains many lines of code taken and adapted from the following sources:
* This reposity was originally forked from the official implementation of Transformer-XL [kimiyoung/transformer-xl](https://github.com/kimiyoung/transformer-xl).
The code for Transformer-XL and standard Transformer models, as well as basic functionality needed for language modeling
(including adaptive input and output embeddings) and data preparation (WikiText-103, enwik8, ...) is from the corresponding repository.
* For Performers, helper functions from [lucidrains/performer-pytorch](https://github.com/lucidrains/performer-pytorch) are used.
* For cuda implementations of our fast weight programmers with the delta rule:
    * Code from [idiap/fast-transformers](https://github.com/idiap/fast-transformers/tree/master/fast_transformers/causal_product) is used with minor changes for the sum update rule.
    * We modified it to implement our update rule.
See comments in code for exact locations and modifications.

## General Instructions

Please check files under `example_scripts` for general instructions and examples to train and evaluate models. 


## BibTex

```
@inproceedings{schlag2021linear,
      title={Linear Transformers Are Secretly Fast Weight Programmers}, 
      author={Imanol Schlag and Kazuki Irie and J\"urgen Schmidhuber},
      booktitle={Proc. Int. Conf. on Machine Learning (ICML)},
      address = {Virtual only},
      month = jul,
      year={2021}
}
```

```
@article{irie2021going,
      title={Going Beyond Linear Transformers with Recurrent Fast Weight Programmers}, 
      author={Kazuki Irie and Imanol Schlag and R\'obert Csord\'as and J\"urgen Schmidhuber},
      journal={Preprint arXiv:2106.06295},
      year={2021}
}
```

## Links
* The code for synthetic retrieval experiments in the paper ["Linear Transformers are Secretly Fast Weight Programmers" (ICML 2021)](https://arxiv.org/abs/2102.11174) can be found at [ischlag/fast-weight-transformers](https://github.com/ischlag/fast-weight-transformers).
* The full repository for the paper "Going Beyond Linear Transformers with Recurrent Fast Weight Programmers" can be found at: [IDSIA/recurrent-fwp](https://github.com/IDSIA/recurrent-fwp)
* [Jürgen Schmidhuber's AI blog post on Fast Weight Programmers (March 26, 2021)](https://people.idsia.ch/~juergen/fast-weight-programmer-1991-transformer.html).
