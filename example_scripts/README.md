## Train models from the paper "[Linear Transformers are Secretly Fast Weight Memory Systems]()"

To train the baseline Transformer model, run
```
bash small_standard_transformer_baseline.sh train --work_dir MY_DIR
```

After training, run the following evaluation commands to obtain perplexity values similar to those reported in the paper:
```
bash small_standard_transformer_baseline.sh valid --work_dir MY_DIR/WHERE_MY_MODEL_IS  
bash small_standard_transformer_baseline.sh eval --work_dir MY_DIR/WHERE_MY_MODEL_IS
```

The instructions above applies to all other scripts.

### Remarks
* **Important**: Train/valid perplexity values displayed during training are not exact in general,
in the sense that the part of the text which does not fit to batch-size/backpropagation-span is discarded.
In addition, for models with a limited context size,
the perplexity is computed by splitting the text into segments which are treated independently (during training).
The model thus has no context at the beginning of a new segment.
This is avoided in the evaluation commands above by using a sliding window.

* The perplexity computation with the evaluation commands above is done with the batch size of 1. If this needs to be changed, simply add `--batch_size` in the evaluation command (or edit it in the script).

* All scripts use 2 GPUs. We used two V100 machines with 16GB memory.
If you need to train on less machines or machines with less meomory, the batch size has to be reduced.

* If you do not use Weights & Biases, remove the corresponding flags:
```
--use_wandb 
--project_name 
```
If you use them, you can optionally also specify the job name via
```
--job_name
```
By default (recommended) a job name containing the corresponding hyper-parameters is automatically generated.

* All language modeling experiments were conducted using this toolkit:
we gave different index/number to different combinations of
linear attention types (standard Transformer, Linear, Performer, our DPFP) and update rules (sum vs. ours),
which we can specify via `attn_type`. See examples in [small_linear_fast_weight_memory.sh](https://github.com/IDSIA/lmtool-fwms/blob/master/example_scripts/small_linear_fast_weight_memory.sh)

* We used flags to specify extra model variants such as the ones without the positional encoding (`no_pos`) or the attention normalisation (`skip_attn_normalization`), or training and evaluation without trucating the context (`carry_over_fast_weight`).

