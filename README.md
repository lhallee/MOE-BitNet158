# MOE-Bitformer
This repo combines implementations of Mixtral and BitNet1.58 into Bitformer - A Mixture of Experts version of BitNet. Supports GPT-like and BERT-like objectives with the `is_causal` setting in the config.

Very open to open-source collaboration for improving the repo! If it gets enough use I will wrap it into a Python package. Reach out via issue, pull requests, or contact me at `lhallee@udel.edu`

## Requirements:
pytorch, transformers, sentencepiece

Also flash-attn if you are going to use the SelfFlashAttention class - `config._attn_implementation = 'flash_attention_2'`

Training scripts, vision version, and documentation coming soon!

## TO DO:
- [x] MOE-bitnet - 3/9/24 - Logan
- [x] Training script with Huggingface trainer - 3/24/24 - Logan
- [ ] Produce convergence graphs
- [ ] Evaluation pipelines - MMLU, PubmedQA, etc.
- [ ] Fine-tuning pipelines - Classification
- [ ] Try protein language objectives
