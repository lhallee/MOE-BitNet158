# MOE-Bitformer
This repo combines implementations of Mixtral and BitNet1.58 into Bitformer - A Mixture of Experts version of BitNet. Supports GPT-like and BERT-like objectives with the `is_causal` setting in the config.

Very open to open-source collaboration for improving the repo! If it gets enough use I will wrap it into a Python package. Reach out via issue, pull requests, or contact me at `lhallee@udel.edu`

## Requirements:
pytorch, transformers, sentencepiece

Training scripts, vision version, and documentation coming soon!

## TO DO:
- [x] MOE-bitnet - 3/9/24 - Logan Hallee 
- [ ] Training script with huggingface trainer
- [ ] Training script with normal pytorch loop
- [ ] Vision implementation
- [ ] Data streaming using Dolma
- [ ] Produce convergence graphs
- [ ] Evaluation pipelines - MMLU, PubmedQA, etc.
- [ ] Try protein language objectives
