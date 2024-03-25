import argparse

from utils import get_yaml, pretrain, finetine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml_path', type=str, default='yamls/small_bitformer.yaml')
    parser.add_argument('data_path', type=str, default='allenai/dolma')
    parser.add_argument('tokenizer_path', type=str, default='allenai/OLMo-7B')
    parser.add_argument('save_path', type=str, default='lhallee/bitformer_example')
    parser.add_argument('token', type=str, help='Huggingface token')
    parser.add_argument('pretrain', action='store_true', help='Pretrain or finetune')
    args = parser.parse_args()
    yargs = get_yaml(args.yaml_path)

    if args.pretrain:
        pretrain(args, yargs)
    else:
        finetine(args, yargs)


if __name__ == "__main__":
    main()