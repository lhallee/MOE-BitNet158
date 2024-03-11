import yaml


def get_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        args = yaml.safe_load(file)
    return args