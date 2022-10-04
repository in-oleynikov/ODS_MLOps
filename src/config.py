import yaml

def load_config(path):
    with open('./src/config.yaml', 'r') as file:
        configs = yaml.safe_load(file)
    return configs

if __name__ == '__main__':
    path = './config.yaml'
    configs = load_config(path)