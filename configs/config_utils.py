from omegaconf import OmegaConf

def get_yaml_config(yaml_file):
    config = OmegaConf.load(yaml_file)
    return config
