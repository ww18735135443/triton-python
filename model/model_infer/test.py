
from tools.parser import get_config
config_path='/home/ww/work/project/triton_project/config/triton_config.yaml'
cfg_input = get_config()
cfg_input.merge_from_file(config_path)
print(cfg_input)