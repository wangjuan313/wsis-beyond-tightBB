import yaml
import collections.abc
import copy


def read_config_file(file_name):
    with open(file_name) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data


def save_config_file(file_name, data_dict):
    with open(file_name, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


def config_updates(config,config_new):
	config_out = copy.deepcopy(config)
	for k,v in config_new.items():
		if isinstance(v,collections.abc.Mapping):
			config_out[k] = config_updates(config_out.get(k,{}),v)
		else:
			config_out[k] = v
	return config_out