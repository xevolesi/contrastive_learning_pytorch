import yaml
import addict
from typeguard import typechecked


@typechecked
def read_config(config_path: str) -> addict.Dict:
    with open(config_path, "r") as config_file_stream:
        yaml_file = yaml.safe_load(config_file_stream)
    return addict.Dict(yaml_file)
