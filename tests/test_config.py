from source.utils.general import read_config


CONFIG_PATH = "source/config.yml"


def test_read_config():
    config = read_config(CONFIG_PATH)
    assert isinstance(config.project_name, str)
