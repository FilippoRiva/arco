import os
from dataclasses import fields
from pathlib import Path

import pytest

from arco.core import ArcoConfig, AgentType, AgentConfig
from arco.data import DatabaseSchema

ROOT_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = ROOT_DIR / "data"

@pytest.fixture
def complete_config_path(tmp_path):
    config = f"""
    global:
        provider: "ollama" 
        model: "my_custom_model"
        orchestration_enabled: true
        caching_enabled: true
        schema: "{DATA_DIR.absolute()}"  # path to YAML schemas for the database
        
    agents:
        Retriever:
            n: 2
            bon_parameter: "temperature"
            temp_min: 0.2
            temp_max: 0.8
        Analyzer:
            n: 3
            bon_parameter: "top_p"
            top_p_min: 0.90
            top_p_max: 0.95
        Visualizer:
            n: 2
            bon_parameter: "top_k"
            top_k_min: 64            
            top_k_max: 128

    run:
        prompt: "Show me the sales in Nov 2021"
        visualization_goal: "Sales trend for Nov 2021, with date as x and sales value as y"
    """

    config_file = tmp_path / "complete.yml"
    config_file.write_text(config)
    return str(config_file.absolute())


@pytest.fixture
def base_config_path(tmp_path):
    minimal = f"""
    global:
        provider: "openai" 
        model: "openai_model" 
        schema: "{DATA_DIR.absolute()}"  # path to YAML schemas for the database
            
    run:
        prompt: "Show me the sales in Nov 2021"
        visualization_goal: "Sales trend for Nov 2021, with date as x and sales value as y"
    """

    config_file = tmp_path / "minimal.yml"
    config_file.write_text(minimal)
    return str(config_file.absolute())

@pytest.fixture
def minimal_config(base_config_path):
    return ArcoConfig.from_yaml(base_config_path)

@pytest.fixture
def complete_config(complete_config_path):
    return ArcoConfig.from_yaml(complete_config_path)

def test_yaml_initialization(base_config_path, complete_config_path):

    # using the pytest tmp_path fixture to create a temp YAML file from a string
    os.environ['OPENAI_API_KEY'] = "test_api_key"
    config = ArcoConfig.from_yaml(base_config_path)
    complete_config = ArcoConfig.from_yaml(complete_config_path)

    def check_config(conf:ArcoConfig):
        for agent_type in AgentType:
            agent_config = conf.get_agent_config(agent_type)
            assert isinstance(agent_config,
                              AgentConfig), "Should not be None and if not provided defaults should be loaded"
        assert isinstance(conf.schema, DatabaseSchema), "The schema should be initialized"
        assert conf.model is not None, "Should be specified"
        assert conf.provider in ["openai", "ollama"], "Should be a known provider"
        if conf.provider == "ollama":
            assert conf.ollama_url is not None, "Should be set"
        for agent_type, agent_config in conf.agent_configs.items():
            assert isinstance(agent_config, AgentConfig), "Should be an AgentConfig"
            assert agent_type in AgentType, "Should be an AgentType"
        assert isinstance(conf.config_path, str), "Should be a str"

    check_config(config)
    check_config(complete_config)

def test_setters_and_getters(base_config_path):
    n = 100
    agent_config = AgentConfig.from_yaml(base_config_path, "my_custom_agent")
    agent_config.n = n

    for agent_type in AgentType:
        config = ArcoConfig.from_yaml(base_config_path)
        original_config = config.get_agent_config(agent_type)
        config.set_agent_config(agent_type, agent_config)
        final_config = config.get_agent_config(agent_type)

        assert original_config != final_config
        assert final_config == agent_config
        assert final_config.n == n

def test_copy(minimal_config):
    conf = minimal_config
    copy_conf = conf.copy()
    assert conf == copy_conf # deep checks
    assert conf is not copy_conf # class check

def test_candidate_parameters(complete_config):
    for agent_type in AgentType:
        agent_config = complete_config.get_agent_config(agent_type)

        # Map the active Best-of-N parameter to its index inside the tuple
        bon_param = agent_config.bon_parameter
        bon_idx = {"temperature": 0, "top_p": 1, "top_k": 2}[bon_param]

        candidates = agent_config.get_candidate_params()
        temps, top_ps, top_ks = zip(*candidates)
        columns = [temps, top_ps, top_ks]

        for idx in range(3):
            column_values = columns[idx]

            if agent_config.n >= 1 and idx == bon_idx:
                assert len(set(column_values)) == len(column_values), (
                    f"BON parameter '{bon_param}' values should be completely unique"
                )
            else:
                assert len(set(column_values)) == 1, (
                    f"Non-BON parameter at index {idx} should be identical across all candidates"
                )


