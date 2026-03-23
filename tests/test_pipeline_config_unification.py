import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

from src.config import load_config
from src.pipeline.experiment_to_symbolic import normalize_component_family


def test_pipeline_unified_config_contains_symbolic_memory_fields():
    config = load_config("src/config/default.yaml")

    assert str(config.pipeline.symbolic_memory_path) == "idea_skill_priors"
    assert "generator" in list(config.pipeline.default_macro_roles)


def test_normalize_component_family_uses_unified_pipeline_roles():
    family = normalize_component_family("flow_matching_generator")
    assert family == "generator.flow_matching"
