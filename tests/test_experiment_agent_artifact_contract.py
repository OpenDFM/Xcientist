import os

from src.agents.experiment_agent.runtime.manifests import artifact_paths


def test_ablation_results_path_is_workspace_root(tmp_path):
    paths = artifact_paths(str(tmp_path))

    assert paths["agent_reports_dir"].endswith("agent_reports")
    assert paths["results_dir"].endswith("results")
    assert paths["ablation_results"] == os.path.join(str(tmp_path), "ablation_results.json")
    assert not paths["ablation_results"].startswith(paths["results_dir"] + os.sep)

