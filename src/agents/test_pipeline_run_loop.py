import json

import pytest
from omegaconf import OmegaConf

from src.pipeline import run_loop


def _minimal_runtime_config(tmp_path):
    survey_dir = tmp_path / "survey"
    return OmegaConf.create(
        {
            "survey": {
                "BasicInfo": {
                    "base_dir": str(survey_dir),
                    "save_path": str(survey_dir / "survey.md"),
                    "save_json_path": str(survey_dir / "survey.json"),
                    "evaluation_save_path": str(survey_dir / "evaluation.txt"),
                }
            },
            "idea": {
                "run": {
                    "topic": "default topic",
                    "mature_idea": "default mature idea",
                }
            },
        }
    )


def test_run_experiment_ignores_skip_prepare_flag(monkeypatch, tmp_path):
    captured = {}

    def fake_run_command(cmd, env=None):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(env or {})
        return 0

    monkeypatch.setattr(run_loop, "run_command", fake_run_command)

    ok = run_loop.run_experiment(
        experiment_id="demo-exp",
        workspace_root=str(tmp_path),
        max_agent_iterations=3,
        resume=True,
        skip_prepare=True,
    )

    assert ok is True
    assert "--resume" in captured["cmd"]
    assert "--skip-prepare" not in captured["cmd"]
    assert captured["env"]["EXPERIMENT_AGENT_WORKSPACE_DIR"] == str(tmp_path)


def test_run_idea_materializes_runtime_config_and_uses_run_dir_fallback(monkeypatch, tmp_path):
    config = _minimal_runtime_config(tmp_path)
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    survey_dir = tmp_path / "survey-output"
    survey_dir.mkdir()
    ablation_dir = tmp_path / "ablation"
    ablation_dir.mkdir()
    previous_candidate_path = tmp_path / "previous_candidate.json"
    previous_candidate_path.write_text("{}", encoding="utf-8")

    run_dir = tmp_path / "idea-run"
    run_dir.mkdir()
    payload = {
        "title": "Demo idea",
        "abstract": "A pipeline-aligned idea.",
        "method": "Use fallback result-dir resolution.",
    }
    (run_dir / "idea_result.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    (run_dir / "idea_candidate.json").write_text("{}", encoding="utf-8")

    captured = {}

    class _FakeProcess:
        def __init__(self, cmd, stdout=None, stderr=None, text=None, bufsize=None, env=None):
            captured["cmd"] = list(cmd)
            captured["env"] = dict(env or {})
            self.stdout = iter([f"[demo] ✅ completed -> {run_dir}\n"])
            self.returncode = 0

        def wait(self):
            return self.returncode

    monkeypatch.setattr(run_loop.subprocess, "Popen", _FakeProcess)

    output_file = runtime_dir / "copied_idea_result.json"
    result = run_loop.run_idea(
        topic="New Topic",
        mature_idea="New Mature Idea",
        output_file=str(output_file),
        config=config,
        runtime_dir=str(runtime_dir),
        survey_output_dir=str(survey_dir),
        ablation_results_path=str(ablation_dir),
        previous_candidate_path=str(previous_candidate_path),
    )

    assert output_file.exists()
    assert result["payload"] == payload
    assert result["idea_path"] == str(run_dir / "idea_result.json")
    assert result["candidate_path"] == str(run_dir / "idea_candidate.json")
    assert captured["env"]["IDEA_AGENT_ABLATION_RESULTS_PATH"] == str(ablation_dir)
    assert captured["env"]["IDEA_AGENT_PREVIOUS_CANDIDATE_PATH"] == str(previous_candidate_path)

    runtime_config = OmegaConf.load(captured["env"]["IDEA_AGENT_CONFIG"])
    assert runtime_config.idea.run.topic == "New Topic"
    assert runtime_config.idea.run.mature_idea == "New Mature Idea"
    assert runtime_config.survey.BasicInfo.base_dir == str(survey_dir)
    assert runtime_config.survey.BasicInfo.save_json_path == str(survey_dir / "survey.json")


def test_run_idea_raises_on_failure_marker_even_with_zero_exit(monkeypatch, tmp_path):
    config = _minimal_runtime_config(tmp_path)
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()

    class _FakeProcess:
        def __init__(self, cmd, stdout=None, stderr=None, text=None, bufsize=None, env=None):
            self.stdout = iter(["[demo] ❌ failed: synthetic failure\n"])
            self.returncode = 0

        def wait(self):
            return self.returncode

    monkeypatch.setattr(run_loop.subprocess, "Popen", _FakeProcess)

    with pytest.raises(RuntimeError, match="reported failure"):
        run_loop.run_idea(
            topic="New Topic",
            mature_idea="New Mature Idea",
            output_file=str(runtime_dir / "copied_idea_result.json"),
            config=config,
            runtime_dir=str(runtime_dir),
            survey_output_dir=str(tmp_path / "survey-output"),
        )
