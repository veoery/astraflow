from __future__ import annotations

import json

import pytest

from astraflow.dataflow.dataset.terminal_bench import get_harbor_task_path_dataset
from astraflow.core.workflow.impl.terminal_bench_harbor import (
    TerminalBenchHarborRLWorkflow,
    TerminalBenchHarborWorkflow,
    _collect_harbor_rewards,
    _extract_reward_from_result,
    _load_harbor_trial_result,
)


def test_extract_reward_from_harbor_trial_result():
    result = {
        "verifier_result": {
            "rewards": {
                "reward": 1.0,
            }
        }
    }

    assert _extract_reward_from_result(result) == pytest.approx(1.0)


def test_collect_harbor_rewards_reads_trial_result_json(tmp_path):
    run_dir = tmp_path / "job" / "2026-05-15__17-17-54"
    result_file = run_dir / "trial-a" / "result.json"
    result_file.parent.mkdir(parents=True)
    result_file.write_text(
        json.dumps({"verifier_result": {"rewards": {"reward": 0.0}}})
    )

    with pytest.raises(RuntimeError, match="missing agent_result/verifier_result"):
        _collect_harbor_rewards(tmp_path / "job")

    result_file.write_text(
        json.dumps(
            {
                "agent_result": {},
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        )
    )

    assert _collect_harbor_rewards(tmp_path / "job") == [pytest.approx(1.0)]


def test_load_harbor_trial_result_uses_single_trial_result(tmp_path):
    aggregate = tmp_path / "job" / "2026-05-15__17-17-54" / "result.json"
    aggregate.parent.mkdir(parents=True)
    aggregate.write_text(json.dumps({"n_total_trials": 1, "stats": {}}))

    trial = aggregate.parent / "trial-a" / "result.json"
    trial.parent.mkdir(parents=True)
    trial.write_text(
        json.dumps(
            {
                "agent_result": {"rollout_details": []},
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        )
    )

    assert _load_harbor_trial_result(tmp_path / "job") == (
        trial,
        json.loads(trial.read_text()),
    )


def test_harbor_task_path_dataset_loads_skyrl_layout(tmp_path, monkeypatch):
    root = tmp_path / "CodeContests"
    task = root / "task-a"
    task.mkdir(parents=True)
    (task / "instruction.md").write_text("do task\n")
    monkeypatch.setenv("HARBOR_DATA", str(root))

    dataset = get_harbor_task_path_dataset(
        path="$HARBOR_DATA",
        dataset_name="test_harbor_tasks",
    )

    assert len(dataset) == 1
    assert dataset[0]["task_path"] == str(task)
    assert dataset[0]["prompt"] == str(task)
    assert dataset[0]["task_name"] == "task-a"


def test_build_command_supports_conda_wrapped_harbor(tmp_path):
    workflow = TerminalBenchHarborWorkflow(
        gconfig=object(),
        tokenizer=None,
        api_base="http://127.0.0.1:12345/v1",
        harbor_command=[
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "harbor-tb2",
            "harbor",
        ],
    )

    cmd = workflow._build_command("build-pmars", tmp_path)

    assert cmd[:7] == [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "harbor-tb2",
        "harbor",
        "run",
    ]
    assert "--include-task-name" in cmd
    assert "--yes" in cmd
    assert "build-pmars" in cmd
    assert "api_base=http://127.0.0.1:12345/v1" in cmd


def test_build_command_supports_harbor_task_path(tmp_path):
    task_dir = tmp_path / "task-a"
    task_dir.mkdir()
    workflow = TerminalBenchHarborWorkflow(
        gconfig=object(),
        tokenizer=None,
        api_base="http://127.0.0.1:12345/v1",
    )

    cmd = workflow._build_command(
        "task-a",
        tmp_path / "job",
        task_path=task_dir,
    )

    assert "--path" in cmd
    assert str(task_dir) in cmd
    assert "--include-task-name" not in cmd
    assert "task-a" not in cmd


def test_build_command_uses_configured_api_base(tmp_path):
    workflow = TerminalBenchHarborWorkflow(
        gconfig=object(),
        tokenizer=None,
        api_base="http://127.0.0.1:20001/v1",
    )

    cmd0 = workflow._build_command("task-a", tmp_path / "a")
    cmd1 = workflow._build_command("task-b", tmp_path / "b")

    assert "api_base=http://127.0.0.1:20001/v1" in cmd0
    assert "api_base=http://127.0.0.1:20001/v1" in cmd1


def test_resolve_api_base_errors_without_config_or_raas(tmp_path):
    # No configured api_base and no RaaS rollout context → clear error
    # (the legacy "infer from SGLang addresses" fallback was removed).
    workflow = TerminalBenchHarborWorkflow(gconfig=object(), tokenizer=None)
    with pytest.raises(RuntimeError, match="No model API base"):
        workflow._build_command("task-a", tmp_path)


def test_rl_workflow_disables_summarize_and_drops_rollout_details(tmp_path):
    workflow = TerminalBenchHarborRLWorkflow(
        gconfig=object(),
        tokenizer=None,
        api_base="http://127.0.0.1:12345/v1",
    )

    cmd = workflow._build_command("task-a", tmp_path)

    # Summarization stays off (it would break the ledger's linear-append
    # assumption). collect_rollout_details is no longer forced: token-level
    # training data now comes from the RaaS gateway ledger, not the harness.
    assert "enable_summarize=false" in cmd
    assert not any(c.startswith("collect_rollout_details=") for c in cmd)
