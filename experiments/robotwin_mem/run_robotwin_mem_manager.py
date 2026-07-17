import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SINGLE_ENTRY = PROJECT_ROOT / "experiments" / "robotwin_mem" / "eval_robotwin_mem_single.py"
EVAL_STEP_LIMIT_FILE = PROJECT_ROOT / "third_party" / "RoboTwin-Mem" / "task_config" / "_eval_step_limit.yml"
BENCH_NAME = "robotwin_mem"
DEFAULT_ROBOTWIN_MEM_ROOT = "third_party/RoboTwin-Mem"
DEMO_CLEAN_TASK_CONFIG = "demo_clean"
TERMINATE_TIMEOUT_SEC = 10
POLL_INTERVAL_SEC = 2


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 1 >= len(parts):
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<run_name>/..., got: {ckpt_path}"
            )
        run_name = parts[runs_idx + 1]
        if run_name == "":
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<run_name>/..., got: {ckpt_path}"
            )
        return f"{run_name}_{ckpt_path.stem}"
    return ckpt_path.stem


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    if key in {
        "ckpt",
        "gpu_id",
        "EVALUATION.task_name",
        "EVALUATION.task_config",
        "EVALUATION.output_dir",
        "EVALUATION.robotwin_root",
    }:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def _collect_worker_overrides() -> list[str]:
    return [ov for ov in HydraConfig.get().overrides.task if not _is_blocked_override(ov)]


def _ensure_demo_clean_requested(cfg: DictConfig) -> None:
    override_keys = {
        ov.split("=", 1)[0].lstrip("+~")
        for ov in HydraConfig.get().overrides.task
    }
    if "EVALUATION.task_config" not in override_keys:
        return
    requested = str(cfg.EVALUATION.task_config)
    if requested != DEMO_CLEAN_TASK_CONFIG:
        raise ValueError(
            "RoboTwin-Mem evaluation only supports "
            f"`EVALUATION.task_config={DEMO_CLEAN_TASK_CONFIG}`, got: {requested}"
        )


def _load_all_tasks() -> list[str]:
    if not EVAL_STEP_LIMIT_FILE.exists():
        raise FileNotFoundError(f"Task list file not found: {EVAL_STEP_LIMIT_FILE}")
    with EVAL_STEP_LIMIT_FILE.open("r", encoding="utf-8") as f:
        task_map = yaml.safe_load(f)
    if not isinstance(task_map, dict) or len(task_map) == 0:
        raise ValueError(f"Invalid task map in: {EVAL_STEP_LIMIT_FILE}")
    tasks = list(task_map.keys())
    # Keep original order and remove duplicates.
    seen = set()
    dedup_tasks: list[str] = []
    for task in tasks:
        if task in seen:
            continue
        seen.add(task)
        dedup_tasks.append(task)
    return dedup_tasks


def _parse_task_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text == "" or text.lower() in {"none", "null"}:
            return []
        return [item.strip() for item in text.split(",") if item.strip() != ""]
    if isinstance(value, (list, tuple, ListConfig)):
        tasks = [str(item).strip() for item in value if str(item).strip() != ""]
        return tasks
    raise TypeError(f"Unsupported MULTIRUN.task_names type: {type(value).__name__}")


def _parse_result_file(result_file: Path) -> dict[str, float | None]:
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    text = result_file.read_text(encoding="utf-8")
    success_rate: float | None = None
    reward: float | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        success_match = re.match(r"^Success Rate:\s*([-+0-9.eE]+)\s*$", stripped)
        if success_match is not None:
            success_rate = float(success_match.group(1))
            continue
        reward_match = re.match(r"^Reward:\s*([-+0-9.eE]+)\s*$", stripped)
        if reward_match is not None:
            reward = float(reward_match.group(1))
            continue
        if success_rate is None:
            try:
                success_rate = float(stripped)
            except ValueError:
                continue
    if success_rate is None:
        raise ValueError(f"Failed to parse success rate from: {result_file}")
    return {"success_rate": float(success_rate), "reward": reward}


def _task_result_candidates(run_output_dir: Path, task_name: str) -> list[Path]:
    task_dir = run_output_dir / task_name
    return [
        task_dir / "_result_demo_clean.txt",
        task_dir / "_result_clean.txt",
        task_dir / "_result.txt",
    ]


def _parse_task_result(run_output_dir: Path, task_name: str) -> dict[str, float | None]:
    candidates = _task_result_candidates(run_output_dir, task_name)
    for result_file in candidates:
        if result_file.exists():
            return _parse_result_file(result_file)
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Result file not found for task={task_name}. Tried: {tried}")


def _mean_or_none(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    if len(valid) == 0:
        return None
    return float(sum(valid) / len(valid))


def _to_jsonable(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


@dataclass
class RunningState:
    task_name: str
    gpu_id: int
    process: subprocess.Popen[str]


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if not SINGLE_ENTRY.exists():
        raise FileNotFoundError(f"Single evaluation entry not found: {SINGLE_ENTRY}")
    _ensure_demo_clean_requested(cfg)

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)

    robotwin_root_cfg = str(cfg.EVALUATION.robotwin_root)
    if robotwin_root_cfg == "third_party/RoboTwin":
        robotwin_root_cfg = DEFAULT_ROBOTWIN_MEM_ROOT
    robotwin_root = _resolve_path(robotwin_root_cfg, base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin-Mem root not found: {robotwin_root}")

    num_gpus = int(cfg.MULTIRUN.num_gpus)
    if num_gpus <= 0:
        raise ValueError("`MULTIRUN.num_gpus` must be > 0.")
    max_tasks_per_gpu = int(cfg.MULTIRUN.max_tasks_per_gpu)
    if max_tasks_per_gpu <= 0:
        raise ValueError("`MULTIRUN.max_tasks_per_gpu` must be > 0.")
    gpu_ids = list(range(num_gpus))

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    run_ts = output_dir.name
    if run_ts == "":
        raise ValueError(f"Invalid EVALUATION.output_dir (missing run_ts): {output_dir}")
    run_output_dir = PROJECT_ROOT / "evaluate_results" / BENCH_NAME / ckpt_tag / run_ts
    run_output_dir.mkdir(parents=True, exist_ok=True)

    manager_log = run_output_dir / "manager.log"
    failed_tasks_file = run_output_dir / "failed_tasks.txt"
    summary_csv = run_output_dir / "summary.csv"
    summary_json = run_output_dir / "summary.json"

    task_names_cfg = _parse_task_names(cfg.MULTIRUN.get("task_names", None))
    task_name_cfg = cfg.EVALUATION.task_name
    if len(task_names_cfg) > 0:
        tasks = task_names_cfg
    elif task_name_cfg is None or str(task_name_cfg).strip() == "":
        tasks = _load_all_tasks()
    else:
        tasks = [str(task_name_cfg)]

    seen_tasks: set[str] = set()
    duplicate_tasks: list[str] = []
    for task_name in tasks:
        if task_name in seen_tasks:
            duplicate_tasks.append(task_name)
        seen_tasks.add(task_name)
    if len(duplicate_tasks) > 0:
        raise ValueError(f"Duplicate task names in MULTIRUN.task_names: {duplicate_tasks}")

    extra_overrides = _collect_worker_overrides()

    task_results: dict[str, dict[str, float | None]] = {
        task: {"success_rate": None, "reward": None} for task in tasks
    }
    failed_records: list[dict[str, Any]] = []
    pending_tasks = deque(tasks)
    running_states: list[RunningState] = []

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def build_cmd(*, task_name: str, gpu_id: int) -> list[str]:
        cmd = [
            sys.executable,
            str(SINGLE_ENTRY),
            f"ckpt={str(ckpt_path)}",
            f"gpu_id={gpu_id}",
            f"EVALUATION.task_name={task_name}",
            f"EVALUATION.task_config={DEMO_CLEAN_TASK_CONFIG}",
            f"EVALUATION.output_dir={str(output_dir)}",
            f"EVALUATION.robotwin_root={str(robotwin_root)}",
        ]
        cmd.extend(extra_overrides)
        return cmd

    def launch_task(task_name: str, gpu_id: int) -> RunningState:
        cmd = build_cmd(task_name=task_name, gpu_id=gpu_id)
        log(
            f"launch task={task_name} task_config={DEMO_CLEAN_TASK_CONFIG} gpu={gpu_id} "
            f"cmd={' '.join(cmd)}"
        )
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            text=True,
        )
        return RunningState(
            task_name=task_name,
            gpu_id=gpu_id,
            process=process,
        )

    def terminate_all_running() -> None:
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            log(f"terminating task={state.task_name} gpu={state.gpu_id}")
            state.process.terminate()
        deadline = time.time() + TERMINATE_TIMEOUT_SEC
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            remaining = max(0.0, deadline - time.time())
            try:
                state.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                log(f"killing task={state.task_name} gpu={state.gpu_id}")
                state.process.kill()
                state.process.wait()

    def gpu_running_count(gpu_id: int) -> int:
        count = 0
        for state in running_states:
            if state.gpu_id != gpu_id:
                continue
            if state.process.poll() is None:
                count += 1
        return count

    def try_launch_pending(gpu_id: int) -> None:
        while len(pending_tasks) > 0 and gpu_running_count(gpu_id) < max_tasks_per_gpu:
            task_name = pending_tasks.popleft()
            running_states.append(launch_task(task_name=task_name, gpu_id=gpu_id))

    def write_outputs() -> None:
        success_mean = _mean_or_none([task_results[t]["success_rate"] for t in tasks])
        reward_mean = _mean_or_none([task_results[t]["reward"] for t in tasks])

        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_name", "demo_clean_success_rate", "reward"])
            for task in tasks:
                writer.writerow(
                    [
                        task,
                        task_results[task]["success_rate"],
                        task_results[task]["reward"],
                    ]
                )
            writer.writerow(["__overall__", success_mean, reward_mean])

        payload = {
            "per_task": [
                {
                    "task_name": task,
                    "demo_clean_success_rate": _to_jsonable(task_results[task]["success_rate"]),
                    "reward": _to_jsonable(task_results[task]["reward"]),
                }
                for task in tasks
            ],
            "overall": {
                "demo_clean_mean_success_rate": _to_jsonable(success_mean),
                "mean_reward": _to_jsonable(reward_mean),
            },
        }
        summary_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with failed_tasks_file.open("w", encoding="utf-8") as f:
            for rec in failed_records:
                f.write(
                    f"{rec['task_name']},gpu={rec['gpu_id']},"
                    f"return_code={rec['return_code']},reason={rec['reason']}\n"
                )

    log(
        f"manager start tasks={len(tasks)} gpu_ids={gpu_ids} "
        f"max_tasks_per_gpu={max_tasks_per_gpu} output_dir={run_output_dir}"
    )

    # Launch initial tasks for each GPU up to capacity.
    for gpu_id in gpu_ids:
        try_launch_pending(gpu_id)

    has_failure = False
    failure_message = ""

    while len(running_states) > 0:
        progressed = False
        for state in list(running_states):
            gpu_id = state.gpu_id
            return_code = state.process.poll()
            if return_code is None:
                continue
            progressed = True
            running_states.remove(state)

            if return_code != 0:
                has_failure = True
                failure_message = (
                    f"worker failed: task={state.task_name}, gpu={gpu_id}, "
                    f"return_code={return_code}"
                )
                failed_records.append(
                    {
                        "task_name": state.task_name,
                        "gpu_id": gpu_id,
                        "return_code": return_code,
                        "reason": "process_failed",
                    }
                )
                log(failure_message)
                terminate_all_running()
                running_states.clear()
                break

            try:
                result = _parse_task_result(run_output_dir, state.task_name)
            except Exception as exc:
                has_failure = True
                failure_message = (
                    f"result parse failed: task={state.task_name}, gpu={gpu_id}, "
                    f"error={repr(exc)}"
                )
                failed_records.append(
                    {
                        "task_name": state.task_name,
                        "gpu_id": gpu_id,
                        "return_code": return_code,
                        "reason": "result_parse_failed",
                    }
                )
                log(failure_message)
                terminate_all_running()
                running_states.clear()
                break

            task_results[state.task_name]["success_rate"] = result["success_rate"]
            task_results[state.task_name]["reward"] = result["reward"]
            log(
                f"done task={state.task_name} task_config={DEMO_CLEAN_TASK_CONFIG} "
                f"gpu={gpu_id} success_rate={result['success_rate']:.4f} "
                f"reward={result['reward']}"
            )

            try_launch_pending(gpu_id)

        if has_failure:
            break
        if not progressed:
            time.sleep(POLL_INTERVAL_SEC)

    # Mark not started tasks when failure happened.
    if has_failure:
        for task_name in pending_tasks:
            failed_records.append(
                {
                    "task_name": task_name,
                    "gpu_id": -1,
                    "return_code": -1,
                    "reason": "aborted_not_started",
                }
            )

    write_outputs()
    log(f"summary saved: {summary_csv} and {summary_json}")

    if has_failure:
        raise RuntimeError(failure_message)

    log("manager finished successfully")


if __name__ == "__main__":
    main()
