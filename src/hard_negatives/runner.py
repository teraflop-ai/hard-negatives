import argparse
import os
import shutil
import socket
import time

os.environ.setdefault("RAY_DEDUP_LOGS", "0")

import daft


def init_runner(use_ray: bool = True):
    if use_ray:
        import ray

        ray.init(ignore_reinit_error=True, log_to_driver=True)
        daft.set_runner_ray()
        print(f"Ray resources: {ray.cluster_resources()}", flush=True)


def parquet_files(path: str) -> list[str]:
    return sorted(
        os.path.join(root, f)
        for root, _, names in os.walk(path)
        for f in names
        if f.endswith(".parquet")
    )


def shard_files(files: list[str], task_id: int, num_tasks: int) -> list[str]:
    """Deterministic, size-balanced split of a file list across Slurm array tasks."""
    return sorted(files, key=os.path.getsize)[task_id::num_tasks]


def slurm_task() -> tuple[int, int]:
    env = os.environ.get
    return int(env("SLURM_ARRAY_TASK_ID", 0)), int(env("SLURM_ARRAY_TASK_COUNT", 1))


class SlurmRunner:
    def __init__(
        self,
        input: str,
        output: str,
        num_tasks: int | None = None,
        copy_input: bool = True,
    ):
        self.task_id, count = slurm_task()
        self.num_tasks = num_tasks or count
        self.input, self.copy_input = input, copy_input
        self.output = os.path.join(output, f"task_{self.task_id:05d}")

    @staticmethod
    def add_args(parser):
        parser.add_argument("--num_tasks", type=int)
        parser.add_argument(
            "--copy_input", action=argparse.BooleanOptionalAction, default=True
        )
        return parser

    def files(self) -> list[str]:
        return shard_files(parquet_files(self.input), self.task_id, self.num_tasks)

    def run(self, pipeline, use_ray: bool = True) -> None:
        started = time.monotonic()
        print(
            f"Task {self.task_id}/{self.num_tasks}: host={socket.gethostname()} pid={os.getpid()}",
            flush=True,
        )
        files = self.files()
        print(f"Selected {len(files)} files", flush=True)
        if not files:
            return
        if self.copy_input:
            os.makedirs("/tmp/input", exist_ok=True)
            files = [
                shutil.copy(f, f"/tmp/input/{i}_{os.path.basename(f)}")
                for i, f in enumerate(files)
            ]
        init_runner(use_ray)
        pipeline(daft.read_parquet(files)).write_parquet(
            self.output, write_mode="overwrite"
        )
        print(
            f"Completed task {self.task_id} in {time.monotonic() - started:.1f}s",
            flush=True,
        )
