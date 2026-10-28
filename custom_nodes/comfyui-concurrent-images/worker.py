import concurrent.futures
import os
from pathlib import Path
import sys
import time

from engine import WorkerLock, process_item, read_json, redact, snapshot, write_json
from official import generate


def run(directory):
    directory = Path(directory)
    with WorkerLock(directory) as acquired:
        if not acquired:
            return
        manifest, states = snapshot(directory)
        write_json(directory / "worker.json", {"pid": os.getpid(), "started": time.time()})
        with concurrent.futures.ThreadPoolExecutor(max_workers=manifest["options"]["concurrency"]) as pool:
            futures = {pool.submit(process_item, directory, item["index"], generate): item["index"]
                       for item in states if item["state"] not in ("succeeded", "failed")}
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    path = directory / f"item-{index:06d}.json"
                    state = read_json(path)
                    state.update(state="failed", error="后台异常：" + redact(exc))
                    write_json(path, state)
        write_json(directory / "worker.json", {"finished": time.time()})


if __name__ == "__main__":
    run(sys.argv[1])
