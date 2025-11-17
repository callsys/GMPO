import os
import io
import sys
import time
import queue
import threading
import subprocess
import random
import copy


lock = threading.Lock()


class DualWriter(io.IOBase):
    def __init__(self, file_path: str, mode: str = 'w'):
        super().__init__()
        self.file = open(file_path, mode)
        self.record = []

    def write(self, data: str) -> int:
        self.file.write(data)
        self.record.append(data)
        return len(data)

    def flush(self) -> None:
        self.file.flush()

        lock.acquire(blocking=True)
        sys.stdout.write(".")
        sys.stdout.flush()
        lock.release()

    def close(self) -> None:
        self.file.close()

        lock.acquire(blocking=True)
        tag0 = "RUN#######################"
        s = "".join(self.record)
        s = "\n" + tag0 + s.split(tag0)[-1]
        sys.stdout.write(s)
        sys.stdout.flush()
        lock.release()

    @property
    def closed(self) -> bool:
        return self.file.closed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def run_cmd(
    env,
    cmd,
    stdout: io.TextIOWrapper,
    tag=None,
):
    def stream_reader(pipe, prefix: str):
        with pipe:
            for line in iter(pipe.readline, ""):  # 按行读取，直到 EOF
                stdout.write(f"{prefix}{line}")
                stdout.flush()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        bufsize=1,
    )

    prefix = f"{tag}" if tag else ""
    t_out = threading.Thread(target=stream_reader, args=(proc.stdout, prefix), daemon=True)
    t_err = threading.Thread(target=stream_reader, args=(proc.stderr, prefix), daemon=True)
    t_out.start()
    t_err.start()

    return_code = proc.wait()
    t_out.join()
    t_err.join()

    if return_code != 0:
        stdout.write(f"{prefix} failed with {return_code}\n")
        stdout.flush()


def run_tasks(
    cmd_list_queue: queue.Queue = None,
    gpu_id: int = 0,
    outdir: str = "",
):
    os.makedirs(outdir, exist_ok=True)
    while not cmd_list_queue.empty():
        obj = cmd_list_queue.get()
        env, cmd, tag = obj
        env = env.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # env["MASTER_PORT"] = str(13524 + random.randint(0, 5) * 16 + gpu_id)
        env["MASTER_PORT"] = "0"
        with DualWriter(os.path.join(outdir, f"tmp_{gpu_id}.log"), "a") as f:
            # f.write("\nENV#######\n")
            # f.write("\t".join(env))
            f.write("\nCMD#######\n")
            f.write(" ".join(cmd))
            f.write("\nRUN#######################\n")
            run_cmd(env=env, cmd=cmd, stdout=f)
            f.write("\nCMD#######\n")
            f.write(f"\nTAG:{tag}\n")
            f.write(" ".join(cmd))
            f.write("\n##########################\n")
            f.flush()


def main(cmds, gpu_ids=[0,], outdir="./outputs/gen_ppl/logs"):
    q = queue.Queue(maxsize=0)
    for c in cmds:
        q.put(c)
    
    threads = []
    for gpu_id in gpu_ids:
        t = threading.Thread(
            target=run_tasks,
            args=(q, gpu_id, outdir),
            daemon=True,
        )
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()


def run(path=""):
    ptags = [
        f"{path}/step_00{step:03d}"
        for step in range(16, 512, 16)
    ]
    cmds = []
    for ptag in ptags:
        cmd = ['python', '-u', 'utils/evaluation/evaluate_model.py', '--model_name', ptag]
        cmds.append((copy.deepcopy(os.environ), cmd, ptag))
    main(cmds, gpu_ids=list(range(8)), outdir=f"./outputs/evaloat/{path}")


if __name__ == "__main__":
    # run("oat-output/qwen2.5-Math-7b-drgrpo-qwenmathtemplate_1114T04:04:08/saved_models")
    run("oat-output/qwen2.5-Math-7b-drgrpo-qwenmathtemplate_1116T15:54:42/saved_models")


