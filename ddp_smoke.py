import os, sys, argparse, subprocess, tempfile, torch, torch.distributed as dist

def _child(backend, store_path):
    init_method = "file:///" + store_path.replace("\\", "/")
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )

    t = torch.tensor([1.0])
    dist.all_reduce(t)
    print(f"[RANK {rank}] world={world_size} sum={t.item():.1f}")
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc", type=int, default=2)
    args = parser.parse_args()

    backend = "gloo"  # Windows는 NCCL 없음
    store_file = os.path.join(tempfile.gettempdir(), "ddp_store_test")

    if os.environ.get("RANK") is None and args.nproc > 1:
        env_base = os.environ.copy()
        env_base["WORLD_SIZE"] = str(args.nproc)
        env_base["DDP_STORE_PATH"] = store_file

        procs = []
        for r in range(args.nproc):
            env = env_base.copy()
            env["RANK"] = str(r)
            procs.append(subprocess.Popen([sys.executable, __file__], env=env))
        for p in procs:
            p.wait()
        return

    if os.environ.get("RANK") is None:
        print("[single] run with --nproc N")
        return

    _child(backend, os.environ["DDP_STORE_PATH"])

if __name__ == "__main__":
    main()


