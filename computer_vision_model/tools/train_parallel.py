import torch.multiprocessing as mp
from train_fold_worker import train_fold_loop
from aggregate import compute_global_and_avg_zone_r2
import yaml

def main():
    n_folds = 5
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    manager = mp.Manager()
    shared_dict = manager.dict()  # stores val preds for the current epoch only
    save_flag = manager.Value('i', 0)
    barrier = mp.Barrier(parties=6)
    best_r2 = float("-inf")
    # Launch one process per fold
    processes = []
    for fold in range(n_folds):
        p = mp.Process(
            target=train_fold_loop,
            args=(fold, config, shared_dict, barrier, save_flag)
        )
        p.start()
        processes.append(p)
    # Loop over epochs
    for epoch in range(config["training"]["epochs"]):
        print(f"\n[Main] Waiting for epoch {epoch} predictions...", flush=True)

        barrier.wait()

        # Aggregate zone-wise R² from all folds' val preds
        global_r2, avg_zone_r2 = compute_global_and_avg_zone_r2(shared_dict)
        print(f"[Main] Epoch {epoch}: Aggregated global R² = {global_r2:.4f}. Aggregated zone-wise R² = {avg_zone_r2:.4f}.",flush=True)

        # Save if best so far
        if avg_zone_r2 > best_r2:
            print(f"[Main] ✅ New best! Saving checkpoints...",flush=True)
            best_r2 = avg_zone_r2
            save_flag.value=1
        else:
            save_flag.value=0

        barrier.wait()
        barrier.wait()

        # Clear shared_dict for next epoch
        shared_dict.clear()
        barrier.wait()

    for p in processes:
        p.join()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
