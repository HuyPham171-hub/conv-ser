import json
from pathlib import Path
from typing import Optional

def get_best_checkpoint(fold_dir: Path, metric_key: str = "eval_uar", greater_is_better: bool = True) -> Optional[Path]:
    """
    Identifies the best checkpoint directory within a Fold, replicating Hugging Face Trainer's internal logic.
    """
    if not fold_dir.exists() or not fold_dir.is_dir():
        raise FileNotFoundError(f"[ERROR] Fold directory does not exist: {fold_dir}")

    # Step 1: Find all checkpoint directories and sort them by step number
    checkpoints = [d for d in fold_dir.glob("checkpoint-*") if d.is_dir()]
    if not checkpoints:
        print(f"[ERROR] No checkpoints found in {fold_dir}")
        return None

    # Sort descending by step number to find the latest checkpoint
    checkpoints.sort(key=lambda x: int(x.name.split("-")[1]), reverse=True)
    latest_checkpoint = checkpoints[0]
    
    state_file = latest_checkpoint / "trainer_state.json"
    if not state_file.exists():
        print(f"[WARNING] trainer_state.json not found in {latest_checkpoint}. Returning the latest checkpoint as fallback.")
        return latest_checkpoint

    with open(state_file, "r", encoding="utf-8") as f:
        state_data = json.load(f)

    # Step 2: Primary Hugging Face Logic - Read 'best_model_checkpoint'
    best_model_path_str = state_data.get("best_model_checkpoint")
    if best_model_path_str:
        best_cp_name = Path(best_model_path_str).name
        actual_best_dir = fold_dir / best_cp_name
        
        if actual_best_dir.exists():
            return actual_best_dir

    # Step 3: Fallback Logic - Parse log_history from the LATEST state file only
    print(f"[WARNING] 'best_model_checkpoint' flag not found. Reconstructing from log_history...")
    
    best_step = None
    best_metric_val = -float("inf") if greater_is_better else float("inf")
    best_eval_loss = float("inf") # Tie-breaker

    for log in state_data.get("log_history", []):
        if metric_key in log and "step" in log:
            current_metric = log[metric_key]
            current_loss = log.get("eval_loss", float("inf"))
            current_step = log["step"]

            is_better = False
            if greater_is_better:
                if current_metric > best_metric_val:
                    is_better = True
                elif current_metric == best_metric_val and current_loss < best_eval_loss:
                    is_better = True
            else:
                if current_metric < best_metric_val:
                    is_better = True
                elif current_metric == best_metric_val and current_loss < best_eval_loss:
                    is_better = True

            if is_better:
                best_metric_val = current_metric
                best_eval_loss = current_loss
                best_step = current_step

    if best_step:
        target_cp_dir = fold_dir / f"checkpoint-{best_step}"
        if target_cp_dir.exists():
            return target_cp_dir

    # Step 4: Final Fallback
    print(f"[WARNING] Could not determine best checkpoint from logs. Returning the latest checkpoint.")
    return latest_checkpoint

# ==========================================
# EXECUTION EXAMPLE
# ==========================================
if __name__ == "__main__":
    STAGE1_DIR = Path(r"d:\Resfes\Project\Ser\checkpoints\wav2vec2_stage1")
    
    print(f"[INFO] Scanning folds at: {STAGE1_DIR}\n")
    
    for fold_idx in range(1, 6):
        fold_path = STAGE1_DIR / f"fold_{fold_idx}"
        
        if fold_path.exists():
            best_cp = get_best_checkpoint(fold_path, metric_key="eval_uar", greater_is_better=True)
            print(f"✅ Fold {fold_idx}: Best Checkpoint -> {best_cp.name}")
        else:
            print(f"❌ Fold {fold_idx}: Directory not found {fold_path}")