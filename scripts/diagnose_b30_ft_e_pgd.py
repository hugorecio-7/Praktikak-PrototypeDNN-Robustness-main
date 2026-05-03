"""
Temporary diagnostic script for the B30-FT-E PGD-Linf anomaly.

It checks the evaluation invariants that can explain a low-epsilon accuracy dip:
epsilon grid/value, input scale, Linf budget, alpha/steps/random-start settings,
eval mode, pixel clipping, cached-result reuse, and checkpoint provenance.

Example:
    python scripts/diagnose_b30_ft_e_pgd.py --max-batches 2
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import random
import sys
from pathlib import Path
from functools import partial

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adversarial_attacks import PGDLInf_attack  # noqa: E402
from data_loader import get_test_loader  # noqa: E402
from loss_functions import CELoss  # noqa: E402
from run_metrics import attack_params, load_models, paths  # noqa: E402


def ok(label: str, detail: str = "") -> None:
    print(f"[PASS] {label}" + (f" :: {detail}" if detail else ""))


def warn(label: str, detail: str = "") -> None:
    print(f"[WARN] {label}" + (f" :: {detail}" if detail else ""))


def fail(label: str, detail: str = "") -> None:
    print(f"[FAIL] {label}" + (f" :: {detail}" if detail else ""))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_eps_grid(max_eps: float, step: float) -> np.ndarray:
    return np.arange(0.0, max_eps + 1e-12, step, dtype=np.float64)


def assert_close(label: str, value: float, expected: float, tol: float = 1e-12) -> None:
    if math.isclose(float(value), float(expected), rel_tol=0.0, abs_tol=tol):
        ok(label, f"value={value:.17g}")
    else:
        fail(label, f"value={value:.17g}, expected={expected:.17g}, tol={tol:g}")


def check_source_contracts() -> None:
    pgd_src = inspect.getsource(PGDLInf_attack)
    eval_src = (ROOT / "model_testing_metrics.py").read_text(encoding="utf-8", errors="replace")
    run_src = (ROOT / "run_metrics.py").read_text(encoding="utf-8", errors="replace")

    if "/ 255" in pgd_src or "255" in pgd_src:
        fail("No silent 255 scaling in PGDLInf_attack source", "Found literal 255 in attack source")
    else:
        ok("No silent 255 scaling in PGDLInf_attack source")

    if "torch.clamp" in pgd_src and "min=-eps" in pgd_src and "max=eps" in pgd_src:
        ok("PGD projection is Linf", "delta is clamped elementwise to [-eps, eps]")
    else:
        fail("PGD projection is Linf", "Could not find elementwise delta clamp to [-eps, eps]")

    if "min=0" in pgd_src and "max=1" in pgd_src:
        ok("PGD clips pixels to [0,1]", "attack source clamps adversarial image to valid pixel range")
    else:
        fail("PGD clips pixels to [0,1]", "Could not find clamp to [0,1] in attack source")

    if "uniform_(" in pgd_src and "-eps" in pgd_src and "eps" in pgd_src:
        ok("Random start is inside Linf ball", "uniform_(-eps, eps)")
    else:
        warn("Random start source pattern not recognized")

    if "np.arange(0.0, max_eps + 1e-12, step" in eval_src:
        ok("Evaluation epsilon grid matches expected np.arange logic")
    else:
        warn("Evaluation epsilon grid source changed", "Review model_testing_metrics.py grid construction")

    cache_read_tokens = ["np.load", "torch.load"]
    reads_cache_like = [token for token in cache_read_tokens if token in eval_src]
    if reads_cache_like:
        warn("Evaluation cache read scan", f"Found possible read tokens: {reads_cache_like}")
    else:
        ok("No cached adversarial-example/result reads in model_testing_metrics.py")

    if '"PGDLInf_attack":                         {"iters": 80, "alpha": 0.01, "random_start": True}' in run_src:
        ok("run_metrics PGD registry is the expected one", "iters=80, alpha=0.01, random_start=True")
    else:
        warn("run_metrics PGD registry changed", str(attack_params.get("PGDLInf_attack")))


def check_provenance(model_name: str, expected_summary_variant: str) -> None:
    ckpt_path = ROOT / paths[model_name]
    if ckpt_path.exists():
        ok("Expected checkpoint exists", str(ckpt_path))
        print(f"       sha256={sha256_file(ckpt_path)[:16]}... size={ckpt_path.stat().st_size} bytes")
    else:
        fail("Expected checkpoint exists", str(ckpt_path))
        return

    summary_path = ROOT / "tfg_models" / "B30" / model_name / "seed=1" / "summary.json"
    config_path = ROOT / "tfg_models" / "B30" / model_name / "seed=1" / "config.json"
    trainable_path = ROOT / "tfg_models" / "B30" / model_name / "seed=1" / "trainable_parameters.txt"
    summary = load_json(summary_path)
    config = load_json(config_path)

    if summary and summary.get("variant") == expected_summary_variant:
        ok("Summary variant matches model", summary.get("variant"))
    else:
        fail("Summary variant matches model", f"summary={summary_path}")

    if summary:
        best_basename = Path(summary.get("best_checkpoint_path", "")).name
        if best_basename == ckpt_path.name:
            ok("Summary best checkpoint basename matches loaded checkpoint", best_basename)
        else:
            fail("Summary best checkpoint basename matches loaded checkpoint", f"{best_basename} != {ckpt_path.name}")
        print(f"       best_epoch={summary.get('best_epoch')} best_val_adv_acc={summary.get('best_val_adv_acc')}")

    if config:
        ok(
            "Training attack config loaded",
            f"adv_eps={config.get('adv_eps')}, adv_alpha={config.get('adv_alpha')}, "
            f"adv_steps={config.get('adv_steps')}, random_start={config.get('adv_random_start')}",
        )

    if trainable_path.exists():
        names = [line.strip() for line in trainable_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        outside_ae = [n for n in names if not (n.startswith("encoder.") or n.startswith("decoder."))]
        if not outside_ae:
            ok("Only autoencoder params were trainable", f"{len(names)} params: encoder/decoder only")
        else:
            fail("Only autoencoder params were trainable", f"Outside AE: {outside_ae}")
    else:
        warn("Trainable-parameter file missing", str(trainable_path))


def check_saved_run(model_name: str, run_id: str, target_eps: float) -> None:
    metrics_path = (
        ROOT
        / "resultsTFG"
        / "runs_v1"
        / "mnist"
        / "Linf"
        / "PGDLInf_attack"
        / model_name
        / "seed=1"
        / f"run_id={run_id}"
        / "metrics.json"
    )
    metrics = load_json(metrics_path)
    if not metrics:
        warn("Saved metrics for reference run not found", str(metrics_path))
        return

    eps = np.asarray(metrics.get("eps", []), dtype=np.float64)
    acc = np.asarray(metrics.get("acc", []), dtype=np.float64)
    if eps.size and np.any(np.isclose(eps, target_eps, atol=1e-12, rtol=0.0)):
        idx = int(np.where(np.isclose(eps, target_eps, atol=1e-12, rtol=0.0))[0][0])
        ok("Saved metrics contain target epsilon", f"eps[{idx}]={eps[idx]:.17g}, acc={acc[idx]:.4f}")
        left = max(0, idx - 1)
        right = min(len(eps), idx + 4)
        print("       local saved curve:")
        for j in range(left, right):
            print(f"         eps={eps[j]:.6f} acc={acc[j]:.4f}")
    else:
        fail("Saved metrics contain target epsilon", f"target={target_eps}")

    params = metrics.get("attack_params", {})
    if params == attack_params.get("PGDLInf_attack"):
        ok("Saved run attack params match current registry", str(params))
    else:
        warn("Saved run attack params differ from current registry", f"saved={params}, current={attack_params.get('PGDLInf_attack')}")

    violations = metrics.get("monotonicity_violations", [])
    if violations:
        warn("Saved curve is non-monotonic", json.dumps(violations[:5]))
    else:
        ok("Saved curve has no >1pp monotonicity violations")

    correct_npz = metrics_path.with_name("per_example_correct.npz")
    adv_npz = metrics_path.with_name("adversarial_examples.npz")
    if correct_npz.exists():
        ok("Saved per-example correctness exists", str(correct_npz))
        data = np.load(correct_npz)
        correct = data["correct"].astype(bool)
        saved_eps = data["eps"].astype(np.float64)
        target_matches = np.where(np.isclose(saved_eps, target_eps, atol=1e-7, rtol=0.0))[0]
        if len(target_matches) == 0:
            warn("Could not locate target epsilon in per_example_correct.npz", f"target={target_eps}")
            target_idx = 1
        else:
            target_idx = int(target_matches[0])
        next_idx = target_idx + 1 if target_idx + 1 < correct.shape[1] else target_idx
        if next_idx > target_idx:
            recovered = (~correct[:, target_idx]) & correct[:, next_idx]
            warn(
                "Samples recover after target epsilon in saved run",
                f"{int(recovered.sum())}/{correct.shape[0]} fail at eps={saved_eps[target_idx]:.6f} "
                f"but pass at eps={saved_eps[next_idx]:.6f}",
            )
        worst_case_correct = np.logical_and.accumulate(correct, axis=1)
        worst_acc = worst_case_correct.mean(axis=0)
        print("       cumulative worst-case accuracy:")
        for j in range(max(0, target_idx - 1), min(len(saved_eps), target_idx + 4)):
            print(f"         eps={saved_eps[j]:.6f} worst_acc={worst_acc[j]:.4f}")
    if adv_npz.exists():
        warn("Adversarial-example cache file exists near run", str(adv_npz))
    else:
        ok("No adversarial-example cache file next to saved run")


def check_dynamic_attack(args: argparse.Namespace) -> None:
    params = attack_params["PGDLInf_attack"].copy()
    eval_iters = int(params["iters"])
    eval_alpha = float(params["alpha"])
    random_start = bool(params["random_start"])

    assert_close("Target epsilon is exactly the first non-zero grid value", args.target_eps, args.step)
    eps_grid = build_eps_grid(args.max_eps, args.step)
    assert_close("Generated epsilon grid column 1", eps_grid[1], args.target_eps)
    if math.isclose(args.target_eps, 25.0 / 255.0, rel_tol=0.0, abs_tol=1e-12):
        fail("Target epsilon is not 25/255", f"target={args.target_eps}")
    else:
        ok("Target epsilon is not 25/255", f"25/255={25.0 / 255.0:.17g}")

    if eval_alpha <= args.target_eps and eval_iters * eval_alpha >= args.target_eps:
        ok("Evaluation alpha is coherent with target epsilon", f"alpha={eval_alpha}, eps={args.target_eps}, iters={eval_iters}")
    else:
        warn("Evaluation alpha may be incoherent", f"alpha={eval_alpha}, eps={args.target_eps}, iters={eval_iters}")

    ok("Evaluation PGD step count", f"iters={eval_iters}")
    ok("Evaluation random_start setting", f"random_start={random_start}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    models, _ = load_models([args.model])
    model = models[0].to(device)

    if not model.training and all(not m.training for m in model.modules()):
        ok("Model is in eval() mode after loading")
    else:
        fail("Model is in eval() mode after loading", "At least one module has training=True")

    loader = get_test_loader("data", batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    total = 0
    clean_correct = 0
    adv_correct = 0
    max_linf_seen = 0.0
    max_l2_seen = 0.0
    min_x_seen = float("inf")
    max_x_seen = -float("inf")
    min_adv_seen = float("inf")
    max_adv_seen = -float("inf")
    saturated_count = 0
    outside_budget_count = 0

    first_batch = None
    first_labels = None

    for batch_idx, (x, y) in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        first_batch = x if first_batch is None else first_batch
        first_labels = y if first_labels is None else first_labels

        min_x_seen = min(min_x_seen, float(x.min().item()))
        max_x_seen = max(max_x_seen, float(x.max().item()))

        loss_f = partial(CELoss, model=model, batch_y=y)
        x_adv = PGDLInf_attack(
            x,
            loss_f=loss_f,
            iters=eval_iters,
            eps=float(args.target_eps),
            alpha=eval_alpha,
            random_start=random_start,
        )

        delta = (x_adv - x).detach()
        linf_per_sample = delta.flatten(1).abs().max(dim=1).values
        l2_per_sample = delta.flatten(1).norm(p=2, dim=1)
        max_linf_seen = max(max_linf_seen, float(linf_per_sample.max().item()))
        max_l2_seen = max(max_l2_seen, float(l2_per_sample.max().item()))
        outside_budget_count += int((linf_per_sample > args.target_eps + args.tol).sum().item())
        saturated_count += int((linf_per_sample >= args.target_eps - 1e-6).sum().item())
        min_adv_seen = min(min_adv_seen, float(x_adv.min().item()))
        max_adv_seen = max(max_adv_seen, float(x_adv.max().item()))

        with torch.no_grad():
            clean_pred = model(x).argmax(dim=1)
            adv_pred = model(x_adv).argmax(dim=1)
        clean_correct += int((clean_pred == y).sum().item())
        adv_correct += int((adv_pred == y).sum().item())
        total += int(y.numel())

    if total == 0:
        fail("Dynamic attack check", "No batches were evaluated")
        return

    if 0.0 <= min_x_seen <= max_x_seen <= 1.0:
        ok("Input data scale is [0,1]", f"min={min_x_seen:.6f}, max={max_x_seen:.6f}")
    else:
        fail("Input data scale is [0,1]", f"min={min_x_seen:.6f}, max={max_x_seen:.6f}")

    if 0.0 <= min_adv_seen <= max_adv_seen <= 1.0:
        ok("Adversarial pixels are clipped to [0,1]", f"min={min_adv_seen:.6f}, max={max_adv_seen:.6f}")
    else:
        fail("Adversarial pixels are clipped to [0,1]", f"min={min_adv_seen:.6f}, max={max_adv_seen:.6f}")

    if outside_budget_count == 0:
        ok("Generated examples respect target Linf budget", f"max_linf={max_linf_seen:.12f}, eps={args.target_eps}")
    else:
        fail("Generated examples respect target Linf budget", f"{outside_budget_count} samples outside eps; max_linf={max_linf_seen}")

    ok("Actual target-epsilon attack generated fresh examples", f"{saturated_count}/{total} samples reached eps boundary")
    print(f"       clean_acc_sample={clean_correct / total:.4f} adv_acc_eps_{args.target_eps:g}_sample={adv_correct / total:.4f}")
    print(f"       max_l2_seen={max_l2_seen:.6f} (can exceed eps under Linf; this is expected)")

    if first_batch is not None and first_labels is not None:
        loss_f = partial(CELoss, model=model, batch_y=first_labels)
        torch.manual_seed(args.seed + 123)
        adv_a = PGDLInf_attack(first_batch, loss_f, eval_iters, float(args.target_eps), eval_alpha, random_start)
        torch.manual_seed(args.seed + 123)
        adv_b = PGDLInf_attack(first_batch, loss_f, eval_iters, float(args.target_eps), eval_alpha, random_start)
        torch.manual_seed(args.seed + 124)
        adv_c = PGDLInf_attack(first_batch, loss_f, eval_iters, float(args.target_eps), eval_alpha, random_start)

        same_seed_equal = torch.equal(adv_a.cpu(), adv_b.cpu())
        diff_seed_equal = torch.equal(adv_a.cpu(), adv_c.cpu())
        if same_seed_equal:
            ok("Random start is reproducible when seed is reset")
        else:
            warn("Random start reproducibility check failed", "Same seed produced different tensors")
        if random_start and not diff_seed_equal:
            ok("Random start changes examples when seed changes")
        elif random_start:
            warn("Random start seed-change check inconclusive", "Different seeds produced identical final tensors")

    if not model.training and all(not m.training for m in model.modules()):
        ok("Model remained in eval() mode after attack")
    else:
        fail("Model remained in eval() mode after attack")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="B30-FT-E")
    parser.add_argument("--run-id", default="ablation_B30_v1")
    parser.add_argument("--max-eps", type=float, default=0.8)
    parser.add_argument("--step", type=float, default=0.025)
    parser.add_argument("--target-eps", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--max-batches", type=int, default=2, help="Use -1 for the full test set.")
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.max_batches is not None and args.max_batches < 0:
        args.max_batches = None
    return args


def main() -> None:
    args = parse_args()
    print("=== B30-FT-E PGD-Linf diagnostic ===")
    print(f"root={ROOT}")
    print(f"model={args.model} target_eps={args.target_eps} max_eps={args.max_eps} step={args.step}")
    print()

    check_source_contracts()
    print()
    check_provenance(args.model, expected_summary_variant=args.model)
    print()
    check_saved_run(args.model, args.run_id, args.target_eps)
    print()
    check_dynamic_attack(args)


if __name__ == "__main__":
    main()
