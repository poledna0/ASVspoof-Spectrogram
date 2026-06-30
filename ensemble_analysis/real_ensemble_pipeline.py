#!/usr/bin/env python3
import argparse
import csv
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class EnsembleMetrics:
    ensemble_id: str
    models: List[str]
    weights: List[float]
    dev_eer: float
    dev_eer_threshold: float
    dev_min_tdcf: float
    eval_eer: float
    eval_eer_threshold: float
    eval_min_tdcf: float


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("RealEnsemblePipeline")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_protocol(protocol_file: Path) -> List[Dict[str, str]]:
    rows = []
    with protocol_file.open("r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 5:
                continue
            rows.append(
                {
                    "speaker": p[0],
                    "utt_id": p[1],
                    "environment": p[2],
                    "attack": p[3],
                    "label": p[4],
                }
            )
    return rows


def load_scores(score_file: Path) -> Dict[str, float]:
    data: Dict[str, float] = {}
    with score_file.open("r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 2:
                continue
            data[p[0]] = float(p[1])
    return data


def find_checkpoints(checkpoints_dir: Path) -> Dict[str, str]:
    ckpts: Dict[str, str] = {}
    for p in checkpoints_dir.rglob("*_best.pth"):
        name = p.name.replace("_best.pth", "")
        ckpts[name] = str(p)
    return ckpts


def model_to_root_score_paths(root: Path, model_name: str) -> Tuple[Path, Path]:
    arch, feature = model_name.split("_", 1)
    score_dir = root / "scores" / arch
    return (
        score_dir / f"{feature}_DEV_scores.txt",
        score_dir / f"{feature}_EVAL_scores.txt",
    )


def model_to_checkpoint_score_paths(checkpoint_file: Path, model_name: str) -> Tuple[Path, Path]:
    run_dir = checkpoint_file.parents[1]
    score_dir = run_dir / "scores"
    return (
        score_dir / f"{model_name}_DEV_scores.txt",
        score_dir / f"{model_name}_EVAL_scores.txt",
    )


def normalize_to_bonafide(score_vec: np.ndarray, spoof_higher: bool) -> np.ndarray:
    bounded = np.all((score_vec >= 0.0) & (score_vec <= 1.0))
    if bounded:
        return 1.0 - score_vec if spoof_higher else score_vec
    return sigmoid(-score_vec) if spoof_higher else sigmoid(score_vec)


def compute_metrics(
    em,
    bona_scores: np.ndarray,
    spoof_scores: np.ndarray,
    asv_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray],
    cost_model: Dict[str, float],
) -> Tuple[float, float, float]:
    eer, eer_threshold = em.compute_eer(bona_scores, spoof_scores)
    tar_asv, non_asv, spoof_asv = asv_tuple
    Pfa_asv, Pmiss_asv, Pmiss_spoof_asv = em.obtain_asv_error_rates(
        tar_asv, non_asv, spoof_asv, em.compute_eer(tar_asv, non_asv)[1]
    )
    tdcf_curve, _ = em.compute_tDCF(
        bona_scores,
        spoof_scores,
        Pfa_asv,
        Pmiss_asv,
        Pmiss_spoof_asv,
        cost_model,
        print_cost=False,
    )
    return float(eer), float(eer_threshold), float(np.min(tdcf_curve))


def weighted_fusion(model_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return model_matrix @ weights


def write_cm_score_file(
    output_file: Path,
    protocol_rows: List[Dict[str, str]],
    score_by_utt: Dict[str, float],
) -> None:
    with output_file.open("w", encoding="utf-8") as f:
        for row in protocol_rows:
            utt = row["utt_id"]
            if utt not in score_by_utt:
                continue
            f.write(f"{utt} {row['attack']} {row['label']} {score_by_utt[utt]:.10f}\n")


def run_official_tdcf(root: Path, cm_file: Path, asv_file: Path, out_txt: Path) -> None:
    cmd = [
        str(root / "venv" / "bin" / "python"),
        str(root / "tdcf" / "evaluate_tDCF_asvspoof19.py"),
        str(cm_file),
        str(asv_file),
        "--no-plot",
    ]
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    out_txt.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline real de 10 ensembles com métricas oficiais")
    parser.add_argument("--workspace-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-dir", type=str, default="final_ensemble_results")
    parser.add_argument("--ensemble-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--weight-samples", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.workspace_root).resolve()
    output_dir = root / args.output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir / "logs" / "pipeline.log")
    logger.info("Iniciando pipeline real de ensembles")
    logger.info("Workspace: %s", root)

    sys.path.insert(0, str(root / "tdcf"))
    import eval_metrics as em  # pylint: disable=import-error

    random.seed(args.seed)
    np.random.seed(args.seed)

    protocol_dev = load_protocol(root / "PA_cm_protocols" / "ASVspoof2019.PA.cm.dev.trl.txt")
    protocol_eval = load_protocol(root / "PA_cm_protocols" / "ASVspoof2019.PA.cm.eval.trl.txt")
    dev_by_utt = {r["utt_id"]: r for r in protocol_dev}
    eval_by_utt = {r["utt_id"]: r for r in protocol_eval}

    ckpts = find_checkpoints(root / "checkpoints")
    logger.info("Checkpoints encontrados: %d", len(ckpts))

    valid_models = []
    model_dev_scores: Dict[str, Dict[str, float]] = {}
    model_eval_scores: Dict[str, Dict[str, float]] = {}
    spoof_higher_map: Dict[str, bool] = {}

    score_source = {}
    for model_name in sorted(ckpts.keys()):
        ckpt_file = Path(ckpts[model_name])
        ck_dev_path, ck_eval_path = model_to_checkpoint_score_paths(ckpt_file, model_name)
        root_dev_path, root_eval_path = model_to_root_score_paths(root, model_name)

        dev_path = ck_dev_path if ck_dev_path.exists() else root_dev_path
        eval_path = ck_eval_path if ck_eval_path.exists() else root_eval_path

        if not dev_path.exists() or not eval_path.exists():
            logger.warning("Modelo %s ignorado: score DEV/EVAL ausente", model_name)
            continue

        dev_scores = load_scores(dev_path)
        eval_scores = load_scores(eval_path)

        aligned_dev = [(u, s) for u, s in dev_scores.items() if u in dev_by_utt]
        if not aligned_dev:
            logger.warning("Modelo %s ignorado: sem alinhamento com protocolo DEV", model_name)
            continue

        bona = [s for u, s in aligned_dev if dev_by_utt[u]["label"] == "bonafide"]
        spoof = [s for u, s in aligned_dev if dev_by_utt[u]["label"] == "spoof"]
        spoof_higher = float(np.mean(bona)) < float(np.mean(spoof))

        valid_models.append(model_name)
        model_dev_scores[model_name] = dev_scores
        model_eval_scores[model_name] = eval_scores
        spoof_higher_map[model_name] = spoof_higher
        score_source[model_name] = {
            "dev": str(dev_path),
            "eval": str(eval_path),
        }

    valid_models = sorted(valid_models)
    logger.info("Modelos válidos para ensemble: %d", len(valid_models))
    if len(valid_models) < args.ensemble_size:
        raise RuntimeError("Modelos válidos insuficientes para gerar ensembles")

    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with (metadata_dir / "checkpoints_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(ckpts, f, indent=2)
    with (metadata_dir / "valid_models.json").open("w", encoding="utf-8") as f:
        json.dump(valid_models, f, indent=2)
    with (metadata_dir / "score_sources.json").open("w", encoding="utf-8") as f:
        json.dump(score_source, f, indent=2)

    asv_file = root / "PA_scores" / "ASVspoof2019.PA.asv.eval.gi.trl.scores.txt"
    asv_data = np.genfromtxt(asv_file, dtype=str)
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(float)
    asv_tuple = (
        asv_scores[asv_keys == "target"],
        asv_scores[asv_keys == "nontarget"],
        asv_scores[asv_keys == "spoof"],
    )

    cost_model = {
        "Pspoof": 0.05,
        "Ptar": (1 - 0.05) * 0.99,
        "Pnon": (1 - 0.05) * 0.01,
        "Cmiss_asv": 1,
        "Cfa_asv": 10,
        "Cmiss_cm": 1,
        "Cfa_cm": 10,
    }

    combos = list(combinations(valid_models, args.ensemble_size))
    logger.info("Combinações candidatas: %d", len(combos))

    dev_common_utts = sorted(set(dev_by_utt.keys()).intersection(*[set(model_dev_scores[m].keys()) for m in valid_models]))
    eval_common_utts = sorted(set(eval_by_utt.keys()).intersection(*[set(model_eval_scores[m].keys()) for m in valid_models]))
    logger.info("Amostras comuns DEV: %d | EVAL: %d", len(dev_common_utts), len(eval_common_utts))

    candidate_results = []
    for combo in combos:
        combo_list = list(combo)
        dev_matrix = []
        for model_name in combo_list:
            raw = np.array([model_dev_scores[model_name][u] for u in dev_common_utts], dtype=np.float64)
            dev_matrix.append(normalize_to_bonafide(raw, spoof_higher_map[model_name]))
        dev_matrix_np = np.stack(dev_matrix, axis=1)

        labels_dev = np.array([dev_by_utt[u]["label"] for u in dev_common_utts])
        bona_mask = labels_dev == "bonafide"
        spoof_mask = labels_dev == "spoof"

        best = None
        for i in range(args.weight_samples):
            if i == 0:
                weights = np.full((args.ensemble_size,), 1.0 / args.ensemble_size)
            else:
                weights = np.random.dirichlet(np.ones(args.ensemble_size))

            fused = weighted_fusion(dev_matrix_np, weights)
            eer, eer_thr, min_tdcf = compute_metrics(
                em,
                fused[bona_mask],
                fused[spoof_mask],
                asv_tuple,
                cost_model,
            )

            row = {
                "models": combo_list,
                "weights": [float(w) for w in weights.tolist()],
                "dev_eer": eer,
                "dev_eer_threshold": eer_thr,
                "dev_min_tdcf": min_tdcf,
            }
            if best is None or row["dev_min_tdcf"] < best["dev_min_tdcf"] or (
                math.isclose(row["dev_min_tdcf"], best["dev_min_tdcf"], rel_tol=1e-9)
                and row["dev_eer"] < best["dev_eer"]
            ):
                best = row

        candidate_results.append(best)

    candidate_results.sort(key=lambda x: (x["dev_min_tdcf"], x["dev_eer"]))
    selected = candidate_results[: args.top_k]

    top_dir = output_dir / "top10"
    top_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    all_results: List[EnsembleMetrics] = []

    labels_eval = np.array([eval_by_utt[u]["label"] for u in eval_common_utts])
    eval_bona_mask = labels_eval == "bonafide"
    eval_spoof_mask = labels_eval == "spoof"

    for idx, chosen in enumerate(selected, start=1):
        ens_id = f"ensemble_{idx:02d}"
        ens_dir = top_dir / ens_id
        ens_dir.mkdir(parents=True, exist_ok=True)

        models_list = chosen["models"]
        weights = np.array(chosen["weights"], dtype=np.float64)

        eval_matrix = []
        for model_name in models_list:
            raw = np.array([model_eval_scores[model_name][u] for u in eval_common_utts], dtype=np.float64)
            eval_matrix.append(normalize_to_bonafide(raw, spoof_higher_map[model_name]))
        eval_matrix_np = np.stack(eval_matrix, axis=1)
        eval_fused = weighted_fusion(eval_matrix_np, weights)

        eval_eer, eval_eer_thr, eval_min_tdcf = compute_metrics(
            em,
            eval_fused[eval_bona_mask],
            eval_fused[eval_spoof_mask],
            asv_tuple,
            cost_model,
        )

        score_by_utt = {u: float(s) for u, s in zip(eval_common_utts, eval_fused)}
        cm_file = ens_dir / "cm_eval_scores.txt"
        write_cm_score_file(cm_file, protocol_eval, score_by_utt)
        run_official_tdcf(root, cm_file, asv_file, ens_dir / "official_tdcf_stdout.txt")

        metrics = EnsembleMetrics(
            ensemble_id=ens_id,
            models=models_list,
            weights=[float(w) for w in weights.tolist()],
            dev_eer=float(chosen["dev_eer"]),
            dev_eer_threshold=float(chosen["dev_eer_threshold"]),
            dev_min_tdcf=float(chosen["dev_min_tdcf"]),
            eval_eer=float(eval_eer),
            eval_eer_threshold=float(eval_eer_thr),
            eval_min_tdcf=float(eval_min_tdcf),
        )
        all_results.append(metrics)

        with (ens_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(metrics), f, indent=2)

        with (ens_dir / "checkpoints_used.json").open("w", encoding="utf-8") as f:
            json.dump({m: ckpts[m] for m in models_list}, f, indent=2)

        summary_rows.append(
            {
                "ensemble_id": ens_id,
                "models": " + ".join(models_list),
                "weights": " + ".join([f"{w:.4f}" for w in weights.tolist()]),
                "dev_eer_percent": f"{chosen['dev_eer'] * 100:.4f}",
                "dev_min_tdcf": f"{chosen['dev_min_tdcf']:.6f}",
                "eval_eer_percent": f"{eval_eer * 100:.4f}",
                "eval_min_tdcf": f"{eval_min_tdcf:.6f}",
            }
        )

    summary_rows.sort(key=lambda x: (float(x["eval_min_tdcf"]), float(x["eval_eer_percent"])))
    with (top_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with (top_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)

    best_ens_id = summary_rows[0]["ensemble_id"]
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    src_cm = top_dir / best_ens_id / "cm_eval_scores.txt"
    dst_cm = final_dir / "CM_EVAL_TOP1.txt"
    shutil.copy2(src_cm, dst_cm)
    with (final_dir / "best_ensemble.json").open("w", encoding="utf-8") as f:
        for r in all_results:
            if r.ensemble_id == best_ens_id:
                json.dump(asdict(r), f, indent=2)
                break

    logger.info("Top 10 ensembles reais gerados em: %s", top_dir)
    logger.info("Score final salvo em: %s", dst_cm)


if __name__ == "__main__":
    main()