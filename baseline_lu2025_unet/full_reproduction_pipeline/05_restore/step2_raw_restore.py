#!/usr/bin/env python3
"""
Evaluate UNet on full dataset using Lu's restore_raw_output_data (raw-domain eval).
Matches Lu's compute_all_metrics: restore patches → raw domain, exclude building pixels.
"""
import argparse, json, logging, numpy as np, torch, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "references"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from data_formatter_fixed import DataFormatterFixed as DataFormatter

ROOT = Path(__file__).resolve().parent.parent

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-c", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/full_masked_640")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results/full_masked_640/seed_1")
    p.add_argument("--use-final", action="store_true")
    return p.parse_args()

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()

    # Load test data
    data = torch.load(args.data_dir / "test.pt", map_location="cpu", weights_only=False)
    X, Y = data["X"], data["Y"]
    fmt_shape_val = data.get("fmt_shape", 640)

    # Load model
    sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
    from unet_lu import UNetLu
    model = UNetLu(n_c=args.n_c).to(device)
    model_file = "model_final.pt" if args.use_final else "model_best.pt"
    model.load_state_dict(torch.load(args.results_dir / model_file, map_location=device, weights_only=True))
    model.eval()
    logging.info(f"Loaded {model_file}")

    # Inference
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(X), batch_size=args.batch_size, shuffle=False)
    all_pred = []
    with torch.no_grad():
        for (x,) in loader:
            all_pred.append(model(x.to(device)).cpu())
    pred_all = torch.cat(all_pred, dim=0)
    pred_np = np.nan_to_num(pred_all.numpy()[:, 0, :, :], nan=0.0).astype(np.float32)

    # Rebuild DataFormatter for test cases
    import pandas as pd, xarray as xr
    split_dir = ROOT / "scripts"
    test_df = pd.read_csv(split_dir / "lu_full_test_seed1.csv")
    test_cases_list = sorted(test_df['case_name'].tolist())
    RAW_DIR = Path('<PROJECT_HPC_ROOT>/data/urbantales/raw')

    raw_data_list, wind_angle_list, loaded_names = [], [], []
    for cn in test_cases_list:
        case_dir = RAW_DIR / cn
        if not case_dir.is_dir():
            continue
        try:
            topo = np.flipud(np.loadtxt(case_dir / f"{cn}_topo", dtype=np.float32))
            with xr.open_dataset(case_dir / f"{cn}_ped.nc") as ds:
                Uped = ds["Uped"].values.astype(np.float32)
            combined = Uped.copy()
            combined[topo > 0] = -topo[topo > 0]
            m = re.search(r"_d(\d+)$", cn)
            raw_data_list.append(combined)
            wind_angle_list.append(int(m.group(1)))
            loaded_names.append(cn)
        except Exception as e:
            logging.warning(f"Error: {cn}: {e}")

    logging.info(f"Loaded {len(raw_data_list)} test cases")

    formatter = DataFormatter(
        raw_data=raw_data_list,
        wind_angles=wind_angle_list,
        formatted_shape=fmt_shape_val
    )

    n_fmt = len(formatter.get_formatted_output_data())
    logging.info(f"Formatter: {n_fmt} patches, Ours: {pred_np.shape[0]} patches")

    if n_fmt != pred_np.shape[0]:
        logging.error(f"Patch count mismatch! Aborting.")
        return

    # Restore predictions to raw domain
    pred_formatted = pred_np[:, np.newaxis, :, :]
    restored = formatter.restore_raw_output_data(pred_formatted)

    # Compute metrics: exclude building pixels (raw_data >= 0 means non-building for combined matrix)
    # Lu's compute_all_metrics: num_idx = self.raw_data[tidx] >= 0
    # raw_data is the COMBINED matrix: negative=topo, positive=wind speed
    # So raw_data >= 0 means wind speed pixels (including zero wind speed outside buildings)
    # Wait: in combined, topo > 0 → combined = -topo (negative), wind speed ≥ 0
    # So combined >= 0 means everything that's NOT a building. This matches Lu's intent.

    by_case = []
    all_gt, all_pr = [], []

    for i, cn in enumerate(loaded_names):
        truth = raw_data_list[i]
        pred = restored[i]
        # Handle rot90 shape mismatch (should not happen with fixed formatter)
        if truth.shape != pred.shape:
            logging.warning(f"{cn}: shape mismatch truth={truth.shape} pred={pred.shape}, skipping")
            continue
        # Exclude building: combined >= 0 (non-building pixels)
        valid = truth >= 0
        # Also exclude any NaN
        valid = valid & np.isfinite(truth) & np.isfinite(pred)
        
        t = truth[valid]
        p = pred[valid]
        
        if len(t) == 0:
            continue
        
        all_gt.append(t)
        all_pr.append(p)
        
        mae = float(np.mean(np.abs(t - p)))
        ss_res = float(np.sum((t - p)**2))
        ss_tot = float(np.sum((t - t.mean())**2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        nmae = mae / np.mean(t) * 100 if np.mean(t) > 0 else 0.0
        
        by_case.append(dict(
            case_name=cn, mae=mae, r2=float(r2), nmae=float(nmae),
            u_mean=float(np.mean(t)), n_valid=int(len(t)),
            category='idealized' if cn.startswith(('VA','VS','UA','US')) else 'realistic'
        ))

    # Global pooled
    gt_all = np.concatenate(all_gt)
    pr_all = np.concatenate(all_pr)
    g_mae = float(np.mean(np.abs(gt_all - pr_all)))
    g_r2 = 1 - float(np.sum((gt_all - pr_all)**2)) / float(np.sum((gt_all - gt_all.mean())**2))
    g_nmae = g_mae / float(np.mean(gt_all)) * 100

    ideal = [c for c in by_case if c['category'] == 'idealized']
    real = [c for c in by_case if c['category'] == 'realistic']

    result = dict(
        global_metrics=dict(mae=g_mae, r2=g_r2, nmae=g_nmae,
                           n_total=len(by_case), n_ideal=len(ideal), n_real=len(real),
                           model_file=model_file, eval_method='raw_domain_restore'),
        idealized=dict(r2_median=float(np.median([c['r2'] for c in ideal])) if ideal else 0,
                       nmae_median=float(np.median([c['nmae'] for c in ideal])) if ideal else 0),
        realistic=dict(r2_median=float(np.median([c['r2'] for c in real])) if real else 0,
                       nmae_median=float(np.median([c['nmae'] for c in real])) if real else 0),
        by_case=by_case
    )

    suffix = "raw_restore_final" if args.use_final else "raw_restore_best"
    out_file = args.results_dir / f"metrics_{suffix}.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)

    logging.info(f"=== Raw Domain Eval ({model_file}) ===")
    logging.info(f"Global: MAE={g_mae:.4f}, R2={g_r2:.4f}, NMAE={g_nmae:.1f}%")
    if ideal:
        logging.info(f"Idealized ({len(ideal)}): R2_median={result['idealized']['r2_median']:.4f}, NMAE_median={result['idealized']['nmae_median']:.1f}%")
    if real:
        logging.info(f"Realistic ({len(real)}): R2_median={result['realistic']['r2_median']:.4f}, NMAE_median={result['realistic']['nmae_median']:.1f}%")
    logging.info(f"Saved to {out_file}")

if __name__ == "__main__":
    main()
