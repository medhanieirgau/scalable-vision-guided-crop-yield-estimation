import numpy as np
from collections import defaultdict

def r2_score(targets, preds):
    return np.corrcoef(targets, preds)[0, 1] ** 2

def compute_global_and_avg_zone_r2(shared_dict):
    all_preds = []
    all_targets = []
    all_zones = []

    for fold_data in shared_dict.values():
        all_preds.append(fold_data["preds"])
        all_targets.append(fold_data["targets"])
        all_zones.append(fold_data["zones"])

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_zones = np.concatenate(all_zones)

    # Global R²
    global_r2 = r2_score(all_targets, all_preds)

    # Zone-wise R²s
    zone_to_preds = defaultdict(list)
    zone_to_targets = defaultdict(list)

    for pred, target, zone in zip(all_preds, all_targets, all_zones):
        zone_to_preds[zone].append(pred)
        zone_to_targets[zone].append(target)

    zone_r2_list = []
    for zone in zone_to_preds:
        preds = np.array(zone_to_preds[zone])
        targets = np.array(zone_to_targets[zone])
        if np.std(preds) == 0 or np.std(targets) == 0:
            continue  # skip this zone
        zone_r2_list.append(r2_score(targets, preds))

    avg_zone_r2 = np.mean(zone_r2_list)

    return global_r2, avg_zone_r2
