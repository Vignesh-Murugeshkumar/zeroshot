from typing import Tuple, Dict, List
import os
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
from tqdm import tqdm

from .inference import ZeroShotPipeline


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def evaluate_folder(folder: str, pipeline: ZeroShotPipeline, out_path: str = "results_summary.json") -> Dict:
    """Evaluate a folder structured like TrashNet: subfolders per class, images inside.

    Returns a dict of metrics and writes a JSON summary.
    """
    y_true = []
    y_pred = []
    labels = sorted(os.listdir(folder))
    label_to_idx = {l: i for i, l in enumerate(labels)}

    for label in labels:
        p = os.path.join(folder, label)
        if not os.path.isdir(p):
            continue
        for fn in tqdm(os.listdir(p), desc=f"Scanning {label}"):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = load_image(os.path.join(p, fn))
            out = pipeline.classify(img, tta=True, agg_method='mean', temp=0.1)
            y_true.append(label_to_idx[label])
            # top_class
            y_pred.append(labels.index(out['top_class']))

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred).tolist()

    results = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'labels': labels,
    }

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results
