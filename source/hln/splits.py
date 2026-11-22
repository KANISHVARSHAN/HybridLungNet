
# ============================================================================ SPLITS
# One patient-level split shared by every experiment.
#
#  TEST  = (a) fold-2 validation patients of the ORIGINAL classifier CV
#              -> the only patients the existing checkpoint best_hybrid_fold2.pth never trained on
#          (b) + 20 % (seed 0) of patients the classifier never used at all
#              (no clear benign/malignant nodule: only ambiguous, tiny or no nodules) so the
#              test set also contains scans where the right answer is "nothing / not sure".
#  DET_VAL   = 15 % (seed 0) of the remaining patients: detector checkpoint + threshold choice
#  DET_TRAIN = the rest
# TEST patients are never used for training or for choosing any setting.
import numpy as _np
import pandas as _pd
from sklearn.model_selection import StratifiedKFold as _SKF


def classifier_folds(final_df):
    """Exactly reproduces the notebook split (cell 10). Returns list of (train_pts, val_pts)."""
    clean = final_df[final_df.label != -1]
    pat = clean.groupby('patient_id')['label'].apply(lambda x: int(x.mean() >= 0.5)).reset_index()
    pat.columns = ['patient_id', 'pat_label']
    skf = _SKF(n_splits=5, shuffle=True, random_state=42)
    return [(set(pat.patient_id.iloc[a]), set(pat.patient_id.iloc[b])) for a, b in skf.split(pat.patient_id, pat.pat_label)]


def make_split(final_df, available_patients):
    avail = sorted(set(available_patients))
    folds = classifier_folds(final_df)
    in_cv = set().union(*[v for _, v in folds])
    fold2_val = sorted(folds[1][1] & set(avail))
    unused = sorted(set(avail) - in_cv)
    rng = _np.random.default_rng(0)
    extra = sorted(rng.choice(unused, size=int(round(0.2 * len(unused))), replace=False).tolist()) if unused else []
    test = sorted(set(fold2_val) | set(extra))
    rest = sorted(set(avail) - set(test))
    rng = _np.random.default_rng(0)
    det_val = sorted(rng.choice(rest, size=int(round(0.15 * len(rest))), replace=False).tolist())
    det_train = sorted(set(rest) - set(det_val))
    assert not (set(test) & set(det_train)) and not (set(test) & set(det_val)) and not (set(det_val) & set(det_train))
    return dict(test=test, test_from_fold2=fold2_val, test_from_unused=extra, det_val=det_val, det_train=det_train)
