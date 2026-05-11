"""
DriftGuard-LLM v2 — Confidence-Histogram Concept Drift Detector
================================================================
Key design features:

  - KL divergence computed over *unsafe_confidence* histograms (not label counts)
  - KS test uses scipy.stats.ks_2samp → returns both statistic AND p-value
  - Jensen-Shannon divergence and Wasserstein distance added
  - Overlapping windows via step_size
  - Four reference modes: fixed_first | rolling | multi_reference | adaptive_clean
  - Equal-width and quantile binning
  - Robust record normalisation with confidence_source tracking
  - Audit warnings for constant confidence (e.g. all-0.5 from E1)

Usage
-----
    from scripts.drift_detector_v2 import DriftDetectorV2, load_and_normalise

    records = load_and_normalise("predictions/e0_zero_shot_predictions.jsonl")
    detector = DriftDetectorV2(window_size=200, step_size=100, bins=20,
                                binning="equal_width", reference_mode="fixed_first")
    report = detector.analyse(records)
    detector.save_report(report, "results/drift_v2/reports/e0_report.json")
"""

from __future__ import annotations

import json
import math
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from scipy.stats import ks_2samp, wasserstein_distance
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False
    warnings.warn(
        "scipy not found — KS p-value and Wasserstein distance unavailable. "
        "Install with: pip install scipy",
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Canonical record schema
# ---------------------------------------------------------------------------

@dataclass
class NormalisedRecord:
    """Canonical representation of one prediction record."""
    id: str
    language: str
    true_label: str               # "SAFE" | "UNSAFE"
    predicted: str                # "SAFE" | "UNSAFE"
    pred_confidence: Optional[float]  # raw field as-is from file
    unsafe_confidence: Optional[float]  # probability of UNSAFE ∈ [0, 1]
    confidence_source: str        # "direct_unsafe_probability" | "derived_from_pred_confidence" | "missing"
    source: str
    text: Optional[str] = None
    modality: Optional[str] = None
    raw: Optional[Dict] = None    # original dict for debugging


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

_LABEL_ALIASES: Dict[str, str] = {
    "safe": "SAFE", "0": "SAFE", 0: "SAFE",
    "unsafe": "UNSAFE", "1": "UNSAFE", 1: "UNSAFE",
    "hate": "UNSAFE", "toxic": "UNSAFE", "harmful": "UNSAFE",
}


def _normalise_label(v) -> str:
    """Map any label variant to 'SAFE' or 'UNSAFE'."""
    if v is None:
        return "UNKNOWN"
    if isinstance(v, str):
        mapped = _LABEL_ALIASES.get(v.strip().lower())
        if mapped:
            return mapped
        return v.strip().upper()
    # numeric
    return _LABEL_ALIASES.get(v, "UNKNOWN")


def _get_field(rec: dict, *keys, default=None):
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    return default


def normalise_record(rec: dict) -> NormalisedRecord:
    """
    Map a raw prediction dict (any field naming convention) to NormalisedRecord.

    Confidence derivation logic
    ---------------------------
    Priority order for unsafe_confidence:
      1. unsafe_confidence / unsafe_probability  → direct
      2. safe_probability / safe_confidence       → 1 - value (direct complement)
      3. pred_confidence / confidence / score / prob:
            if predicted == UNSAFE  →  unsafe_confidence = pred_confidence
            if predicted == SAFE    →  unsafe_confidence = 1.0 - pred_confidence
      4. None                       → missing

    Emits a warning if unsafe_confidence appears constant (all equal).
    """
    rid = str(_get_field(rec, "id", "sample_id", "idx", default="unknown"))
    lang = str(_get_field(rec, "lang", "language", "lang_code", default="unknown"))

    true_raw = _get_field(rec, "label", "true_label", "ground_truth", "true")
    pred_raw = _get_field(rec, "pred_label", "predicted_label", "prediction", "pred", "predicted")
    true_label = _normalise_label(true_raw)
    predicted = _normalise_label(pred_raw)

    # --- unsafe_confidence ---
    # 1. Direct unsafe probability
    unsafe_conf_raw = _get_field(rec, "unsafe_confidence", "unsafe_probability")
    if unsafe_conf_raw is not None:
        try:
            unsafe_conf = float(unsafe_conf_raw)
            conf_source = "direct_unsafe_probability"
        except (ValueError, TypeError):
            unsafe_conf = None
            conf_source = "missing"
    else:
        unsafe_conf = None
        conf_source = "missing"

    # 2. Complement from safe_probability
    if unsafe_conf is None:
        safe_prob_raw = _get_field(rec, "safe_probability", "safe_confidence")
        if safe_prob_raw is not None:
            try:
                unsafe_conf = 1.0 - float(safe_prob_raw)
                conf_source = "direct_unsafe_probability"
            except (ValueError, TypeError):
                pass

    # 3. Derive from pred_confidence
    pred_conf_raw = _get_field(rec, "pred_confidence", "confidence", "score", "prob")
    pred_conf = None
    if pred_conf_raw is not None:
        try:
            pred_conf = float(pred_conf_raw)
        except (ValueError, TypeError):
            pred_conf = None

    if unsafe_conf is None and pred_conf is not None:
        if predicted == "UNSAFE":
            unsafe_conf = pred_conf
        elif predicted == "SAFE":
            unsafe_conf = 1.0 - pred_conf
        else:
            unsafe_conf = pred_conf  # unknown label — pass through
        conf_source = "derived_from_pred_confidence"

    if unsafe_conf is None:
        conf_source = "missing"

    return NormalisedRecord(
        id=rid,
        language=lang,
        true_label=true_label,
        predicted=predicted,
        pred_confidence=pred_conf,
        unsafe_confidence=unsafe_conf,
        confidence_source=conf_source,
        source=str(_get_field(rec, "source", default="unknown")),
        text=_get_field(rec, "text", "fused_text", "content"),
        modality=_get_field(rec, "modality"),
        raw=rec,
    )


def load_and_normalise(path: Union[str, Path]) -> List[NormalisedRecord]:
    """Load a JSONL predictions file and return normalised records."""
    path = Path(path)
    records: List[NormalisedRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(normalise_record(json.loads(line)))
            except json.JSONDecodeError:
                continue

    # Audit: warn if unsafe_confidence is constant
    uc_vals = [r.unsafe_confidence for r in records if r.unsafe_confidence is not None]
    if uc_vals:
        unique_uc = set(round(v, 6) for v in uc_vals)
        if len(unique_uc) == 1:
            val = next(iter(unique_uc))
            warnings.warn(
                f"[DriftGuard] {path.name}: unsafe_confidence is CONSTANT = {val} for all "
                f"{len(uc_vals)} records. Confidence-based drift results will be UNINFORMATIVE. "
                f"Rerun inference with real token probabilities before reporting paper results.",
                stacklevel=2,
            )
        elif np.std(uc_vals) < 0.01:
            warnings.warn(
                f"[DriftGuard] {path.name}: unsafe_confidence std={np.std(uc_vals):.4f} "
                f"(near-constant). Drift detection may be unreliable.",
                stacklevel=2,
            )

    return records


# ---------------------------------------------------------------------------
# Histogram utilities
# ---------------------------------------------------------------------------

def _build_histogram(
    values: List[float],
    bins: int = 20,
    binning: str = "equal_width",
    bin_edges: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a normalised histogram over unsafe_confidence values.

    Parameters
    ----------
    values    : List of floats in [0, 1]
    bins      : Number of bins
    binning   : "equal_width" | "quantile"
    bin_edges : If provided, use these edges (for consistent binning across windows)

    Returns
    -------
    hist      : normalised probability array (sums to 1)
    edges     : bin edges used
    """
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        edges = np.linspace(0, 1, bins + 1) if bin_edges is None else bin_edges
        return np.zeros(len(edges) - 1), edges

    if bin_edges is not None:
        edges = bin_edges
    elif binning == "quantile":
        quantiles = np.linspace(0, 100, bins + 1)
        edges = np.percentile(arr, quantiles)
        edges = np.unique(edges)  # deduplicate
        if len(edges) < 2:
            edges = np.linspace(0, 1, bins + 1)
    else:  # equal_width
        edges = np.linspace(0.0, 1.0, bins + 1)

    counts, _ = np.histogram(arr, bins=edges)
    total = counts.sum()
    hist = counts / total if total > 0 else np.zeros_like(counts, dtype=float)
    return hist, edges


# ---------------------------------------------------------------------------
# Divergence metrics
# ---------------------------------------------------------------------------

EPSILON = 1e-10


def kl_divergence_hist(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence D(P‖Q) between two normalised histograms."""
    p = np.clip(p, EPSILON, None)
    q = np.clip(q, EPSILON, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def js_divergence_hist(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen–Shannon divergence (symmetric, bounded [0, ln 2])."""
    p = np.clip(p, 0, None)
    q = np.clip(q, 0, None)
    m = (p + q) / 2
    eps = EPSILON
    m = np.clip(m, eps, None)
    p2 = np.clip(p, eps, None)
    q2 = np.clip(q, eps, None)
    js = 0.5 * np.sum(p2 * np.log(p2 / m)) + 0.5 * np.sum(q2 * np.log(q2 / m))
    return float(js)


def ks_test(ref_vals: List[float], win_vals: List[float]) -> Tuple[float, float]:
    """
    Two-sample KS test using scipy.stats.ks_2samp.

    Returns
    -------
    (ks_statistic, p_value)
    If scipy unavailable, returns (manual_ks_stat, nan).
    """
    if not ref_vals or not win_vals:
        return 0.0, 1.0

    if _SCIPY_OK:
        result = ks_2samp(ref_vals, win_vals)
        return float(result.statistic), float(result.pvalue)
    else:
        # Fallback: manual KS statistic, no p-value
        a = np.sort(ref_vals)
        b = np.sort(win_vals)
        combined = np.sort(np.concatenate([a, b]))
        cdf_a = np.searchsorted(a, combined, side="right") / len(a)
        cdf_b = np.searchsorted(b, combined, side="right") / len(b)
        stat = float(np.max(np.abs(cdf_a - cdf_b)))
        return stat, float("nan")


def wasserstein_dist(ref_vals: List[float], win_vals: List[float]) -> float:
    """1-D Wasserstein (Earth Mover's) distance via scipy."""
    if not ref_vals or not win_vals:
        return 0.0
    if _SCIPY_OK:
        return float(wasserstein_distance(ref_vals, win_vals))
    # Fallback: area between sorted CDFs
    a = np.sort(ref_vals)
    b = np.sort(win_vals)
    n = max(len(a), len(b))
    grid = np.linspace(0, 1, n + 1)
    cdf_a = np.interp(grid, np.linspace(0, 1, len(a)), np.sort(a))
    cdf_b = np.interp(grid, np.linspace(0, 1, len(b)), np.sort(b))
    return float(np.mean(np.abs(cdf_a - cdf_b)))


# ---------------------------------------------------------------------------
# Window stats dataclass
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    window_idx: int
    start_record: int
    end_record: int
    n_records: int
    # Histogram metrics vs reference
    kl: float
    js: float
    ks_stat: float
    ks_pvalue: float
    wasserstein: float
    # Distribution summary
    unsafe_conf_mean: float
    unsafe_conf_std: float
    unsafe_conf_min: float
    unsafe_conf_max: float
    n_missing_conf: int
    # Label distribution
    label_counts: Dict[str, int]
    language_counts: Dict[str, int]
    # Drift decision
    drift_detected: bool
    drift_rule: str
    severity: str       # "none" | "warning" | "critical"
    # Which reference was used
    reference_window_idx: int


@dataclass
class DriftReportV2:
    source_file: str
    n_total_records: int
    window_size: int
    step_size: int
    bins: int
    binning: str
    reference_mode: str
    detector_rule: str
    # Calibrated or manual thresholds
    kl_warn: float
    kl_crit: float
    ks_alpha: float
    js_warn: float
    wasserstein_warn: float
    # Per-window results
    windows: List[WindowResult]
    # High-level summary
    drift_detected: bool
    first_detection_window: Optional[int]
    n_warnings: int
    n_criticals: int
    false_alarm_count_before_drift: int  # windows before drift_point with alert (for synthetic)
    max_kl: float
    min_ks_pvalue: float
    max_js: float
    max_wasserstein: float
    confidence_source_counts: Dict[str, int]
    constant_confidence_warning: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class DriftDetectorV2:
    """
    Sliding-window concept drift detector using *unsafe_confidence* histograms.

    Parameters
    ----------
    window_size      : Records per window (default 200)
    step_size        : Step between consecutive windows. step_size < window_size → overlapping.
                       Default = window_size (non-overlapping).
    bins             : Number of histogram bins (default 20)
    binning          : "equal_width" | "quantile"
    reference_mode   : "fixed_first" | "rolling" | "multi_reference" | "adaptive_clean"
    reference_size   : Records for initial reference window (default = window_size)
    detector_rule    : "KL_only" | "KS_only" | "KL_AND_KS" | "JS_only"
                       | "Wasserstein_only" | "Ensemble_majority"
    kl_warn          : KL warning threshold (default 0.05, calibrate with calibrate_thresholds.py)
    kl_crit          : KL critical threshold (default 0.15)
    ks_alpha         : KS p-value significance level (default 0.05)
    js_warn          : JS warning threshold (default 0.03)
    wasserstein_warn : Wasserstein warning threshold (default 0.05)
    """

    def __init__(
        self,
        window_size: int = 200,
        step_size: Optional[int] = None,
        bins: int = 20,
        binning: str = "equal_width",
        reference_mode: str = "fixed_first",
        reference_size: Optional[int] = None,
        detector_rule: str = "KL_AND_KS",
        kl_warn: float = 0.05,
        kl_crit: float = 0.15,
        ks_alpha: float = 0.05,
        js_warn: float = 0.03,
        wasserstein_warn: float = 0.05,
    ):
        self.window_size = window_size
        self.step_size = step_size if step_size is not None else window_size
        self.bins = bins
        self.binning = binning
        self.reference_mode = reference_mode
        self.reference_size = reference_size if reference_size is not None else window_size
        self.detector_rule = detector_rule
        self.kl_warn = kl_warn
        self.kl_crit = kl_crit
        self.ks_alpha = ks_alpha
        self.js_warn = js_warn
        self.wasserstein_warn = wasserstein_warn

    # ------------------------------------------------------------------
    def analyse(
        self,
        records: List[NormalisedRecord],
        drift_point: Optional[int] = None,
        source_file: str = "unknown",
    ) -> DriftReportV2:
        """
        Run drift analysis over a list of normalised records.

        Parameters
        ----------
        records      : Output of load_and_normalise()
        drift_point  : Record index where synthetic drift begins (for eval metrics)
        source_file  : Label for the source file (informational)
        """
        n_total = len(records)
        uc_all = [r.unsafe_confidence for r in records]

        # Confidence source audit
        source_counts: Counter = Counter(r.confidence_source for r in records)
        constant_conf_warning = self._check_constant(uc_all)

        # Build reference bin edges from the full data for consistent binning
        ref_vals_for_edges = [v for v in uc_all[:self.reference_size] if v is not None]
        _, shared_edges = _build_histogram(
            ref_vals_for_edges if ref_vals_for_edges else [0.0, 1.0],
            bins=self.bins, binning=self.binning
        )

        # Initial reference window
        ref_records = records[:self.reference_size]
        ref_uc = [r.unsafe_confidence for r in ref_records if r.unsafe_confidence is not None]
        ref_hist, _ = _build_histogram(ref_uc, bins=self.bins, binning=self.binning,
                                        bin_edges=shared_edges)
        ref_stack = [ref_records]  # for multi_reference mode

        windows: List[WindowResult] = []
        start = self.reference_size

        while start < n_total:
            end = min(start + self.window_size, n_total)
            chunk = records[start:end]

            if len(chunk) < max(5, self.window_size // 10):
                break

            win_uc = [r.unsafe_confidence for r in chunk if r.unsafe_confidence is not None]
            win_hist, _ = _build_histogram(win_uc, bins=self.bins, binning=self.binning,
                                            bin_edges=shared_edges)

            # Compute all metrics vs current reference
            kl = kl_divergence_hist(ref_hist, win_hist)
            js = js_divergence_hist(ref_hist, win_hist)
            ks_stat, ks_pval = ks_test(ref_uc, win_uc)
            wass = wasserstein_dist(ref_uc, win_uc)

            # Drift decision per rule
            drift_flag, severity = self._apply_rule(kl, ks_pval, js, wass)

            win_idx = len(windows)
            win_result = WindowResult(
                window_idx=win_idx,
                start_record=start,
                end_record=end - 1,
                n_records=len(chunk),
                kl=round(kl, 6),
                js=round(js, 6),
                ks_stat=round(ks_stat, 6),
                ks_pvalue=round(ks_pval, 6) if not math.isnan(ks_pval) else float("nan"),
                wasserstein=round(wass, 6),
                unsafe_conf_mean=round(float(np.mean(win_uc)), 4) if win_uc else float("nan"),
                unsafe_conf_std=round(float(np.std(win_uc)), 4) if win_uc else float("nan"),
                unsafe_conf_min=round(float(np.min(win_uc)), 4) if win_uc else float("nan"),
                unsafe_conf_max=round(float(np.max(win_uc)), 4) if win_uc else float("nan"),
                n_missing_conf=sum(1 for r in chunk if r.unsafe_confidence is None),
                label_counts=dict(Counter(r.predicted for r in chunk)),
                language_counts=dict(Counter(r.language for r in chunk)),
                drift_detected=drift_flag,
                drift_rule=self.detector_rule,
                severity=severity,
                reference_window_idx=len(ref_stack) - 1,
            )
            windows.append(win_result)

            # Update reference
            ref_records, ref_uc, ref_hist = self._update_reference(
                ref_records, ref_uc, ref_hist,
                chunk, win_uc, win_hist,
                drift_flag, shared_edges, ref_stack
            )

            start += self.step_size

        # Summary
        drift_windows = [w for w in windows if w.drift_detected]
        first_det = drift_windows[0].window_idx if drift_windows else None
        n_warn = sum(1 for w in windows if w.severity == "warning")
        n_crit = sum(1 for w in windows if w.severity == "critical")

        # False alarms before drift_point (for synthetic stream evaluation)
        false_alarms = 0
        if drift_point is not None:
            false_alarms = sum(
                1 for w in windows
                if w.drift_detected and w.end_record < drift_point
            )

        return DriftReportV2(
            source_file=source_file,
            n_total_records=n_total,
            window_size=self.window_size,
            step_size=self.step_size,
            bins=self.bins,
            binning=self.binning,
            reference_mode=self.reference_mode,
            detector_rule=self.detector_rule,
            kl_warn=self.kl_warn,
            kl_crit=self.kl_crit,
            ks_alpha=self.ks_alpha,
            js_warn=self.js_warn,
            wasserstein_warn=self.wasserstein_warn,
            windows=windows,
            drift_detected=len(drift_windows) > 0,
            first_detection_window=first_det,
            n_warnings=n_warn,
            n_criticals=n_crit,
            false_alarm_count_before_drift=false_alarms,
            max_kl=round(max((w.kl for w in windows), default=0.0), 6),
            min_ks_pvalue=round(min((w.ks_pvalue for w in windows
                                     if not math.isnan(w.ks_pvalue)), default=1.0), 6),
            max_js=round(max((w.js for w in windows), default=0.0), 6),
            max_wasserstein=round(max((w.wasserstein for w in windows), default=0.0), 6),
            confidence_source_counts=dict(source_counts),
            constant_confidence_warning=constant_conf_warning,
            metadata={"drift_point": drift_point},
        )

    # ------------------------------------------------------------------
    def _apply_rule(
        self, kl: float, ks_pval: float, js: float, wass: float
    ) -> Tuple[bool, str]:
        """
        Apply the chosen detector rule.
        Returns (drift_detected: bool, severity: str)
        """
        kl_warn_flag = kl >= self.kl_warn
        kl_crit_flag = kl >= self.kl_crit
        ks_flag = (not math.isnan(ks_pval)) and ks_pval < self.ks_alpha
        js_flag = js >= self.js_warn
        wass_flag = wass >= self.wasserstein_warn

        rule = self.detector_rule
        if rule == "KL_only":
            detected = kl_warn_flag
        elif rule == "KS_only":
            detected = ks_flag
        elif rule == "KL_AND_KS":
            detected = kl_warn_flag and ks_flag
        elif rule == "JS_only":
            detected = js_flag
        elif rule == "Wasserstein_only":
            detected = wass_flag
        elif rule == "Ensemble_majority":
            votes = [kl_warn_flag, ks_flag, js_flag, wass_flag]
            detected = sum(votes) >= 2  # majority (2 of 4)
        else:
            detected = kl_warn_flag

        if not detected:
            return False, "none"
        severity = "critical" if kl_crit_flag else "warning"
        return True, severity

    # ------------------------------------------------------------------
    def _update_reference(
        self,
        ref_records, ref_uc, ref_hist,
        chunk, win_uc, win_hist,
        drift_flag: bool,
        shared_edges: np.ndarray,
        ref_stack: list,
    ):
        """Update the reference window according to reference_mode."""
        mode = self.reference_mode

        if mode == "fixed_first":
            # Never update reference
            return ref_records, ref_uc, ref_hist

        elif mode == "rolling":
            # Always slide reference to the latest window
            new_uc = win_uc
            new_hist, _ = _build_histogram(new_uc, bins=self.bins, binning=self.binning,
                                            bin_edges=shared_edges)
            ref_stack.append(chunk)
            return chunk, new_uc, new_hist

        elif mode == "multi_reference":
            # Accumulate all previous windows into reference
            all_prev = []
            for prev in ref_stack:
                all_prev.extend(prev)
            all_uc = [r.unsafe_confidence for r in all_prev if r.unsafe_confidence is not None]
            all_hist, _ = _build_histogram(all_uc, bins=self.bins, binning=self.binning,
                                            bin_edges=shared_edges)
            ref_stack.append(chunk)
            return all_prev, all_uc, all_hist

        elif mode == "adaptive_clean":
            # Only update reference if NO drift detected (stay clean)
            if not drift_flag:
                new_uc = win_uc
                new_hist, _ = _build_histogram(new_uc, bins=self.bins, binning=self.binning,
                                                bin_edges=shared_edges)
                ref_stack.append(chunk)
                return chunk, new_uc, new_hist
            else:
                return ref_records, ref_uc, ref_hist

        return ref_records, ref_uc, ref_hist

    # ------------------------------------------------------------------
    @staticmethod
    def _check_constant(uc_vals: List[Optional[float]]) -> bool:
        vals = [v for v in uc_vals if v is not None]
        if not vals:
            return False
        return len(set(round(v, 6) for v in vals)) == 1

    # ------------------------------------------------------------------
    def save_report(self, report: DriftReportV2, output_path: Union[str, Path]) -> None:
        """Save drift report as pretty-printed JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def _serial(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if hasattr(obj, "__dataclass_fields__"):
                return asdict(obj)
            raise TypeError(f"Not serialisable: {type(obj)}")

        report_dict = asdict(report)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, default=_serial)
        print(f"[DriftGuard] Report saved → {output_path}")

    # ------------------------------------------------------------------
    def print_summary(self, report: DriftReportV2) -> None:
        width = 72
        print("=" * width)
        print("DriftGuard-LLM v2  |  Confidence-Histogram Drift Report")
        print(f"  Source     : {Path(report.source_file).name}")
        print(f"  Records    : {report.n_total_records:,}  |  Windows: {len(report.windows)}")
        print(f"  Window     : {report.window_size}  Step: {report.step_size}  "
              f"Bins: {report.bins} ({report.binning})")
        print(f"  Ref. mode  : {report.reference_mode}  Rule: {report.detector_rule}")
        print("-" * width)
        print(f"  Drift detected    : {report.drift_detected}")
        if report.first_detection_window is not None:
            print(f"  First detection   : window {report.first_detection_window}")
        print(f"  Warnings / Crits  : {report.n_warnings} / {report.n_criticals}")
        print(f"  Max KL            : {report.max_kl:.4f}  (warn≥{report.kl_warn}, "
              f"crit≥{report.kl_crit})")
        print(f"  Min KS p-value    : {report.min_ks_pvalue:.4f}  (α={report.ks_alpha})")
        print(f"  Max JS            : {report.max_js:.4f}  (warn≥{report.js_warn})")
        print(f"  Max Wasserstein   : {report.max_wasserstein:.4f}  "
              f"(warn≥{report.wasserstein_warn})")
        print(f"  Conf sources      : {dict(report.confidence_source_counts)}")
        if report.constant_confidence_warning:
            print("  *** WARNING: unsafe_confidence is CONSTANT — results not valid for paper ***")
        print("-" * width)
        if report.windows:
            print("  Per-window (KL | KS-p | JS | Wass | Drift):")
            for w in report.windows:
                flag = "⚠ DRIFT" if w.drift_detected else "  ok   "
                print(f"    W{w.window_idx:03d} [{w.start_record:5d}-{w.end_record:5d}] "
                      f"KL={w.kl:.4f} KS-p={w.ks_pvalue:.3f} JS={w.js:.4f} "
                      f"W={w.wasserstein:.4f}  {flag}")
        print("=" * width)
