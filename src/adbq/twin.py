"""A digital twin of the two-country disaster-bond portfolio.

The twin is a versioned portfolio model that answers a stream of related questions, each with
a certified confidence interval and a recorded query cost, and reuses work between nearby
model versions. It runs classically; the certified estimator is iterative quantum amplitude
estimation (IQAE, Chernoff-Hoeffding bounds by default), emulated shot by shot at the exact amplitude of the
verified circuit (oracle.py), so every interval can be checked against the exact answer.

Building blocks (each one is checked against exact values in tests/test_twin.py):

* ``SourceLaw``      the term law as independent sources (model.py): own quakes per country,
                     the shared quake pair, and the three rain triggers. Every version of the
                     model is a ``Scenario``; the twin keeps them all.
* ``ratio_bounds``   exact likelihood-ratio bounds between two versions, source by source
                     (min / max of p'_s / p_s over each source, multiplied). Scalable: the
                     bound never touches the joint state space.
* ``Ledger``         every quantum query spent, by purpose.
* ``certified``      one certified estimate: IQAE at a target relative half-width.
* ``sweep``          dependence sweep with interval transport: an interval [L, U] for
                     mu = E_p[f] (f >= 0) at an anchor version gives [r_lo L, r_hi U] at any
                     version whose source ratios are bounded by [r_lo, r_hi] (no new queries);
                     the controller recalibrates only when the transported interval is too wide.
* ``ScenarioLibrary`` conditional tail scenarios (exact samples given an event, as amplitude
                     amplification would return them), reused across versions by exact
                     thinning: keep a sample with probability r(x) / R, R >= max r on the event.
* ``paired_sign``    certify which of two contract structures has the larger expected payoff
                     by estimating the difference in one circuit (amplitude 1/2 + E[H] / 2R).
* ``allocate``       split a query budget over many targets: n_q proportional to
                     (w_q kappa_q^2)^(1/3), the minimiser of sum w_q (kappa_q / n_q)^2.

Query convention as in qae.py: one query = one application of A or A^dagger.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .model import STEPS, Scenario, payoff_arrays, two_country_sources
from .qae import iqae

# Interval method for every certified estimate in the twin. Chernoff-Hoeffding ("ch") is the
# default: in experiments/twin_intervals.py and experiments/coverage.py both methods meet the
# 99% coverage, but a Clopper-Pearson ("cp") miss can be confidently wrong by several times the
# value (IQAE locks onto a wrong angle branch), and the twin carries anchor intervals across
# many model versions, so one bad anchor would spread through a sweep.
METHOD = "ch"

SHAPE = (STEPS + 1, STEPS + 1, STEPS + 1, STEPS + 1, 2, 2, 2)   # o1, o2, x, y, u1, u2, z


# ---------------------------------------------------------------------------
# The source law of one model version
# ---------------------------------------------------------------------------
@dataclass
class SourceLaw:
    s: Scenario
    factors: dict = field(init=False)
    joint: np.ndarray = field(init=False)
    term_index: tuple = field(init=False)

    def __post_init__(self):
        src = two_country_sources(self.s)
        b = lambda q: np.array([1 - q, q])
        self.factors = {"own_quake_1": src.own_quake[0], "own_quake_2": src.own_quake[1],
                        "shared_quake": src.shared_quake, "own_rain_1": b(src.own_rain[0]),
                        "own_rain_2": b(src.own_rain[1]), "shared_rain": b(src.shared_rain)}
        f = self.factors
        self.joint = np.einsum("a,b,xy,u,v,z->abxyuvz", f["own_quake_1"], f["own_quake_2"], f["shared_quake"],
                               f["own_rain_1"], f["own_rain_2"], f["shared_rain"])
        o1, o2, x, y, u1, u2, z = np.indices(SHAPE)
        self.term_index = (np.minimum(STEPS, o1 + x), np.minimum(STEPS, o2 + y), u1 | z, u2 | z)

    def payoff(self, key: str, s: Scenario | None = None) -> np.ndarray:
        """Payoff of target ``key`` on every source configuration (contract of ``s``)."""
        return payoff_arrays(s or self.s)[key][self.term_index]

    def mean(self, f: np.ndarray) -> float:
        return float((self.joint * f).sum())


def ratio_bounds(old: SourceLaw, new: SourceLaw) -> tuple[float, float]:
    """Bounds on p_new / p_old over every configuration that the new law can produce.

    Returns (r_lo, r_hi). r_hi = inf if the new law puts mass where the old one has none:
    then nothing can be transported and the twin must recalibrate."""
    lo, hi = 1.0, 1.0
    for k, pn in new.factors.items():
        po = old.factors[k]
        sup = pn > 0
        if np.any(po[sup] == 0):
            return 0.0, np.inf
        r = pn[sup] / po[sup]
        lo *= r.min()
        hi *= r.max()
    return float(lo), float(hi)


def exact_ratio(old: SourceLaw, new: SourceLaw) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(old.joint > 0, new.joint / np.where(old.joint > 0, old.joint, 1), np.where(new.joint > 0, np.inf, 0))


# ---------------------------------------------------------------------------
# Certified estimates and the query ledger
# ---------------------------------------------------------------------------
@dataclass
class Ledger:
    entries: list = field(default_factory=list)

    def add(self, purpose: str, queries: float, **info):
        self.entries.append({"purpose": purpose, "queries": float(queries), **info})

    def total(self, purpose: str | None = None) -> float:
        return float(sum(e["queries"] for e in self.entries if purpose is None or e["purpose"] == purpose))


def certified(mu: float, f_max: float, rel_half_width: float, rng, alpha: float = 0.01, ledger: Ledger | None = None,
              purpose: str = "calibrate", method: str | None = None, **info) -> tuple[float, float, float]:
    """Certified estimate of mu = E[f], 0 <= f <= f_max, by IQAE on a = mu / f_max
    (``method``: "ch" or "cp"; default METHOD).

    Returns (estimate, lower, upper) in the units of f."""
    a = mu / f_max
    if a <= 0:
        return 0.0, 0.0, 0.0
    est, n, (lo, hi) = iqae(a, rel_half_width * a, rng, alpha=alpha, method=method or METHOD)
    if ledger is not None:
        ledger.add(purpose, n, **info)
    return est * f_max, lo * f_max, hi * f_max


# ---------------------------------------------------------------------------
# Dependence sweep with interval transport
# ---------------------------------------------------------------------------
def _transported_width(h: float, r_lo: float, r_hi: float) -> float:
    """Relative half-width after transporting [mu(1-h), mu(1+h)] with ratio bounds."""
    lo, hi = r_lo * (1 - h), r_hi * (1 + h)
    return (hi - lo) / (hi + lo)


def plan_calibration(laws, i: int, tol: float, fractions=(1.0, 0.8, 0.6, 0.45, 0.3)) -> tuple[float, int]:
    """Look-ahead planning (the ratio bounds are exact and cost no queries): choose the
    calibration half-width h = f * tol at version i that covers the most upcoming versions per
    query (IQAE cost grows as 1 / h). Returns (h, versions covered)."""
    best = (tol, 1, 0.0)
    for f in fractions:
        h = f * tol
        n = 1
        for j in range(i + 1, len(laws)):
            r_lo, r_hi = ratio_bounds(laws[i], laws[j])
            if not np.isfinite(r_hi) or _transported_width(h, r_lo, r_hi) > tol:
                break
            n += 1
        score = n * h
        if score > best[2] * (1 + 1e-9):
            best = (h, n, score)
    return best[0], best[1]


def sweep(base: Scenario, param: str, values, key: str, tol: float, rng, alpha: float = 0.01,
          plan: bool = True, cal_fraction: float = 0.5, method: str | None = None) -> dict:
    """Sweep ``param`` over ``values`` for target ``key``; certify every point to relative
    half-width ``tol``.

    Twin: calibrate at an anchor, then transport the interval with the source ratio bounds
    until it would be wider than ``tol``; then recalibrate and re-anchor. With ``plan`` the
    calibration precision is chosen by look-ahead over the upcoming versions
    (``plan_calibration``); otherwise it is ``cal_fraction * tol``.
    Baseline: an independent certified estimate at every point, to half-width ``tol``.
    Coverage of both is checked against the exact value."""
    led_twin, led_base = Ledger(), Ledger()
    laws = [SourceLaw(replace(base, **{param: float(v)})) for v in values]
    rows = []
    anchor, interval = None, None
    for i, (v, law) in enumerate(zip(values, laws)):
        f = law.payoff(key)
        f_max = float(f.max())
        mu = law.mean(f)
        _, blo, bhi = certified(mu, f_max, tol, rng, alpha, led_base, "recompute", method, value=float(v))
        mode = "transport"
        if anchor is not None:
            r_lo, r_hi = ratio_bounds(anchor, law)
            lo, hi = r_lo * interval[0], min(f_max, r_hi * interval[1])
        if anchor is None or not np.isfinite(hi) or (hi - lo) / (hi + lo) > tol:
            mode = "recalibrate"
            h = plan_calibration(laws, i, tol)[0] if plan else cal_fraction * tol
            _, lo, hi = certified(mu, f_max, h, rng, alpha, led_twin, "calibrate", method, value=float(v))
            anchor, interval = law, (lo, hi)
        rows.append({"value": float(v), "exact": mu, "lo": lo, "hi": hi, "mode": mode,
                     "covered": bool(lo <= mu <= hi), "base_covered": bool(blo <= mu <= bhi),
                     "rel_half_width": (hi - lo) / (hi + lo) if hi + lo > 0 else 0.0})
    return {"param": param, "key": key, "tol": tol, "rows": rows,
            "twin_queries": led_twin.total(), "baseline_queries": led_base.total(),
            "recalibrations": sum(r["mode"] == "recalibrate" for r in rows)}


# ---------------------------------------------------------------------------
# Conditional tail scenarios
# ---------------------------------------------------------------------------
def amplification_cost(p_event: float) -> float:
    """Expected queries for one exact conditional sample by amplitude amplification with the
    near-optimal number of Grover rounds k* (a failed round is repeated)."""
    th = np.arcsin(np.sqrt(p_event))
    k = max(0, int(np.floor(np.pi / (4 * th) - 0.5)))
    succ = np.sin((2 * k + 1) * th) ** 2
    return float((2 * k + 1) / succ)


@dataclass
class ScenarioLibrary:
    """Exact conditional samples of source configurations given an event, cached and reused."""
    law: SourceLaw
    event: np.ndarray                        # boolean over SHAPE
    samples: np.ndarray                      # flat indices into SHAPE

    @classmethod
    def generate(cls, law: SourceLaw, event: np.ndarray, n: int, rng, ledger: Ledger):
        p = np.where(event, law.joint, 0.0).ravel()
        pe = p.sum()
        idx = rng.choice(p.size, size=n, p=p / pe)
        ledger.add("generate", n * amplification_cost(pe), samples=n)
        return cls(law, event, idx)

    def move_to(self, new: SourceLaw, n_target: int, rng, ledger: Ledger, ratio_bound: float | None = None):
        """Exact thinning to the new version, then top up with fresh samples."""
        r = exact_ratio(self.law, new).ravel()
        R = ratio_bound if ratio_bound is not None else ratio_bounds(self.law, new)[1]
        if not np.isfinite(R):
            fresh = ScenarioLibrary.generate(new, self.event, n_target, rng, ledger)
            return fresh, 0
        keep = rng.random(self.samples.size) < r[self.samples] / R
        kept = self.samples[keep]
        top = max(0, n_target - kept.size)
        p = np.where(self.event, new.joint, 0.0).ravel()
        pe = p.sum()
        add = rng.choice(p.size, size=top, p=p / pe) if top else np.array([], int)
        if top:
            ledger.add("top-up", top * amplification_cost(pe), samples=top)
        return ScenarioLibrary(new, self.event, np.concatenate([kept, add])), int(kept.size)


def drivers(law: SourceLaw, idx: np.ndarray | None = None) -> dict:
    """What drives 'both bonds exhausted': shares of quake and rain patterns among the
    configurations (exact if idx is None, else in the sample)."""
    o1, o2, x, y, u1, u2, z = (np.indices(SHAPE)[i].ravel() for i in range(7))
    cats = {
        "shared quakes alone exhaust both quake covers": (x >= STEPS) & (y >= STEPS),
        "each country's own quakes alone exhaust its cover": (o1 >= STEPS) & (o2 >= STEPS),
        "a shared storm triggers both rain covers": z == 1,
    }
    if idx is None:
        ev = (law.payoff("term_both_bonds_exhausted_probability") > 0).ravel()
        w = np.where(ev, law.joint.ravel(), 0.0)
        return {k: float(w[c].sum() / w.sum()) for k, c in cats.items()}
    return {k: float(c[idx].mean()) for k, c in cats.items()}


# ---------------------------------------------------------------------------
# Paired structure comparison
# ---------------------------------------------------------------------------
def paired_sign(law: SourceLaw, g1: np.ndarray, g2: np.ndarray, rng, alpha: float = 0.01, max_rounds: int = 14,
                method: str | None = None):
    """Certify the sign of E[g1 - g2] with one circuit: amplitude a = 1/2 + E[H] / (2R).

    Certify-or-stop: halve the IQAE target half-width until the interval excludes 1/2. The
    failure budget is split over the rounds (union bound), so a wrong sign has probability at
    most alpha in total, not alpha per round.
    Returns (sign or 0 if not certified, queries)."""
    H = g1 - g2
    R = float(np.abs(H[law.joint > 0]).max())
    a = 0.5 + law.mean(H) / (2 * R)
    eps, total = 0.25, 0.0
    for _ in range(max_rounds):
        _, n, (lo, hi) = iqae(a, eps, rng, alpha=alpha / max_rounds, method=method or METHOD)
        total += n
        if lo > 0.5:
            return 1, total
        if hi < 0.5:
            return -1, total
        eps /= 2
    return 0, total


def separate_sign(law: SourceLaw, g1: np.ndarray, g2: np.ndarray, rng, alpha: float = 0.01, max_rounds: int = 14,
                  method: str | None = None):
    """The same decision from two separate certified estimates (intervals must separate)."""
    m1, m2 = law.mean(g1), law.mean(g2)
    f1, f2 = float(g1.max()), float(g2.max())
    eps, total = 0.25, 0.0
    for _ in range(max_rounds):
        a_r = alpha / (2 * max_rounds)                 # two estimates per round, union bound over rounds
        _, n1, (l1, h1) = iqae(m1 / f1, eps * m1 / f1, rng, alpha=a_r, method=method or METHOD)
        _, n2, (l2, h2) = iqae(m2 / f2, eps * m2 / f2, rng, alpha=a_r, method=method or METHOD)
        total += n1 + n2
        l1, h1, l2, h2 = l1 * f1, h1 * f1, l2 * f2, h2 * f2
        if l1 > h2:
            return 1, total
        if h1 < l2:
            return -1, total
        eps /= 2
    return 0, total


def crn_classical_samples(law: SourceLaw, g1: np.ndarray, g2: np.ndarray, z: float = 2.576) -> float:
    """Classical paired Monte Carlo (common random numbers): scenarios needed for a z-sigma
    separation of the mean difference from zero (normal approximation)."""
    H = g1 - g2
    m = law.mean(H)
    v = law.mean(H ** 2) - m * m
    return float((z * np.sqrt(v) / abs(m)) ** 2)


# ---------------------------------------------------------------------------
# Budget allocation
# ---------------------------------------------------------------------------
def allocate(kappa: np.ndarray, weights: np.ndarray, budget: float) -> np.ndarray:
    """Queries per target minimising sum_q w_q (kappa_q / n_q)^2 subject to sum n_q = budget
    (error of a QAE estimate falls as kappa / n)."""
    x = (weights * kappa ** 2) ** (1 / 3)
    return budget * x / x.sum()


def weighted_error(kappa, weights, n) -> float:
    return float(np.sum(weights * (kappa / n) ** 2))


# ---------------------------------------------------------------------------
# The twin as a service: ``adbq twin config.yaml``
# ---------------------------------------------------------------------------
TWIN_PARAMS = {"quake_mult": "shared-earthquake multiplier m", "rain_shared_share": "shared-rain share",
               "rain_mult": "rain rate multiplier"}


def load_twin_config(path) -> dict:
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg.setdefault("name", Path(path).stem)
    cfg.setdefault("seed", 1)
    cfg.setdefault("confidence", 0.99)
    cfg.setdefault("interval", METHOD)
    if cfg["interval"] not in ("ch", "cp"):
        raise ValueError("interval must be 'ch' (Chernoff-Hoeffding, default) or 'cp' (Clopper-Pearson)")
    if "portfolio" not in cfg:
        raise ValueError("twin config needs 'portfolio': a two_country config file (see configs/)")
    sw = cfg.get("sweep", {})
    if sw and sw.get("parameter") not in TWIN_PARAMS:
        raise ValueError(f"sweep.parameter must be one of {list(TWIN_PARAMS)}")
    p = Path(cfg["portfolio"])
    cfg["_portfolio_path"] = str(p if p.is_absolute() else (Path(path).parent / p if (Path(path).parent / p).exists() else p))
    return cfg


def run_twin(cfg: dict, out_root="results/twin_runs"):
    """Answer the questions in a twin config with certified, ledgered estimates."""
    import json
    import time
    from pathlib import Path
    from .pipeline import TWO_COUNTRY_KEYS, load_config, scenario
    t0 = time.time()
    pcfg = load_config(cfg["_portfolio_path"])
    if pcfg["model"] != "two_country":
        raise ValueError("the twin currently runs on the two_country model")
    base = scenario(pcfg)
    rng = np.random.default_rng(cfg["seed"])
    alpha = 1 - float(cfg["confidence"])
    m = cfg["interval"]
    rep = {"name": cfg["name"], "portfolio": pcfg["name"], "confidence": cfg["confidence"], "interval": m,
           "questions": {}}
    law = SourceLaw(base)
    # 1. certified answers at the current version
    ans = {}
    led = Ledger()
    hw = float(cfg.get("half_width", 0.05))
    for t in cfg.get("targets", list(TWO_COUNTRY_KEYS)):
        f = law.payoff(TWO_COUNTRY_KEYS[t])
        est, lo, hi = certified(law.mean(f), float(f.max()), hw, rng, alpha, led, "answer", m, target=t)
        ans[t] = {"estimate": est, "lo": lo, "hi": hi, "exact": law.mean(f)}
    rep["questions"]["certified answers"] = {"half_width": hw, "answers": ans, "queries": led.total()}
    # 2. dependence sweep
    if cfg.get("sweep"):
        sw = cfg["sweep"]
        grid = np.round(np.arange(float(sw["from"]), float(sw["to"]) + 1e-9, float(sw["step"])), 6)
        res = {}
        for t in sw.get("targets", ["P(both exhausted)"]):
            r = sweep(base, sw["parameter"], grid, TWO_COUNTRY_KEYS[t], float(sw.get("half_width", 0.1)), rng, alpha,
                      method=m)
            res[t] = {k: r[k] for k in ("twin_queries", "baseline_queries", "recalibrations")} | {
                "coverage": float(np.mean([x["covered"] for x in r["rows"]])), "rows": r["rows"]}
        rep["questions"]["dependence sweep"] = {"parameter": sw["parameter"], "label": TWIN_PARAMS[sw["parameter"]],
                                                "grid": grid.tolist(), "targets": res}
    # 3. structure comparison
    if cfg.get("structures"):
        st = cfg["structures"]
        key = TWO_COUNTRY_KEYS[st.get("metric", "stop-loss")]
        total = base.quake_limit + base.rain_limit
        qls = [float(x) * 1e6 for x in st["quake_limits_musd"]]
        g = {ql: law.payoff(key, replace(base, quake_limit=ql, rain_limit=total - ql)) for ql in qls}
        ex = {ql: law.mean(g[ql]) for ql in qls}
        cmp_ = []
        for a, b in zip(qls[:-1], qls[1:]):
            sgn, n = paired_sign(law, g[a], g[b], rng, alpha, method=m)
            cmp_.append({"A": f"{a / 1e6:g}/{(total - a) / 1e6:g}", "B": f"{b / 1e6:g}/{(total - b) / 1e6:g}",
                         "certified": {1: "A > B", -1: "A < B", 0: "not certified"}[sgn], "queries": n,
                         "exact_A": ex[a], "exact_B": ex[b], "correct": sgn == int(np.sign(ex[a] - ex[b])),
                         "classical_crn_scenarios": crn_classical_samples(law, g[a], g[b])})
        rep["questions"]["structure comparison"] = {"metric": st.get("metric", "stop-loss"), "pairs": cmp_}
    # 4. conditional tail scenarios
    if cfg.get("library"):
        lb = cfg["library"]
        key = TWO_COUNTRY_KEYS[lb.get("event", "P(both exhausted)")]
        ev = law.payoff(key) > 0
        led = Ledger()
        lib = ScenarioLibrary.generate(law, ev, int(lb.get("size", 200)), rng, led)
        steps = []
        for m in lb.get("along_quake_mult", []):
            new = SourceLaw(replace(base, quake_mult=float(m)))
            lib, kept = lib.move_to(new, int(lb.get("size", 200)), rng, led)
            steps.append({"quake_mult": float(m), "kept": kept, "drivers_library": drivers(new, lib.samples),
                          "drivers_exact": drivers(new)})
        rep["questions"]["tail scenarios"] = {"event": lb.get("event", "P(both exhausted)"),
                                              "drivers_now": drivers(law, ScenarioLibrary.generate(law, ev, int(lb.get("size", 200)), rng, Ledger()).samples),
                                              "steps": steps, "queries": led.total()}
    rep["seconds"] = time.time() - t0
    out = Path(out_root) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    (out / "summary.md").write_text(twin_summary(rep), encoding="utf-8")
    return out


def twin_summary(rep) -> str:
    L = [f"# Digital twin: {rep['name']}", "", f"Portfolio `{rep['portfolio']}`. Every answer is certified at "
         f"{rep['confidence']:.0%} confidence (IQAE with {dict(ch='Chernoff-Hoeffding', cp='Clopper-Pearson')[rep.get('interval', 'ch')]} "
         "bounds, emulated at the exact amplitude of the "
         "verified circuit) and every quantum query is recorded. Exact values are shown for checking.", ""]
    q = rep["questions"]
    ca = q["certified answers"]
    L += ["## Certified answers", "", "| question | estimate | certified interval | exact |", "|---|---:|---:|---:|"]
    fm = lambda v: f"USD {v / 1e6:,.3f}m" if v > 1 else f"{v:.4%}"
    for t, a in ca["answers"].items():
        L.append(f"| {t} | {fm(a['estimate'])} | {fm(a['lo'])} to {fm(a['hi'])} | {fm(a['exact'])} |")
    L += ["", f"Quantum queries: {ca['queries']:,.0f}."]
    if "dependence sweep" in q:
        d = q["dependence sweep"]
        L += ["", f"## Dependence sweep: {d['label']} from {d['grid'][0]:g} to {d['grid'][-1]:g} ({len(d['grid'])} versions)", "",
              "| target | twin queries | recompute queries | fresh estimates | coverage |", "|---|---:|---:|---:|---:|"]
        for t, x in d["targets"].items():
            L.append(f"| {t} | {x['twin_queries']:,.0f} | {x['baseline_queries']:,.0f} | {x['recalibrations']} of "
                     f"{len(d['grid'])} | {x['coverage']:.0%} |")
    if "structure comparison" in q:
        s = q["structure comparison"]
        L += ["", f"## Structure comparison by {s['metric']} (earthquake / rain split, USD m)", "",
              "| A | B | certified | quantum queries | exact A | exact B | classical paired MC scenarios |",
              "|---|---|---|---:|---:|---:|---:|"]
        for p in s["pairs"]:
            L.append(f"| {p['A']} | {p['B']} | {p['certified']} | {p['queries']:,.0f} | {fm(p['exact_A'])} | "
                     f"{fm(p['exact_B'])} | {p['classical_crn_scenarios']:,.0f} |")
    if "tail scenarios" in q:
        t = q["tail scenarios"]
        L += ["", f"## What drives '{t['event']}' (shares of conditional scenarios; categories can overlap)", "",
              "| driver | share now (library) |", "|---|---:|"]
        for k, v in t["drivers_now"].items():
            L.append(f"| {k} | {v:.0%} |")
        for st in t["steps"]:
            L.append(f"\nAt m = {st['quake_mult']:g}: kept {st['kept']} cached scenarios; " + "; ".join(
                f"{k}: {st['drivers_library'][k]:.0%} (exact {st['drivers_exact'][k]:.0%})" for k in st["drivers_exact"]))
        L += ["", f"Quantum queries for the library: {t['queries']:,.0f}."]
    L += ["", f"Run time {rep['seconds']:.0f} s. Noiseless emulation; a query is not a unit of time."]
    return "\n".join(L) + "\n"
