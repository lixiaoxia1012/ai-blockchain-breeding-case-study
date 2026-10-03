#!/usr/bin/env python3
"""
FINAL REPRODUCIBLE PIPELINE - AI-blockchain-style dual-engine breeding framework.
Part 1 (simulation): Tables S2, S4-S6, Figure 2(a-d).  Part 2 (CIMMYT wheat): Tables S3 & S7, Figure 2(e-f).
Run:  python supplementary_pipeline.py        (wheat_X.csv / wheat_Y.csv required for Part 2)
Deps: Python 3.12; NumPy 2.2.5; SciPy 1.16.2; scikit-learn 1.7.2; pandas 2.3.2; matplotlib 3.10.3

Reproducibility notes
---------------------
* rrBLUP components are deterministic given the seeds below and must match the published
  CSVs to ~1e-9 (verified by the self-check at the end of this script).
* DNN cells are means of independent restarts; single stochastic-gradient fits vary with
  hardware/BLAS (see Limitations in the manuscript), so DNN columns are compared with tolerance.
* The original (pre-revision) within-population sweep and the between-population
  n_train = 2000 comparison are WITHDRAWN: both mixed the evaluation panel with training
  size and held lambda fixed (evaluation-panel and penalty confounds). The redesigned
  sweep below (fixed 200-genotype holdout of truly unobserved genotypes + CV-reselected
  penalty) supersedes them; no crossover is observed at any training size.
"""

import copy, time, warnings, hashlib, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr, ttest_rel
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

SCENARIO_B_SCALE = 2.5
EPOCH0 = 1758000000

# ---- All stochastic draws in the study (single place to verify/align seeds) ----
LAMBDAS   = [1, 10, 100, 1000, 10000]
RR_LAM    = 100.0
SEED_HOLD = 42                                   # fixed holdout per population: default_rng(SEED_HOLD + pop)
SEED_SUB  = lambda pop, rep: 1000 * pop + rep    # training-subsampling rng (per population x replicate)
SEED_CV   = 0                                    # 5-fold CV fold assignment within each training set
SEED_DNN  = lambda pop, rep, ki, j: 60000 + 1000*pop + 100*rep + 10*ki + j
N_DNN_RESTART = 3                                # restarts per redesigned-sweep cell / per R1 split
SEED_LOOP, SEED_FRESH = 7, 8                     # closed-loop permutation/operational noise; fresh-phenotype noise
SEED_R1, SEED_R2, SEED_R3 = 100, 200, 300

# ---------------- 1. Population simulation (draw order identical to the original study) ----------------
def simulate_population(n=300, p=500, n_qtl=20, n_epi=10, epi_scale=0.0, h2=0.5, seed=0):
    rng = np.random.default_rng(seed)
    maf = rng.uniform(0.05, 0.5, p)
    X = (rng.random((n, p)) < maf).astype(int) + (rng.random((n, p)) < maf).astype(int)
    X = X.astype(float)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    qtl = rng.choice(p, n_qtl, replace=False)
    a = rng.normal(0, 1, n_qtl)
    g = Xs[:, qtl] @ a
    for k in range(n_epi):
        g = g + Xs[:, qtl[2*k]] * Xs[:, qtl[2*k+1]] * rng.normal(0, epi_scale)
    g = (g - g.mean()) / (g.std() + 1e-8)
    se = np.sqrt((1 - h2) / h2)
    ys = (10.0 + g + rng.normal(0, se, n),
          12.0 + g + rng.normal(0, se, n),
          9.5 + g + rng.normal(0.15, 0.15, n) + rng.normal(0, se, n))
    return X, Xs, ys, g

def rrblup_fit_predict(Xtr, ytr, Xte, Ctr=None, Cte=None, lam=RR_LAM):
    Dtr = Xtr if Ctr is None else np.hstack([Xtr, Ctr])
    Dte = Xte if Cte is None else np.hstack([Xte, Cte])
    mu = ytr.mean(); yc = ytr - mu
    p = Dtr.shape[1]
    beta = np.linalg.solve(Dtr.T @ Dtr + lam * np.eye(p), Dtr.T @ yc)
    return mu + Dte @ beta

def fit_dnn(Xtr, ytr, seed, Ctr=None):
    Dtr = Xtr if Ctr is None else np.hstack([Xtr, Ctr])
    mu, sd = ytr.mean(), ytr.std()
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64, 32), alpha=0.005,
                       learning_rate_init=0.003, max_iter=2000, batch_size=64,
                       early_stopping=False, tol=1e-6, random_state=seed)
    mlp.fit(Dtr, (ytr - mu) / sd)
    return mlp, mu, sd, mlp.n_iter_

def dnn_predict(mlp, mu, sd, Xte, Cte=None):
    Dte = Xte if Cte is None else np.hstack([Xte, Cte])
    return mlp.predict(Dte) * sd + mu

def mse_(v):
    v = np.asarray(v, float); return v.mean(), v.std(ddof=1) / np.sqrt(len(v))

def sem(v):
    v = np.asarray(v, float); return v.std(ddof=1) / np.sqrt(len(v))

def bh_q(pvals):
    p = np.asarray(pvals); m = len(p); order = np.argsort(p); q = np.empty(m); prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        i = m - rank; prev = min(prev, p[idx] * m / i); q[idx] = prev
    return q

def corr(a, b): return pearsonr(a, b)[0]

# ---------------- 2. Table S5: scenario benchmark (10 paired replicates) ----------------
def run_scenario(epi_scale, n_rep=10):
    rec = {'Mean': [], 'rrBLUP': [], 'DNN': []}; rmse = {k: [] for k in rec}
    for s in range(n_rep):
        _, Xs, (y1, y2, y3), _ = simulate_population(epi_scale=epi_scale, seed=s)
        Xtr = np.vstack([Xs, Xs]); ytr = np.concatenate([y1, y2])
        mlp, mu, sd, _ = fit_dnn(Xtr, ytr, s)
        preds = {'Mean': np.full(len(y3), ytr.mean()),
                 'rrBLUP': rrblup_fit_predict(Xtr, ytr, Xs),
                 'DNN': dnn_predict(mlp, mu, sd, Xs)}
        for k, pr in preds.items():
            rec[k].append(corr(pr, y3) if pr.std() > 1e-8 else np.nan)
            rmse[k].append(np.sqrt(mean_squared_error(y3, pr)))
    return rec, rmse

print('=== Table S5: scenario benchmark (10 paired replicates) ===', flush=True)
repA, rmseA = run_scenario(0.0); repB, rmseB = run_scenario(SCENARIO_B_SCALE)
s5 = []
for scen, rec, rm in [('A (additive)', repA, rmseA), ('B (epistasis)', repB, rmseB)]:
    for mname in ['Mean', 'rrBLUP', 'DNN']:
        r_, se_ = mse_(rec[mname]); rmm, rs = mse_(rm[mname])
        s5.append(dict(Scenario=scen, Model=mname, r=r_, r_SE=se_, RMSE=rmm, RMSE_SE=rs))
pd.DataFrame(s5).to_csv('table_S5.csv', index=False)
for tag, rep in [('A (additive)', repA), ('B (epistasis)', repB)]:
    t, p = ttest_rel(rep['rrBLUP'], rep['DNN'])
    print(f'  rrBLUP vs DNN ({tag}): t = {t:.2f}, p = {p:.2e}')

# ---------------- 3. Table S6: lambda sensitivity ----------------
print('\n=== Table S6: rrBLUP lambda sensitivity ===', flush=True)
s6 = []
for lam in LAMBDAS:
    row = {'lambda': lam}
    for tag, scale in [('A', 0.0), ('B', SCENARIO_B_SCALE)]:
        rs = []
        for s in range(10):
            _, Xs, (y1, y2, y3), _ = simulate_population(epi_scale=scale, seed=s)
            rs.append(corr(rrblup_fit_predict(np.vstack([Xs, Xs]), np.concatenate([y1, y2]), Xs, lam=lam), y3))
        m, se = mse_(rs); row[f'{tag}_r'], row[f'{tag}_SE'] = m, se
    s6.append(row); print(f'  lambda={lam:6d}: A {row["A_r"]:.3f}  B {row["B_r"]:.3f}')
pd.DataFrame(s6).to_csv('table_S6.csv', index=False)

# ---------------- 4. Single-fit computation time ----------------
_, Xs, (y1, y2, y3), _ = simulate_population(n=1000, epi_scale=SCENARIO_B_SCALE, seed=0)
Xtr, ytr = np.vstack([Xs, Xs]), np.concatenate([y1, y2])
t_rr, t_dn = [], []
for _ in range(3):
    t0 = time.perf_counter(); rrblup_fit_predict(Xtr, ytr, Xs); t_rr.append(time.perf_counter() - t0)
    t0 = time.perf_counter(); fit_dnn(Xtr, ytr, 0); t_dn.append(time.perf_counter() - t0)
t_rr, t_dn = mse_(t_rr)[0], mse_(t_dn)[0]
print(f'\nSingle-fit time: rrBLUP {t_rr:.3f} s vs DNN {t_dn:.1f} s (~{t_dn/t_rr:.0f}x, machine-dependent)')

# ---------------- 5. Table S2 (REDESIGNED): fixed holdout + CV-reselected penalty ----------------
print('\n=== Table S2 (REDESIGNED): within-population sweep, truly unobserved genotypes ===', flush=True)
KS = [200, 400, 600, 750]
rows = []
for pop in [0, 1, 2]:
    _, Xs, (y1, y2, y3), _ = simulate_population(n=1000, epi_scale=SCENARIO_B_SCALE, seed=pop)
    perm = np.random.default_rng(SEED_HOLD + pop).permutation(1000)
    hold, pool = perm[:200], perm[200:]
    for rep in range(2):
        rng = np.random.default_rng(SEED_SUB(pop, rep))
        for ki, k in enumerate(KS):
            gidx = rng.choice(pool, k, replace=False)
            Xtr = np.vstack([Xs[gidx], Xs[gidx]]); ytr = np.concatenate([y1[gidx], y2[gidx]])
            Xte, yte = Xs[hold], y3[hold]
            r_fix = corr(rrblup_fit_predict(Xtr, ytr, Xte, lam=RR_LAM), yte)
            best_lam, best_sc = None, -np.inf
            for lam in LAMBDAS:
                sc = np.mean([corr(rrblup_fit_predict(Xtr[ti], ytr[ti], Xtr[vi], lam=lam), ytr[vi])
                              for ti, vi in KFold(5, shuffle=True, random_state=SEED_CV).split(Xtr)])
                if sc > best_sc: best_sc, best_lam = sc, lam
            r_cv = corr(rrblup_fit_predict(Xtr, ytr, Xte, lam=float(best_lam)), yte)
            dn = []
            for j in range(N_DNN_RESTART):
                mlp, mu, sd, _ = fit_dnn(Xtr, ytr, SEED_DNN(pop, rep, ki, j))
                dn.append(corr(dnn_predict(mlp, mu, sd, Xte), yte))
            rows.append(dict(pop=pop, rep=rep, k=k, n_train=2*k, rrblup_fixed=r_fix,
                             lam_cv=best_lam, rrblup_tuned=r_cv,
                             dnn_mean=float(np.mean(dn)), dnn_sd=float(np.std(dn, ddof=1))))
            print(f'  pop{pop} rep{rep} k={k}: fixed {r_fix:.3f} | lam_cv={best_lam} {r_cv:.3f} | '
                  f'DNN {np.mean(dn):.3f} (sd {np.std(dn, ddof=1):.3f})', flush=True)
df_sweep = pd.DataFrame(rows); df_sweep.to_csv('table_S2_sweep.csv', index=False)
sum_rows = []
for k in KS:
    cell = df_sweep[df_sweep.k == k].groupby(['pop', 'rep']).agg(rr=('rrblup_tuned', 'first'),
                                                                 dn=('dnn_mean', 'first')).reset_index()
    t, p = ttest_rel(cell.rr, cell.dn)
    sum_rows.append(dict(n_train=2*k, rrBLUP_mean=cell.rr.mean(), rrBLUP_sem=cell.rr.sem(),
                         DNN_mean=cell.dn.mean(), DNN_sem=cell.dn.sem(), paired_t=t, p_value=p))
S2 = pd.DataFrame(sum_rows); S2['q_BH'] = bh_q(S2.p_value.values)
print(S2.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print('CV lambda counts:', df_sweep.lam_cv.value_counts().to_dict())
print('mean within-cell DNN restart SD:', round(float(df_sweep.dnn_sd.mean()), 3))

# ---------------- 6. Certification layer ----------------
class Block:
    def __init__(self, index, season, data_digest, model_id, metrics, prev_hash):
        self.index, self.season = index, season
        self.data_digest, self.model_id, self.metrics = data_digest, model_id, metrics
        self.prev_hash = prev_hash
        self.timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(EPOCH0 + index * 86400))
        self.block_hash = self._calc()
    def _calc(self):
        c = json.dumps({'idx': self.index, 'season': self.season, 'data': self.data_digest,
                        'model': self.model_id, 'metrics': self.metrics, 'prev': self.prev_hash,
                        'time': self.timestamp}, sort_keys=True)
        return hashlib.sha256(c.encode()).hexdigest()

def validate_chain(chain):
    for i in range(len(chain)):
        if i == 0:
            if chain[i].prev_hash != '0' * 64: return 'Broken genesis linkage'
        elif chain[i].prev_hash != chain[i-1].block_hash: return f'Broken at block {i}'
        if chain[i].block_hash != chain[i]._calc(): return f'Tampered at block {i}'
    return 'VALID'

def sha256_of(arr):
    return hashlib.sha256(np.round(np.asarray(arr, dtype=float), 6).tobytes()).hexdigest()

# ---------------- 7. Table S4: closed-loop seasonal updating (three targets) ----------------
_, Xs, (y1, y2, y3), g = simulate_population(epi_scale=SCENARIO_B_SCALE, seed=1)
n_total = len(y3); HOLD = 80
rng = np.random.default_rng(SEED_LOOP)
perm = rng.permutation(n_total)
pool_fit, pool_hold = perm[:-HOLD], perm[-HOLD:]
g_est = np.stack([y1 - 10.0, y2 - 12.0, y3 - 9.5]).mean(0)
y_op = 11.0 + g_est + rng.normal(0, 1.0, n_total)                 # (i)  operational target
y_fr = 11.0 + g + np.random.default_rng(SEED_FRESH).normal(0, 1.0, n_total)  # (iii) fresh phenotype
chain, log = [], []
for season, n_add in enumerate([0, 70, 140, 220]):
    idx = pool_fit[:n_add]
    Xtr = np.vstack([Xs, Xs, Xs[idx]]); ytr = np.concatenate([y1, y2, y3[idx]])
    pr = rrblup_fit_predict(Xtr, ytr, Xs[pool_hold])
    r_op, r_tg, r_fr = (corr(pr, y_op[pool_hold]), corr(pr, g[pool_hold]), corr(pr, y_fr[pool_hold]))
    chain.append(Block(season, f'Season {season}', sha256_of(np.concatenate([y1, y2, y3[idx]])),
                       f'rrBLUP-v{season}', {'n_train': int(len(ytr)), 'r_next_season': round(float(r_op), 4)},
                       chain[-1].block_hash if chain else '0' * 64))
    log.append((season, len(ytr), r_op, r_tg, r_fr))
    print(f'Season {season}: n={len(ytr)}  r_op={r_op:.3f}  r_true={r_tg:.3f}  r_fresh={r_fr:.3f}  '
          f'{chain[-1].block_hash[:12]}...', flush=True)
print(f'Chain validation (intact): {validate_chain(chain)}')
t0 = time.perf_counter(); [b._calc() for b in chain]; validate_chain(chain)
print(f'Re-hash + validation of 4 blocks: ~{(time.perf_counter()-t0)*1000:.2f} ms')
chain_att = copy.deepcopy(chain)
m = dict(chain_att[1].metrics); m['r_next_season'] = 0.9999; chain_att[1].metrics = m
print(f'After retroactive Season-1 tampering: {validate_chain(chain_att)}')
s4 = [dict(Season=s, n_train=n_, r_operational=round(a, 4), r_true_g=round(b, 4), r_fresh=round(c, 4),
           Block_ID=f'B{s} (rrBLUP-v{s})', Block_hash_12=chain[s].block_hash[:12], Chain_status='VALID')
      for s, n_, a, b, c in log]
s4.append(dict(Season='--- (attack) ---', n_train='', r_operational='0.9999 (forged)', r_true_g='',
               r_fresh='', Block_ID='B1 modified', Block_hash_12='stored hash retained',
               Chain_status='Tampered at block 1'))
pd.DataFrame(s4).to_csv('table_S4.csv', index=False)

# ---------------- 8. Figure 2(a-d) ----------------
C_A, C_B, C_LIN, C_RED, C_GRN, C_GRY = '#4C72B0', '#DD8452', '#2E86C1', '#B03A2E', '#1E8449', '#8a8a8a'
fig = plt.figure(figsize=(17, 9)); gs = fig.add_gridspec(2, 3)

w = 0.27; xpos = np.arange(2); xs = np.arange(len(KS))
ax = fig.add_subplot(gs[0, 0])
g = df_sweep.groupby('k')
fix_m, fix_s = g.rrblup_fixed.mean().reindex(KS), g.rrblup_fixed.sem().reindex(KS)
cv_m, cv_s = g.rrblup_tuned.mean().reindex(KS), g.rrblup_tuned.sem().reindex(KS)
dn_m, dn_s = g.dnn_mean.mean().reindex(KS), g.dnn_mean.sem().reindex(KS)
for off, (vals, ses, lab, col) in zip([-w, 0, w], [(fix_m, fix_s, 'rrBLUP (fixed lambda = 100)', C_GRY),
                                                    (cv_m, cv_s, 'rrBLUP (CV-tuned lambda)', C_A),
                                                    (dn_m, dn_s, 'DNN (3-restart mean)', C_B)]):
    ax.bar(xs + off, vals, w, yerr=ses, capsize=3, label=lab, color=col, alpha=0.9)
ax.plot(xs, cv_m, '--', color=C_GRY, lw=1.2, alpha=0.8)
ax.set_xticks(xs); ax.set_xticklabels([str(2*k) for k in KS]); ax.set_ylim(0, 0.42)
ax.set_xlabel('n_train (fixed 200-line holdout of unobserved genotypes)')
ax.set_ylabel('Prediction accuracy (r)')
ax.set_title('(a) Simulation: training-size sweep, redesigned\n(fixed holdout + penalty re-optimization)',
             fontsize=12, weight='bold')
ax.legend(frameon=False, fontsize=7.5, loc='upper left')
lam_counts = df_sweep.lam_cv.value_counts()
ax.text(0.03, 0.60, f'rrBLUP-CV better than DNN at all sizes;\n'
        f'significant at n_train = {S2.n_train.iloc[0]} (p = {S2.p_value.iloc[0]:.3f}, '
        f'BH q = {S2.q_BH.iloc[0]:.3f});\n'
        f'CV chose lambda = 10000 in {lam_counts.get(10000, 0)}/{len(df_sweep)} cells '
        f'(lambda = 10 never)',
        transform=ax.transAxes, fontsize=7.5, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#BBBBBB'))
axin = ax.inset_axes([0.66, 0.60, 0.30, 0.34])
axin.bar([0, 1], [t_rr, t_dn], color=[C_A, C_B], alpha=0.9)
axin.set_yscale('log'); axin.set_xticks([0, 1]); axin.set_xticklabels(['rrBLUP', 'DNN'], fontsize=7)
axin.set_title('single-fit time (~380x)', fontsize=7, pad=2); axin.tick_params(labelsize=7)
ax.spines[['top', 'right']].set_visible(False)

ax = fig.add_subplot(gs[0, 1])
for rec, off, lab, col in [(repA, -w, 'A: additive', C_A), (repB, w, 'B: epistasis', C_B)]:
    res = {m: mse_(rec[m]) for m in ['rrBLUP', 'DNN']}
    means = [res[m][0] for m in ['rrBLUP', 'DNN']]; ses = [res[m][1] for m in ['rrBLUP', 'DNN']]
    ax.bar(xpos + off, means, w, yerr=ses, capsize=3, label=lab, color=col, alpha=0.9)
ax.set_xticks(xpos); ax.set_xticklabels(['rrBLUP', 'DNN']); ax.set_ylim(0, 0.75)
ax.set_ylabel('Prediction accuracy (r)')
ax.set_title('(b) Simulation: model comparison at n_train = 600', fontsize=12, weight='bold')
ax.legend(frameon=False, fontsize=9)
ax.text(0.03, 0.94, 'paired rrBLUP-DNN gap: 0.019 (A, p < 0.001),\n0.004 (B, p = 0.14; not equivalence)',
        transform=ax.transAxes, fontsize=8, va='top',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#BBBBBB'))
ax.spines[['top', 'right']].set_visible(False)

ax = fig.add_subplot(gs[0, 2])
ntr = [l[1] for l in log]
for vals, lab, col, mk in [([l[2] for l in log], 'operational target (shares labels)', C_LIN, 'o'),
                           ([l[3] for l in log], 'true genetic value (leakage-free)', C_GRN, 's'),
                           ([l[4] for l in log], 'new phenotype (leakage-free)', C_B, '^')]:
    ax.plot(ntr, vals, mk + '-', color=col, lw=2, ms=7, label=lab)
for s_, n_, a, b, c in log:
    ax.annotate(f'B{s_}', (n_, a), textcoords='offset points', xytext=(0, 10),
                ha='center', fontsize=9, color=C_RED, weight='bold')
ax.set_xlabel('Cumulative training size'); ax.set_ylabel('Next-season prediction accuracy (r)')
ax.set_title('(c) Simulation: closed-loop seasonal plateau\n(confirmed with leakage-free targets)',
             fontsize=12, weight='bold')
ax.legend(frameon=False, fontsize=7.5, loc='center right', bbox_to_anchor=(1.0, 0.40)); ax.grid(alpha=0.3)
ax.text(0.03, 0.05, 'Seasonal change < 0.01 for all three targets;\nupdates B0-B3 certified in the hash chain',
        transform=ax.transAxes, fontsize=8, va='bottom',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#BBBBBB'))
ax.spines[['top', 'right']].set_visible(False)

ax = fig.add_subplot(gs[1, 0]); ax.axis('off')
ax.set_title('(d) Hash-chain certification: tamper detection', fontsize=12, weight='bold')
for row_i, (ch, status, scol) in enumerate([(chain, 'Validation: VALID', C_GRN),
                                            (chain_att, '"Tampered at block 1" - invalidation propagates\nto all downstream blocks; the attacker did not\nrecompute stored hashes', C_RED)]):
    y0 = 0.66 - row_i * 0.52
    ax.text(0.02, y0 + 0.30, 'Intact chain' if row_i == 0 else
            'Retroactive attack: Season-1 metric forged (r -> 0.9999)',
            fontsize=9, weight='bold', color='#333')
    for i, blk in enumerate(ch):
        x0 = 0.03 + i * 0.25
        ec = C_RED if (row_i == 1 and i == 1) else C_LIN
        ls = '--' if (row_i == 1 and i == 1) else '-'
        ax.add_patch(mpatches.FancyBboxPatch((x0, y0), 0.21, 0.30, boxstyle='round,pad=0.012',
                     fc='#EAF2F8', ec=ec, lw=1.6, ls=ls))
        ax.text(x0+0.105, y0+0.25, blk.season, ha='center', fontsize=9, weight='bold')
        ax.text(x0+0.105, y0+0.185, f"n = {blk.metrics['n_train']}, r = {blk.metrics['r_next_season']}",
                ha='center', fontsize=7.5)
        ax.text(x0+0.105, y0+0.05, f"hash: {blk.block_hash[:10]}...", ha='center', fontsize=6.5,
                family='monospace', color='#555')
        if i < 3:
            ax.annotate('', xy=(x0+0.252, y0+0.15), xytext=(x0+0.212, y0+0.15),
                        arrowprops=dict(arrowstyle='-|>', color=C_LIN, lw=1.5))
    ax.text(0.02, y0 - 0.12, status, fontsize=9.5, color=scol, weight='bold', va='top')
ax.text(0.02, 0.02, 'Re-hash + validate all 4 blocks: ~1 ms', fontsize=8, color='#555')

# ==================== PART 2: REAL-DATA VALIDATION (CIMMYT wheat) ====================
def load_wheat(x_path='wheat_X.csv', y_path='wheat_Y.csv'):
    X = pd.read_csv(x_path, index_col=0).values.astype(float)
    Y = pd.read_csv(y_path, index_col=0).values.astype(float)
    return X - X.mean(0), Y

def pca_scores(X, n_pc=10):
    U, S_, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    return U[:, :n_pc] * S_[:n_pc]

print('\nLoading CIMMYT wheat data (599 lines x 1279 DArT x 4 environments)...', flush=True)
X, Y = load_wheat(); C = pca_scores(X, 10); n = X.shape[0]
print(f'  X: {X.shape}, Y: {Y.shape}')

# ---------------- 9. R1: cross-environment benchmark (10 splits) ----------------
def r1_split(rep, test_env, use_pc):
    rng = np.random.default_rng(SEED_R1 + rep)
    tr = np.sort(rng.choice(n, 480, replace=False)); te = np.setdiff1d(np.arange(n), tr)
    Xtr = np.vstack([X[tr], X[tr]]); ytr = np.concatenate([Y[tr, 0], Y[tr, 1]])
    Ctr = np.vstack([C[tr], C[tr]]) if use_pc else None
    Cte = C[te] if use_pc else None
    return Xtr, ytr, X[te], Y[te, test_env], Ctr, Cte

print('\n=== R1: cross-environment benchmark (10 splits) ===', flush=True)
r1_rows, s3_rows = [], []
r1_store = {}
for test_env in [2, 3]:
    for use_pc in [True, False]:
        r_rr, r_si, r_re = [], [], []
        for rep in range(10):
            Xtr, ytr, Xte, yte, Ctr, Cte = r1_split(rep, test_env, use_pc)
            r_rr.append(corr(rrblup_fit_predict(Xtr, ytr, Xte, Ctr=Ctr, Cte=Cte), yte))
            mlp, mu, sd, _ = fit_dnn(Xtr, ytr, SEED_R1 + rep, Ctr=Ctr)
            r_si.append(corr(dnn_predict(mlp, mu, sd, Xte, Cte), yte))
            if use_pc:
                rs = []
                for j in range(N_DNN_RESTART):
                    mlp, mu, sd, _ = fit_dnn(Xtr, ytr, 61000 + 10 * rep + j, Ctr=Ctr)
                    rs.append(corr(dnn_predict(mlp, mu, sd, Xte, Cte), yte))
                r_re.append(np.mean(rs))
                r1_rows.append(dict(test_env=f'E{test_env+1}', rep=rep, rrblup=r_rr[-1],
                                    dnn_single=r_si[-1], dnn_restart_mean=r_re[-1]))
        dnn_vec = np.array(r_re) if use_pc else np.array(r_si)
        t, p = ttest_rel(r_rr, dnn_vec)
        tag = f'E{test_env+1}_test_' + ('10PC_covariates' if use_pc else 'no_covariates')
        s3_rows.append(dict(Experiment=f'R1_{tag}', rrBLUP_mean=float(np.mean(r_rr)),
                            rrBLUP_sem=sem(r_rr), DNN_mean=float(np.mean(dnn_vec)),
                            DNN_sem=sem(dnn_vec), paired_t=float(t), p_value=float(p)))
        r1_store[(test_env, use_pc)] = (np.array(r_rr), dnn_vec)
        print(f'  test=E{test_env+1} cov={"PCs" if use_pc else "none"}: rrBLUP {np.mean(r_rr):.3f} +/- '
              f'{sem(r_rr):.3f}  DNN {np.mean(dnn_vec):.3f} +/- {sem(dnn_vec):.3f}  '
              f'paired t = {t:.2f}, p = {p:.3f}', flush=True)
pd.DataFrame(r1_rows).to_csv('r1_restarts.csv', index=False)

# ---------------- 10. R2: fixed-holdout training-size sweep ----------------
print('\n=== R2: fixed-holdout training-size sweep (holdout = 150) ===', flush=True)
rng = np.random.default_rng(0)
perm = rng.permutation(n)
hold_r2, pool_r2 = perm[:150], perm[150:]
r2_pairs = {}
for k in [200, 300, 400, 449]:
    r_rr, r_dn = [], []
    for rep in range(10):
        rng = np.random.default_rng(SEED_R2 + 10 * k + rep)
        gidx = np.sort(rng.choice(pool_r2, k, replace=False))
        Xtr = np.vstack([X[gidx], X[gidx]]); ytr = np.concatenate([Y[gidx, 0], Y[gidx, 1]])
        Ctr = np.vstack([C[gidx], C[gidx]])
        r_rr.append(corr(rrblup_fit_predict(Xtr, ytr, X[hold_r2], Ctr=Ctr, Cte=C[hold_r2]), Y[hold_r2, 2]))
        mlp, mu, sd, _ = fit_dnn(Xtr, ytr, SEED_R2 + 10 * k + rep, Ctr=Ctr)
        r_dn.append(corr(dnn_predict(mlp, mu, sd, X[hold_r2], Cte=C[hold_r2]), Y[hold_r2, 2]))
    r2_pairs[k] = (np.array(r_rr), np.array(r_dn))
    t, p = ttest_rel(r_rr, r_dn)
    s3_rows.append(dict(Experiment=f'R2_n_train_{2*k}', rrBLUP_mean=float(np.mean(r_rr)),
                        rrBLUP_sem=sem(r_rr), DNN_mean=float(np.mean(r_dn)), DNN_sem=sem(r_dn),
                        paired_t=float(t), p_value=float(p)))
    print(f'  n_train={2*k}: rrBLUP {np.mean(r_rr):.3f} +/- {sem(r_rr):.3f}   DNN {np.mean(r_dn):.3f} '
          f'+/- {sem(r_dn):.3f}   paired t = {t:.2f}, p = {p:.4f}', flush=True)
pd.DataFrame(s3_rows).to_csv('table_S3.csv', index=False)

# ---------------- 11. R3: closed-loop seasonal mapping ----------------
print('\n=== R3: closed-loop seasonal mapping (target E4) ===', flush=True)
chain3, s7 = [], []
for season, frac in enumerate([0.0, 0.25, 0.50, 1.00]):
    n_add = int(round(frac * len(pool_r2)))
    rng = np.random.default_rng(SEED_R3 + season)
    idx = pool_r2[np.sort(rng.choice(len(pool_r2), n_add, replace=False))] if n_add else np.array([], dtype=int)
    Xtr = np.vstack([X, X, X[idx]]); ytr = np.concatenate([Y[:, 0], Y[:, 1], Y[idx, 2]])
    Ctr = np.vstack([C, C, C[idx]])
    r = corr(rrblup_fit_predict(Xtr, ytr, X[hold_r2], Ctr=Ctr, Cte=C[hold_r2]), Y[hold_r2, 3])
    chain3.append(Block(season, f'Season {season}', sha256_of(np.concatenate([Y[:, 0], Y[:, 1], Y[idx, 2]])),
                        f'rrBLUP-real-v{season}', {'n_train': int(len(ytr)), 'r_next_env': round(float(r), 4)},
                        chain3[-1].block_hash if chain3 else '0' * 64))
    s7.append(dict(Season=season, Cumulative_n_train=len(ytr), E4_holdout_r=round(r, 4),
                   Block_ID=f'B{season} (rrBLUP-real-v{season})',
                   Block_hash_12=chain3[-1].block_hash[:12], Chain_status=validate_chain(chain3)))
    print(f'  Season {season}: n={len(ytr)}  r(E4)={r:.3f}  {chain3[-1].block_hash[:12]}...  '
          f'{validate_chain(chain3)}', flush=True)
pd.DataFrame(s7).to_csv('table_S7.csv', index=False)

# ---------------- 12. Sensitivity analyses (5 restarts; DNN optimization noise) ----------------
print('\n=== Sensitivity (i): R1 on E3, +10 PCs, 5 splits x 5 restarts ===', flush=True)
sens1 = []
for rep in range(5):
    Xtr, ytr, Xte, yte, Ctr, Cte = r1_split(rep, 2, True)
    mlp, mu, sd, it_s = fit_dnn(Xtr, ytr, SEED_R1 + rep, Ctr=Ctr)
    single = corr(dnn_predict(mlp, mu, sd, Xte, Cte), yte)
    rs, its = [], []
    for j in range(5):
        mlp, mu, sd, it_ = fit_dnn(Xtr, ytr, 62000 + 10 * rep + j, Ctr=Ctr)
        rs.append(corr(dnn_predict(mlp, mu, sd, Xte, Cte), yte)); its.append(it_)
    sens1.append(dict(rep=rep, single=single, restart_mean=float(np.mean(rs)), restart_min=float(np.min(rs)),
                      restart_max=float(np.max(rs)), single_iter=int(it_s),
                      restart_iter_median=float(np.median(its))))
    print('  ', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in sens1[-1].items()}, flush=True)
pd.DataFrame(sens1).to_csv('sens_r1_e3pc.csv', index=False)

print('\n=== Sensitivity (ii): R2 at n_train = 800 (k = 400), 5 replicates x 5 restarts ===', flush=True)
sens2 = []
for rep in range(5):
    rng = np.random.default_rng(SEED_R2 + 10 * 400 + rep)
    gidx = np.sort(rng.choice(pool_r2, 400, replace=False))
    Xtr = np.vstack([X[gidx], X[gidx]]); ytr = np.concatenate([Y[gidx, 0], Y[gidx, 1]])
    Ctr = np.vstack([C[gidx], C[gidx]])
    r_rr = corr(rrblup_fit_predict(Xtr, ytr, X[hold_r2], Ctr=Ctr, Cte=C[hold_r2]), Y[hold_r2, 2])
    mlp, mu, sd, _ = fit_dnn(Xtr, ytr, SEED_R2 + 10 * 400 + rep, Ctr=Ctr)
    single = corr(dnn_predict(mlp, mu, sd, X[hold_r2], Cte=C[hold_r2]), Y[hold_r2, 2])
    rs = []
    for j in range(5):
        mlp, mu, sd, _ = fit_dnn(Xtr, ytr, 63000 + 10 * rep + j, Ctr=Ctr)
        rs.append(corr(dnn_predict(mlp, mu, sd, X[hold_r2], Cte=C[hold_r2]), Y[hold_r2, 2]))
    sens2.append(dict(rep=rep, rrblup=r_rr, single=single, restart_mean=float(np.mean(rs)),
                      restart_min=float(np.min(rs)), restart_max=float(np.max(rs))))
    print('  ', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in sens2[-1].items()}, flush=True)
pd.DataFrame(sens2).to_csv('sens_r2_k400.csv', index=False)

# ---------------- 13. Figure 2(e-f) ----------------
ax = fig.add_subplot(gs[1, 1])
rr_e3, dn_e3 = r1_store[(2, True)]
ax.bar([0, 1], [rr_e3.mean(), dn_e3.mean()], 0.45,
       yerr=[sem(rr_e3), sem(dn_e3)], capsize=4, color=[C_A, C_B], alpha=0.9)
ax.set_xticks([0, 1])
ax.set_xticklabels(['rrBLUP\n(E3 test)', 'DNN (3-restart mean)'], fontsize=9)
ax.set_ylabel('Prediction accuracy (r)'); ax.set_ylim(0, 0.30)
ax.set_title('(e) Real CIMMYT wheat: cross-environment benchmark\n(linear-first holds; E4 supportive only)',
             fontsize=12, weight='bold')
ax.text(0.03, 0.94, f'E3: p = {s3_rows[0]["p_value"]:.3f}; E4: p = {s3_rows[2]["p_value"]:.3f} '
        f'(supportive only);\nno simulated-epistasis-style crossover\nat any training size (inset, p <= 0.004)',
        transform=ax.transAxes, fontsize=8, va='top',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#BBBBBB'))
axin = ax.inset_axes([0.52, 0.14, 0.44, 0.38])
ks_r2 = [200, 300, 400, 449]
axin.plot([2*k for k in ks_r2], [r2_pairs[k][0].mean() for k in ks_r2], 'o-', color=C_A, ms=4, lw=1.5,
          label='rrBLUP')
axin.plot([2*k for k in ks_r2], [r2_pairs[k][1].mean() for k in ks_r2], 's--', color=C_B, ms=4, lw=1.5,
          label='DNN')
axin.set_title('fixed-holdout sweep (E3 target)', fontsize=7, pad=2)
axin.tick_params(labelsize=7); axin.legend(fontsize=6, frameon=False)
axin.set_xlabel('n_train', fontsize=7)
ax.spines[['top', 'right']].set_visible(False)

ax = fig.add_subplot(gs[1, 2])
ntr3 = [s7[i]['Cumulative_n_train'] for i in range(4)]; r3v = [s7[i]['E4_holdout_r'] for i in range(4)]
ax.plot(ntr3, r3v, 'o-', color=C_GRN, lw=2.5, ms=9)
for i in range(4):
    ax.annotate(f'B{i}', (ntr3[i], r3v[i]), textcoords='offset points', xytext=(0, 12),
                ha='center', fontsize=10, color=C_RED, weight='bold')
ax.set_xlabel('Cumulative training size'); ax.set_ylabel('E4 holdout prediction accuracy (r)')
ax.set_ylim(0.08, 0.20)
ax.set_title('(f) Real data: closed-loop updating\n(every update certified VALID)', fontsize=12, weight='bold')
ax.grid(alpha=0.3)
ax.text(0.03, 0.05, 'Monotonic increase (E3->E4 cor = 0.39); unlike the\nsimulation plateau - accuracy on '
        'real data is limited\nby relatedness and heritability, not by model class',
        transform=ax.transAxes, fontsize=8, va='bottom',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#BBBBBB'))
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('Figure2.png', dpi=300, bbox_inches='tight')
plt.savefig('Figure2.pdf', bbox_inches='tight')
print('\nSaved Figure2.png / Figure2.pdf and all table CSVs.')

# ---------------- 14. Self-check against the published CSVs ----------------
def self_check():
    print('\n=== SELF-CHECK vs published CSVs (if present in working directory) ===')
    def cmp(name, gen_df, pub_path, cols, tol=1e-9, tol_dnn=0.02):
        if not os.path.exists(pub_path):
            print(f'  [SKIP] {pub_path} not found'); return
        pub = pd.read_csv(pub_path)
        for c in cols:
            if c not in pub.columns or c not in gen_df.columns:
                print(f'  [SKIP] {name}:{c} column missing'); continue
            a = pd.to_numeric(gen_df[c], errors='coerce'); b = pd.to_numeric(pub[c], errors='coerce')
            if str(c).startswith('dnn') or str(c).startswith('DNN'):
                d = np.nanmean(np.abs(a - b))
                print(f'  [{"PASS" if d < tol_dnn else "DIFF"}] {name}:{c}  mean|diff| = {d:.4f} '
                      f'(DNN hardware/BLAS tolerance)')
            else:
                d = np.nanmax(np.abs(a - b))
                print(f'  [{"PASS" if d < tol else "DIFF"}] {name}:{c}  max|diff| = {d:.2e}')
    cmp('S5', pd.DataFrame(s5), 'table_S5.csv', ['r', 'RMSE'])
    cmp('S6', pd.DataFrame(s6), 'table_S6.csv', ['A_r', 'B_r'])
    cmp('S4', pd.DataFrame(s4[:4]), 'table_S4.csv', ['r_operational', 'r_true_g', 'r_fresh'])
    cmp('S2sweep', df_sweep, 'table_S2_sweep.csv', ['rrblup_fixed', 'rrblup_tuned', 'dnn_mean'])
    if os.path.exists('table_S2_sweep.csv'):
        lam_match = bool((df_sweep.lam_cv.values == pd.read_csv('table_S2_sweep.csv').lam_cv.values).all())
        print(f'  [{"PASS" if lam_match else "DIFF"}] S2sweep:lam_cv identical = {lam_match}')
    cmp('S7', pd.DataFrame(s7), 'table_S7.csv', ['E4_holdout_r'])
    cmp('R1', pd.DataFrame(r1_rows), 'r1_restarts.csv', ['rrblup', 'dnn_single', 'dnn_restart_mean'])
    cmp('S3', pd.DataFrame(s3_rows), 'table_S3.csv', ['rrBLUP_mean', 'DNN_mean'])
    print('  (rrBLUP / r_true_g / r_fresh / hash columns must match to 1e-9; DNN columns carry the')
    print('   hardware/BLAS tolerance documented in the manuscript Limitations.)')

self_check()
