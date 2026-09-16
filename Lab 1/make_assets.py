"""make_assets.py: every number and figure for Week 1, Data Analysis and the ML Workflow.

AI Principles and Practice · Department of Computing and Mathematics, SETU

Reproduces the printed outputs and charts in the Week 1 slides and reference booklet.
Everything is seeded, so the results match the booklet exactly.

    python -m venv .venv && source .venv/bin/activate
    pip install numpy pandas scikit-learn scipy matplotlib imbalanced-learn
    python make_assets.py          # prints every result, writes figs-week01/

Tested with Python 3.12, numpy 2.4, pandas 3.0, scikit-learn 1.8, scipy 1.17,
matplotlib 3.10, imbalanced-learn 0.14.

Change one thing at a time (the seed, the amount of skew, how many blanks to poke),
rerun, and watch how the numbers and pictures respond.
"""
import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

try:                                    # only needed for the augmentation section
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

OUT = "figs-week01"
os.makedirs(OUT, exist_ok=True)

NAVY, ACC, SKY, RUST, GREEN = "#1f3864", "#2e5aac", "#7d9ed0", "#b5462b", "#2f7d5b"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 12, "font.family": "DejaVu Sans",
    "axes.edgecolor": "#9fb0c8", "axes.linewidth": 0.9,
    "axes.grid": True, "grid.color": "#dde4ef", "grid.linewidth": 0.8,
    "axes.titlecolor": NAVY, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelcolor": "#2a3340", "xtick.color": "#5b6675", "ytick.color": "#5b6675",
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)


def P(title):
    print("\n=== " + title + " ===")


# ====================================================================
#  Running example dataset (booklet §1.3)
# ====================================================================
rng = np.random.default_rng(42)          # fixed seed: same "random" data every run
n = 600                                  # 600 pretend customers
df = pd.DataFrame(dict(
    age=rng.normal(40, 12, n).clip(18, 85).round(0),
    income=rng.lognormal(10.6, 0.45, n).round(0),          # deliberately lopsided
    tenure=rng.integers(0, 72, n),                          # months as a customer
    plan=rng.choice(["Basic", "Standard", "Premium"], n, p=[.5, .3, .2]),
    region=rng.choice(["North", "South", "East", "West"], n),
    satisfaction=rng.choice(["Low", "Medium", "High"], n, p=[.2, .5, .3]),
))
df["monthly_charge"] = (20 + 0.00035*df.income + 0.05*df.tenure
                        + rng.normal(0, 4, n)).round(2)
df["churn"] = ((df.satisfaction.eq("Low")*0.5) + (df.tenure.lt(6)*0.3)
               + rng.random(n)*0.45 > 0.6).astype(int)      # 1 = customer left
df.loc[rng.random(n) < 0.08, "income"] = np.nan             # poke holes (blanks)
df.loc[rng.random(n) < 0.05, "monthly_charge"] = np.nan
out_idx = rng.choice(n, 4, replace=False)                    # 4 absurd values
df.loc[out_idx, "income"] = float(np.nanmax(df.income) * 8)

pd.set_option("display.width", 120)
P("HEAD"); print(df.head().to_string())
P("DESCRIBE (numeric)"); print(df.describe().round(2).to_string())
P("VALUE_COUNTS plan"); print(df["plan"].value_counts().to_string())
P("MISSING per column"); print(df.isna().sum().to_string())
P("CHURN balance")
left = int(df.churn.sum())
print(f"left = {left} / {n}  ({left/n:.1%});  stayed = {n-left}")

# ====================================================================
#  Summary statistics (booklet §3)
# ====================================================================
inc = df["income"].dropna()
P("SUMMARY STATS income")
print(f"mean   = {inc.mean():,.1f}")
print(f"median = {inc.median():,.1f}")
print(f"std    = {inc.std(ddof=1):,.1f}")
q1, q3 = inc.quantile(.25), inc.quantile(.75)
print(f"IQR    = {q3-q1:,.1f}  (Q1={q1:,.0f}, Q3={q3:,.0f})")
print(f"skew   = {stats.skew(inc):.2f}")
print(f"kurt   = {stats.kurtosis(inc):.2f}  (excess kurtosis, bell curve = 0)")

# ====================================================================
#  Correlation (booklet §5)
# ====================================================================
sub = df[["income", "monthly_charge"]].dropna()
pear = stats.pearsonr(sub.income, sub.monthly_charge)
spear = stats.spearmanr(sub.income, sub.monthly_charge)
P("CORRELATION income vs monthly_charge")
print(f"Pearson  r   = {pear.statistic:.3f}  (p = {pear.pvalue:.1e})")
print(f"Spearman rho = {spear.statistic:.3f}  (p = {spear.pvalue:.1e})")

num_df = df.select_dtypes("number")      # as in corr.py: Pearson, pairwise-complete
P("CORRELATION matrix (numeric columns)")
print(num_df.corr().round(2).to_string())

# ====================================================================
#  Encoding (booklet §2)
# ====================================================================
sat_sample = df[["satisfaction"]].head(4)
oe = OrdinalEncoder(categories=[["Low", "Medium", "High"]])   # we state the order
P("ORDINAL ENCODING satisfaction (first 4)")
print(sat_sample.assign(encoded=oe.fit_transform(sat_sample).astype(int).ravel())
      .to_string(index=False))

oh = OneHotEncoder(sparse_output=False, dtype=int, handle_unknown="ignore")
oh.fit(df[["plan"]])                     # fit before asking for the column names
P("ONE-HOT ENCODING plan (first 4)")
enc = oh.transform(df[["plan"]].head(4))
print(pd.DataFrame(enc, columns=oh.get_feature_names_out()).to_string(index=False))

# ====================================================================
#  Scaling (booklet §6.2)
#  The z-scores below come from a scaler fit on these five rows only,
#  which is how the slide and booklet output tables were produced.
# ====================================================================
sample = df[["age", "monthly_charge"]].dropna().head(5).reset_index(drop=True)
scaled = pd.DataFrame(StandardScaler().fit_transform(sample),
                      columns=["age_z", "charge_z"]).round(2)
P("STANDARDISATION (first 5 rows)")
print(pd.concat([sample, scaled], axis=1).to_string(index=False))

# ====================================================================
#  Imputation (booklet §6.3)
# ====================================================================
P("IMPUTATION income")
print(f"missing before     = {df['income'].isna().sum()}")
median_fill = SimpleImputer(strategy="median").fit_transform(df[["income"]])
knn_fill = KNNImputer(n_neighbors=5).fit_transform(df[["age", "income", "tenure"]])
print(f"median used        = {np.nanmedian(df['income']):,.0f}")
print(f"missing after median fill = {int(np.isnan(median_fill).sum())}")
print(f"missing after KNN fill    = {int(np.isnan(knn_fill).sum())}")

# ====================================================================
#  Outliers, IQR rule (booklet §6.4)
# ====================================================================
lo, hi = q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1)
flagged = inc[(inc < lo) | (inc > hi)]
P("OUTLIERS income (IQR rule)")
print(f"lower fence = {lo:,.0f}")
print(f"upper fence = {hi:,.0f}")
print(f"flagged     = {len(flagged)} values; max = {flagged.max():,.0f}")

# ====================================================================
#  Leakage demonstration (booklet §7.2)
# ====================================================================
g = np.random.default_rng(0)
Xn, yn = g.normal(size=(200, 5000)), g.integers(0, 2, size=200)   # pure noise
Xsel = SelectKBest(f_classif, k=20).fit_transform(Xn, yn)        # WRONG: peeks at all data
wrong = cross_val_score(LogisticRegression(max_iter=1000), Xsel, yn, cv=5).mean()
pipe = Pipeline([("sel", SelectKBest(f_classif, k=20)),
                 ("clf", LogisticRegression(max_iter=1000))])    # RIGHT: refit per fold
right = cross_val_score(pipe, Xn, yn, cv=5).mean()
P("LEAKAGE demo (pure-noise features, true signal = none)")
print(f"leaked  CV accuracy = {wrong:.3f}")
print(f"honest  CV accuracy = {right:.3f}")
print("chance              = 0.500")

# ====================================================================
#  Augmentation with SMOTE (booklet §6.5): split first, augment train only
# ====================================================================
aug_cols = ["age", "income", "tenure", "monthly_charge"]
X_aug_src = SimpleImputer(strategy="median").fit_transform(df[aug_cols])
X_tr, X_te, y_tr, y_te = train_test_split(X_aug_src, df["churn"], test_size=0.25,
                                          random_state=0, stratify=df["churn"])
P("AUGMENTATION churn (SMOTE on the training split only)")
if SMOTE is None:
    print("skipped: imbalanced-learn is not installed "
          "(pip install imbalanced-learn), so augment.png is not written")
    X_sm = y_sm = None
else:
    X_sm, y_sm = SMOTE(random_state=0).fit_resample(X_tr, y_tr)
    before, after = dict(Counter(y_tr)), dict(Counter(y_sm))
    print(f"train before: {before}")
    print(f"train after : {after}")
    print(f"synthetic   = {after[1] - before[1]} 'left' rows")
    print(f"train rows  = {len(y_tr)} -> {len(y_sm)};  test rows untouched = {len(y_te)}")

# ====================================================================
#  FIGURES
#  (kept in this order so the seeded draws for each chart never change)
# ====================================================================
# 1) Histogram of income with mean/median
fig, ax = plt.subplots(figsize=(7, 4))
clip = inc[inc < inc.quantile(.99)]
ax.hist(clip, bins=40, color=SKY, edgecolor="white")
ax.axvline(clip.mean(), color=RUST, lw=2, label=f"mean = {clip.mean():,.0f}")
ax.axvline(clip.median(), color=NAVY, lw=2, ls="--", label=f"median = {clip.median():,.0f}")
ax.set_title("Income is right-skewed: mean is pulled above the median")
ax.set_xlabel("income"); ax.set_ylabel("count"); ax.legend()
save(fig, "hist_income")

# 2) Box plot anatomy
fig, ax = plt.subplots(figsize=(7, 2.8))
ax.boxplot(df["age"], orientation="horizontal", widths=0.5, patch_artist=True,
           boxprops=dict(facecolor="#dbe5f3", color=NAVY),
           medianprops=dict(color=RUST, lw=2),
           whiskerprops=dict(color=NAVY), capprops=dict(color=NAVY),
           flierprops=dict(marker="o", mfc=RUST, mec=RUST, ms=4, alpha=.6))
for lab, x in [("Q1", df.age.quantile(.25)), ("median", df.age.median()),
               ("Q3", df.age.quantile(.75))]:
    ax.annotate(lab, (x, 1.32), ha="center", color=NAVY, fontsize=11, fontweight="bold")
ax.set_title("Anatomy of a box plot (age)")
ax.set_xlabel("age"); ax.set_yticks([])
save(fig, "box_anatomy")

# 3) Distribution shapes
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
data = {"Left-skewed": -rng.gamma(2, 1.4, 4000),
        "Symmetric": rng.normal(0, 1, 4000),
        "Right-skewed": rng.gamma(2, 1.4, 4000)}
for ax, (t, d) in zip(axes, data.items()):
    ax.hist(d, bins=45, color=SKY, edgecolor="white")
    ax.set_title(f"{t}\nskew = {stats.skew(d):+.2f}")
    ax.set_yticks([])
save(fig, "shapes")

# 4) Same stats, different shape
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
a = rng.normal(50, 10, 5000)
mix = np.concatenate([rng.normal(35, 4, 2500), rng.normal(65, 4, 2500)])
mix = (mix - mix.mean())/mix.std()*a.std() + a.mean()       # match mean & std
for ax, d, t in zip(axes, [a, mix], ["Unimodal", "Bimodal"]):
    ax.hist(d, bins=45, color=SKY, edgecolor="white")
    ax.set_title(f"{t}: mean={d.mean():.1f}, std={d.std():.1f}")
    ax.set_yticks([]); ax.set_xlim(10, 90)
fig.suptitle("Same mean and standard deviation, completely different data",
             color=NAVY, fontweight="bold", fontsize=13, y=1.03)
save(fig, "same_stats")

# 5) Anscombe's quartet
ax_x = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
q = {
    "I":   (ax_x, [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]),
    "II":  (ax_x, [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]),
    "III": (ax_x, [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]),
    "IV":  ([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
            [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]),
}
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
P("ANSCOMBE's quartet (mean x, mean y, var y, r, fitted line)")
for ax, (k, (xx, yy)) in zip(axes.ravel(), q.items()):
    xx, yy = np.array(xx, float), np.array(yy, float)
    ax.scatter(xx, yy, color=ACC, s=42, zorder=3, edgecolor="white")
    m, b = np.polyfit(xx, yy, 1)
    xs = np.array([3, 20]); ax.plot(xs, m*xs+b, color=RUST, lw=1.8)
    r = np.corrcoef(xx, yy)[0, 1]
    print(f"set {k:>3}: {xx.mean():.2f}  {yy.mean():.2f}  {yy.var(ddof=1):.2f}  "
          f"{r:.3f}  y = {b:.2f} + {m:.3f}x")
    ax.set_title(f"Set {k}:  r = {r:.2f}", fontsize=11)
    ax.set_xlim(2, 20); ax.set_ylim(2, 14)
fig.suptitle("Anscombe's quartet: identical r, slope and means; very different data",
             color=NAVY, fontweight="bold", fontsize=12)
fig.tight_layout()
save(fig, "anscombe")

# 6) Correlation heatmap (same matrix as corr.py and the printout above)
C = num_df.corr()
fig, ax = plt.subplots(figsize=(5.6, 5))
im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(C.columns)), C.columns, rotation=35, ha="right")
ax.set_yticks(range(len(C.columns)), C.columns)
ax.grid(False)
for i in range(len(C)):
    for j in range(len(C)):
        v = C.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if abs(v) > .55 else "#1a1a1a", fontsize=10)
ax.set_title("Correlation matrix (Pearson)")
fig.colorbar(im, fraction=0.046, pad=0.04)
save(fig, "corr_heatmap")

# 7) Pearson vs Spearman
x = np.sort(rng.uniform(0, 3, 200))
y = x**3 + rng.normal(0, 1.2, 200)
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.scatter(x, y, color=ACC, s=24, alpha=.7, edgecolor="white")
pr = stats.pearsonr(x, y).statistic; sr = stats.spearmanr(x, y).statistic
ax.set_title(f"Monotonic but non-linear\nPearson r = {pr:.2f}   ·   Spearman ρ = {sr:.2f}")
ax.set_xlabel("x"); ax.set_ylabel("y = x³ + noise")
save(fig, "pearson_spearman")

# 8) Scaling: before / after
sc = df[["age", "monthly_charge"]].dropna()
z = pd.DataFrame(StandardScaler().fit_transform(sc), columns=sc.columns)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
box_style = dict(patch_artist=True, boxprops=dict(facecolor="#dbe5f3", color=NAVY),
                 medianprops=dict(color=RUST))
axes[0].boxplot([sc.age, sc.monthly_charge], tick_labels=["age", "monthly_charge"], **box_style)
axes[0].set_title("Before: raw units\nthe larger numbers dominate")
axes[1].boxplot([z.age, z.monthly_charge], tick_labels=["age", "monthly_charge"], **box_style)
axes[1].set_title("After StandardScaler\ncomparable z-scores")
fig.tight_layout()
save(fig, "scaling")

# 9) Missingness matrix
fig, ax = plt.subplots(figsize=(7, 4))
sub80 = df.head(80).isna().astype(int).T
ax.imshow(sub80.values, aspect="auto",
          cmap=matplotlib.colors.ListedColormap(["#eef1f6", NAVY]))
ax.set_yticks(range(len(sub80.index)), sub80.index)
ax.set_xlabel("row (first 80)"); ax.set_title("Missing-value pattern (navy = missing)")
save(fig, "missing")

# 10) Imputation effect
obs = df["income"].dropna()
obs = obs[obs < obs.quantile(.99)]
med = df["income"].fillna(df["income"].median())
med = med[med < med.quantile(.99)]
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(obs, bins=40, color=SKY, edgecolor="white", alpha=.85, label="observed only")
ax.hist(med, bins=40, color=RUST, alpha=.35, label="after median fill")
ax.axvline(df["income"].median(), color=RUST, ls="--", lw=2)
ax.set_title("Median imputation creates an artificial spike at the median")
ax.set_xlabel("income"); ax.set_ylabel("count"); ax.legend()
save(fig, "impute")

# 11) Outliers highlighted
fig, ax = plt.subplots(figsize=(7, 4))
clean = inc[(inc >= lo) & (inc <= hi)]
outl = inc[(inc < lo) | (inc > hi)]
jit = lambda s: rng.normal(0, .04, len(s))
ax.scatter(clean, 1+jit(clean), color=ACC, s=22, alpha=.5, label="within fences")
ax.scatter(outl, 1+jit(outl), color=RUST, s=60, zorder=3, label=f"outliers (n={len(outl)})")
ax.axvline(hi, color=NAVY, ls="--", lw=1.4)
ax.text(hi, 0.02, f"  upper fence = {hi:,.0f}", color=NAVY, fontsize=10,
        transform=ax.get_xaxis_transform(), va="bottom")
ax.set_yticks([]); ax.set_title("IQR rule flags extreme incomes")
ax.set_xlabel("income"); ax.legend()
save(fig, "outliers")

# 12) Leakage bar chart
fig, ax = plt.subplots(figsize=(6.4, 4))
bars = ax.bar(["Leaked\n(select on all data)", "Honest\n(select in pipeline)", "Chance"],
              [wrong, right, 0.5], color=[RUST, GREEN, "#9fb0c8"], edgecolor="white")
ax.bar_label(bars, fmt="%.2f", padding=3, color=NAVY, fontweight="bold")
ax.set_ylim(0, 1); ax.set_ylabel("5-fold CV accuracy")
ax.set_title("Data leakage manufactures accuracy from pure noise")
save(fig, "leakage")

# 13) The ML workflow: eight stages and the feedback loop
stages = ["Frame", "Collect", "EDA", "Preprocess", "Model", "Evaluate", "Deploy", "Monitor"]
W, STEP = 1.3, 1.5
fig, ax = plt.subplots(figsize=(12, 2.6)); ax.axis("off")
for i, s in enumerate(stages):
    x0 = i*STEP
    ax.add_patch(plt.Rectangle((x0, 0), W, 1, fc="#eef1f6", ec=NAVY, lw=1.4))
    ax.text(x0 + W/2, .5, f"{i+1}\n{s}", ha="center", va="center",
            color=NAVY, fontsize=10, fontweight="bold")
    if i < len(stages) - 1:
        ax.annotate("", (x0 + STEP - 0.02, .5), (x0 + W + 0.02, .5),
                    arrowprops=dict(arrowstyle="->", color=ACC, lw=1.6))
first_mid, last_mid = W/2, (len(stages)-1)*STEP + W/2
ax.annotate("", xy=(first_mid, -0.05), xytext=(last_mid, -0.05),
            arrowprops=dict(arrowstyle="-|>", color=RUST, lw=1.4, ls="--",
                            shrinkA=0, shrinkB=0,
                            connectionstyle="arc3,rad=-0.12"))
ax.text((first_mid + last_mid)/2, -1.05, "iterate: monitoring and evaluation feed back into framing",
        ha="center", va="center", color=RUST, fontsize=10)
ax.set_xlim(-.2, len(stages)*STEP); ax.set_ylim(-1.3, 1.1)
save(fig, "workflow")

# 14) SMOTE before / after (2-D PCA view of the four scaled numeric features)
if X_sm is not None:
    view_scaler = StandardScaler().fit(X_tr)                 # fit on training rows only
    pca = PCA(n_components=2, random_state=0).fit(view_scaler.transform(X_tr))
    Z_tr = pca.transform(view_scaler.transform(X_tr))
    Z_sm = pca.transform(view_scaler.transform(X_sm))
    lo_v, hi_v = np.percentile(Z_tr, [1, 99], axis=0)        # keep the bulk readable
    pad = 0.15*(hi_v - lo_v)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    y_tr_arr, y_sm_arr = np.asarray(y_tr), np.asarray(y_sm)
    panels = [(axes[0], Z_tr, y_tr_arr, "Before: training rows as collected"),
              (axes[1], Z_sm, y_sm_arr, "After SMOTE: minority synthesised")]
    for ax, Z, yy, title in panels:
        ax.scatter(*Z[yy == 0].T, s=14, color=ACC, alpha=.55, label="stayed (0)")
        ax.scatter(*Z[yy == 1].T, s=14, color=RUST, alpha=.8, label="left (1)")
        ax.set_title(title, fontsize=12)
        ax.text(0.02, 0.03, f"left: {int((yy == 1).sum())} / {len(yy)}",
                transform=ax.transAxes, color=RUST, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("PCA component 1")
    axes[0].set_ylabel("PCA component 2")
    axes[0].set_xlim(lo_v[0] - pad[0], hi_v[0] + pad[0])
    axes[0].set_ylim(lo_v[1] - pad[1], hi_v[1] + pad[1])
    axes[0].legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    save(fig, "augment")

print(f"\n\nALL FIGURES WRITTEN to {OUT}/:", sorted(os.listdir(OUT)))
