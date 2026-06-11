#!/usr/bin/env python3
"""Generate spectral_pipeline_colab.ipynb with all updates."""
import json

def code_cell(cid, source):
    return {"cell_type": "code", "execution_count": None, "id": cid,
            "metadata": {}, "outputs": [], "source": source}

def md_cell(cid, source):
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": source}

# ─────────────────────────────────────────────────────────────────────────────
CELL_1 = """\
from google.colab import drive
drive.mount('/content/drive')
print('✓ Google Drive mounted at /content/drive')

!pip install --quiet openpyxl anthropic scikit-learn
print('✓ Packages ready')
"""

# ─────────────────────────────────────────────────────────────────────────────
CELL_2 = """\
import os, textwrap, warnings
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd

try:
    import google.colab
    _IN_COLAB = True
    import matplotlib
    matplotlib.use('module://matplotlib_inline.backend_inline')
except ImportError:
    _IN_COLAB = False
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc, roc_auc_score)
from IPython.display import display, Image as IPImage

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 10, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})

PALETTE = ['#2166AC','#D6604D','#4DAC26','#7B2D8B',
           '#F4A582','#1A9850','#E08214','#762A83']

def color_for(label, labels):
    return PALETTE[labels.index(label) % len(PALETTE)]

def savefig(fig, path_png):
    if path_png == os.devnull:
        plt.close(fig); return
    fig.savefig(path_png)
    fig.savefig(path_png.replace('.png', '.svg'))
    if _IN_COLAB:
        display(IPImage(path_png))
    plt.close(fig)
    print('  ✓  ' + os.path.basename(path_png) + '  +  .svg')

def ci95(data):
    n = data.shape[0]
    se = data.std(axis=0, ddof=1) / np.sqrt(n)
    return stats.t.ppf(0.975, df=n-1) * se

def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

# ── Normalisations ────────────────────────────────────────────────────────
def normalise_peak(X):
    mx = np.max(np.abs(X), axis=1, keepdims=True); mx[mx==0] = 1
    return X / mx

def normalise_snv(X):
    mu  = X.mean(axis=1, keepdims=True)
    sig = X.std(axis=1,  keepdims=True); sig[sig==0] = 1
    return (X - mu) / sig

def normalise_msc(X):
    ref = X.mean(axis=0); out = np.zeros_like(X)
    for i, row in enumerate(X):
        slope, intercept, *_ = stats.linregress(ref, row)
        if slope == 0: slope = 1
        out[i] = (row - intercept) / slope
    return out

NORM_FNS = {'peak': normalise_peak, 'snv': normalise_snv, 'msc': normalise_msc}

_norm_cache = {}

def get_normalised(name, spectra):
    if name not in _norm_cache:
        print('  Computing ' + name.upper() + ' ...', end=' ', flush=True)
        _norm_cache[name] = NORM_FNS[name](spectra)
        print('done.')
    else:
        print('  Using cached ' + name.upper() + '.')
    return _norm_cache[name]

# ── Marker-band RSD ───────────────────────────────────────────────────────
MARKER_BANDS = {
    720:  'Lipid C-N',
    1003: 'Phe ring',
    1155: 'C-C/C-N',
    1249: 'Amide III',
    1449: 'CH2/CH3',
    1655: 'Amide I',
}

def nearest_idx(wn_arr, target):
    return int(np.argmin(np.abs(wn_arr - target)))

def compute_rsd_table(wn, spectra, classes, normalised_dict):
    cls_arr = np.asarray(classes)
    records = []
    for band, assignment in MARKER_BANDS.items():
        j = nearest_idx(wn, band)
        actual = round(float(wn[j]), 1)
        for cls in sorted(np.unique(cls_arr)):
            mask = cls_arr == cls
            raw  = spectra[mask, j]
            row  = {
                'Band_cm': actual,
                'Assignment': assignment,
                'Class': cls,
                'n': int(mask.sum()),
                'RSD_raw_pct': round(raw.std(ddof=1) / abs(raw.mean()) * 100, 2)
                               if raw.mean() != 0 else float('nan'),
            }
            for nm, Xn in normalised_dict.items():
                v = Xn[mask, j]
                key = 'RSD_' + nm.upper() + '_pct'
                row[key] = round(v.std(ddof=1) / abs(v.mean()) * 100, 2) \
                           if v.mean() != 0 else float('nan')
            records.append(row)
    return pd.DataFrame(records)

def fig_rsd_comparison(wn, spectra, classes, normalised_dict, out_dir):
    df       = compute_rsd_table(wn, spectra, classes, normalised_dict)
    rsd_cols = [c for c in df.columns if c.startswith('RSD_')]
    agg      = df.groupby('Band_cm')[rsd_cols].mean().reset_index()
    bands    = [str(b) for b in agg['Band_cm'].tolist()]
    x        = np.arange(len(bands))
    width    = 0.8 / len(rsd_cols)
    col_labels = {'RSD_raw_pct': 'Raw'}
    for nm in normalised_dict:
        col_labels['RSD_' + nm.upper() + '_pct'] = nm.upper()
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, col in enumerate(rsd_cols):
        ax.bar(x + i * width, agg[col], width=width,
               label=col_labels.get(col, col), color=PALETTE[i], alpha=0.8)
    ax.set_xticks(x + width * (len(rsd_cols) - 1) / 2)
    ax.set_xticklabels(bands, rotation=15, ha='right')
    ax.set_ylabel('Mean RSD across classes (%)')
    ax.set_title('Marker-band Spot-to-Spot RSD: Raw vs. Normalised\\n(lower = better normalisation)')
    ax.legend(title='Method', frameon=False)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, '00_rsd_comparison.png'))
    df.to_csv(os.path.join(out_dir, '00_rsd_table.csv'), index=False)
    df.to_excel(os.path.join(out_dir, '00_rsd_table.xlsx'), index=False)
    print('  ✓  00_rsd_table.csv  +  .xlsx')
    return df

# ── Effect sizes ──────────────────────────────────────────────────────────
def compute_effect_sizes(groups, wn):
    n_wn = groups[0][1].shape[1]
    if len(groups) == 2:
        Xa, Xb = groups[0][1], groups[1][1]
        na, nb = len(Xa), len(Xb)
        mu_a, mu_b = Xa.mean(0), Xb.mean(0)
        sa, sb     = Xa.std(0, ddof=1), Xb.std(0, ddof=1)
        pooled     = np.sqrt(((na-1)*sa**2 + (nb-1)*sb**2) / (na+nb-2))
        cohens_d   = np.where(pooled > 0, (mu_a - mu_b) / pooled, np.nan)
        se_diff    = np.sqrt(sa**2/na + sb**2/nb)
        df_w       = np.clip(se_diff**4 / ((sa**2/na)**2/(na-1) + (sb**2/nb)**2/(nb-1)), 1, None)
        t_crit     = stats.t.ppf(0.975, df=df_w)
        diff       = mu_a - mu_b
        return {'type': 'cohens_d', 'effect': cohens_d,
                'ci_lo': diff - t_crit*se_diff,
                'ci_hi': diff + t_crit*se_diff,
                'diff': diff}
    else:
        all_X      = np.vstack([g[1] for g in groups])
        grand_mean = all_X.mean(0)
        ss_total   = ((all_X - grand_mean)**2).sum(0)
        ss_between = np.zeros(n_wn)
        for _, X in groups:
            ss_between += len(X) * (X.mean(0) - grand_mean)**2
        eta2 = np.where(ss_total > 0, ss_between / ss_total, 0.0)
        return {'type': 'eta_squared', 'effect': eta2}

# ── Figures ───────────────────────────────────────────────────────────────
def fig_mean_spectra(wn, groups, norm_name, out):
    labels = [g[0] for g in groups]
    fig, ax = plt.subplots(figsize=(9, 4))
    for lbl, X in groups:
        m  = X.mean(axis=0); ci = ci95(X); c = color_for(lbl, labels)
        ax.plot(wn, m, color=c, lw=1.5, label=lbl)
        ax.fill_between(wn, m-ci, m+ci, color=c, alpha=0.18)
    ax.set_xlabel('Wavenumber (cm\\u207b\\u00b9)'); ax.set_ylabel(norm_name + ' Intensity')
    ax.set_title('Mean Spectra \\u00b1 95 % CI  [' + norm_name + ']')
    ax.legend(frameon=False, bbox_to_anchor=(1,1), loc='upper left')
    fig.tight_layout(); savefig(fig, out)

def fig_pca(wn, groups, out_scores, out_loadings):
    labels = [g[0] for g in groups]
    X  = np.vstack([g[1] for g in groups])
    y  = np.concatenate([[lbl]*len(mx) for lbl, mx in groups])
    Xs = StandardScaler().fit_transform(X)
    n_comp = min(5, X.shape[0]-1, X.shape[1])
    pca = PCA(n_components=n_comp); sc = pca.fit_transform(Xs)
    ev  = pca.explained_variance_ratio_ * 100
    fig, ax = plt.subplots(figsize=(7, 6))
    for lbl in labels:
        m = y == lbl
        ax.scatter(sc[m,0], sc[m,1], c=color_for(lbl,labels), s=35, alpha=0.75,
                   edgecolors='white', lw=0.3, label=lbl)
    ax.axhline(0, color='grey', lw=0.5, ls='--'); ax.axvline(0, color='grey', lw=0.5, ls='--')
    ax.set_xlabel('PC 1 ({:.1f} %)'.format(ev[0])); ax.set_ylabel('PC 2 ({:.1f} %)'.format(ev[1]))
    ax.set_title('PCA Scores'); ax.legend(frameon=False, bbox_to_anchor=(1,1), loc='upper left')
    fig.tight_layout(); savefig(fig, out_scores)
    n_show = min(3, n_comp)
    fig, axes = plt.subplots(n_show, 1, figsize=(9, 2.5*n_show), sharex=True)
    if n_show == 1: axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(wn, pca.components_[i], color=PALETTE[i], lw=1)
        ax.axhline(0, color='grey', lw=0.5, ls='--'); ax.set_ylabel('PC {} loading'.format(i+1))
    axes[-1].set_xlabel('Wavenumber (cm\\u207b\\u00b9)'); axes[0].set_title('PCA Loadings')
    fig.tight_layout(); savefig(fig, out_loadings)
    return pca, sc, ev, y

def fig_significance(wn, groups, out, p_threshold=0.05):
    labels = [g[0] for g in groups]
    if len(groups) == 2:
        _, p = stats.ttest_ind(groups[0][1], groups[1][1], axis=0, equal_var=False)
        test_label = 'Welch t-test'
    else:
        p = np.array([f_oneway(*[g[1][:,j] for g in groups]).pvalue
                      for j in range(groups[0][1].shape[1])])
        test_label = 'One-way ANOVA'
    logp = -np.log10(np.clip(p, 1e-300, 1)); sig = p < p_threshold
    eff  = compute_effect_sizes(groups, wn)
    effect = np.abs(eff['effect']) if eff['type'] == 'cohens_d' else eff['effect']
    eff_label = "|Cohen's d|" if eff['type'] == 'cohens_d' else 'η² (eta-squared)'

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1, 1]})
    for lbl, X in groups:
        axes[0].plot(wn, X.mean(0), color=color_for(lbl,labels), lw=1.2, label=lbl)
    axes[0].set_ylabel('Normalised Intensity')
    axes[0].legend(frameon=False, bbox_to_anchor=(1,1), loc='upper left')
    axes[0].set_title('Point-wise ' + test_label)

    axes[1].fill_between(wn, 0, logp, where=~sig, color='lightgrey', step='mid',
                         label='p \\u2265 ' + str(p_threshold))
    axes[1].fill_between(wn, 0, logp, where=sig,  color='#D6604D', alpha=0.85, step='mid',
                         label='p < '   + str(p_threshold))
    axes[1].axhline(-np.log10(p_threshold), color='black', lw=0.8, ls='--',
                    label='\\u03b1 = ' + str(p_threshold))
    axes[1].set_ylabel('\\u2212log\\u2081\\u2080(p)')
    axes[1].legend(frameon=False, fontsize=9, bbox_to_anchor=(1,1), loc='upper left')

    axes[2].plot(wn, effect, color='#4DAC26', lw=1.0)
    axes[2].fill_between(wn, 0, effect, color='#4DAC26', alpha=0.2)
    axes[2].set_ylabel(eff_label)
    axes[2].set_xlabel('Wavenumber (cm\\u207b\\u00b9)')

    fig.tight_layout(); savefig(fig, out)
    return p, eff

def fig_heatmap(wn, groups, out):
    labels = [g[0] for g in groups]; chunks, dividers = [], []; cursor = 0
    for lbl, X in groups:
        n_show = min(len(X), max(10, 80//len(groups))); step = max(1, len(X)//n_show)
        chunks.append(X[::step]); cursor += len(X[::step]); dividers.append(cursor)
    Xplot = np.vstack(chunks)
    fig, ax = plt.subplots(figsize=(11, max(4, len(Xplot)*0.12+1.5)))
    im = ax.imshow(Xplot, aspect='auto', cmap='RdYlBu_r',
                   extent=[wn[0], wn[-1], len(Xplot), 0])
    fig.colorbar(im, ax=ax, pad=0.02, shrink=0.75).set_label('Normalised Intensity')
    for d in dividers[:-1]: ax.axhline(d, color='white', lw=1.5, ls='--')
    patches = [mpatches.Patch(color=color_for(lbl,labels), label=lbl) for lbl in labels]
    ax.legend(handles=patches, loc='upper right', frameon=False, fontsize=9)
    ax.set_xlabel('Wavenumber (cm\\u207b\\u00b9)'); ax.set_ylabel('Sample index')
    ax.set_title('Spectral Heatmap by Class')
    fig.tight_layout(); savefig(fig, out)

def fig_violin(wn, groups, pca_sc, pca_ev, y, out):
    labels = [g[0] for g in groups]; n_pc = min(3, pca_sc.shape[1])
    fig, axes = plt.subplots(1, n_pc, figsize=(4.5*n_pc, 5))
    if n_pc == 1: axes = [axes]
    for i, ax in enumerate(axes):
        data_list = [pca_sc[y==lbl, i] for lbl in labels]
        positions = list(range(1, len(labels)+1))
        parts = ax.violinplot(data_list, positions=positions, showmedians=False, showextrema=False)
        for pc, lbl in zip(parts['bodies'], labels):
            pc.set_facecolor(color_for(lbl, labels)); pc.set_alpha(0.55)
        ax.boxplot(data_list, positions=positions, widths=0.1,
                   medianprops=dict(color='black',lw=2), whiskerprops=dict(color='grey'),
                   capprops=dict(color='grey'), flierprops=dict(marker='o',ms=3,color='grey',alpha=0.5))
        ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=20, ha='right')
        ax.set_ylabel('PC {} score ({:.1f} %)'.format(i+1, pca_ev[i]))
        if len(labels) == 2:
            _, pv = stats.ttest_ind(data_list[0], data_list[1])
            ymax  = max(d.max() for d in data_list)
            ax.annotate(sig_stars(pv), xy=(1.5, ymax*1.08), ha='center', fontsize=14)
        else:
            pv = f_oneway(*data_list).pvalue
            ax.set_title('ANOVA ' + sig_stars(pv), fontsize=10)
    fig.suptitle('PC Score Distributions', y=1.01, fontsize=12)
    fig.tight_layout(); savefig(fig, out)

def fig_lda(groups, out):
    labels = [g[0] for g in groups]
    X = np.vstack([g[1] for g in groups])
    y = np.concatenate([[i]*len(g[1]) for i, g in enumerate(groups)])
    n_comp = min(X.shape[0]-1, X.shape[1], 50)
    Xr  = PCA(n_components=n_comp).fit_transform(StandardScaler().fit_transform(X))
    lda = LinearDiscriminantAnalysis(); lda.fit(Xr, y); sc = lda.transform(Xr)
    cv  = StratifiedKFold(n_splits=min(5, min(len(g[1]) for g in groups)))
    acc = cross_val_score(lda, Xr, y, cv=cv, scoring='accuracy').mean()
    fig, ax = plt.subplots(figsize=(7, 6))
    if sc.shape[1] >= 2:
        for i, lbl in enumerate(labels):
            m = y == i
            ax.scatter(sc[m,0], sc[m,1], c=color_for(lbl,labels), s=35, alpha=0.75,
                       edgecolors='white', lw=0.3, label=lbl)
        ax.set_xlabel('LD 1'); ax.set_ylabel('LD 2')
    else:
        for i, lbl in enumerate(labels):
            m = y == i
            ax.hist(sc[m,0], bins=20, color=color_for(lbl,labels), alpha=0.6,
                    label=lbl, edgecolor='white')
        ax.set_xlabel('LD 1 score'); ax.set_ylabel('Count')
    ax.set_title('LDA  (CV accuracy: {:.1%})'.format(acc))
    ax.legend(frameon=False, bbox_to_anchor=(1,1), loc='upper left')
    fig.tight_layout(); savefig(fig, out)
    return acc

def fig_plsda(groups, out):
    labels = [g[0] for g in groups]
    X  = np.vstack([g[1] for g in groups])
    y  = np.concatenate([[i]*len(g[1]) for i, g in enumerate(groups)]).astype(float)
    Xs = StandardScaler().fit_transform(X)
    pls = PLSRegression(n_components=2, scale=False); pls.fit(Xs, y); sc = pls.transform(Xs)
    var_exp = [np.var(Xs@pls.x_weights_[:,k]) / np.sum(np.var(Xs, axis=0))*100 for k in range(2)]
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, lbl in enumerate(labels):
        m = y == i
        ax.scatter(sc[m,0], sc[m,1], c=color_for(lbl,labels), s=35, alpha=0.75,
                   edgecolors='white', lw=0.3, label=lbl)
    ax.axhline(0, color='grey', lw=0.5, ls='--'); ax.axvline(0, color='grey', lw=0.5, ls='--')
    ax.set_xlabel('LV 1 ({:.1f}% X var)'.format(var_exp[0]))
    ax.set_ylabel('LV 2 ({:.1f}% X var)'.format(var_exp[1]))
    ax.set_title('PLS-DA Scores')
    ax.legend(frameon=False, bbox_to_anchor=(1,1), loc='upper left')
    fig.tight_layout(); savefig(fig, out)

def fig_correlation(wn, groups, out):
    bin_size = max(1, len(wn)//40); idx = np.arange(0, len(wn), bin_size)
    def bin_mean(X): return np.array([X[:,i:i+bin_size].mean(axis=1) for i in idx]).T
    Xbin = np.vstack([bin_mean(g[1]) for g in groups]); corr = np.corrcoef(Xbin.T)
    tick_labels = ['{:.0f}'.format(wn[i]) for i in idx]; step = max(1, len(tick_labels)//10)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, shrink=0.8).set_label('Pearson r')
    ax.set_xticks(range(0, len(tick_labels), step))
    ax.set_xticklabels(tick_labels[::step], rotation=45, ha='right')
    ax.set_yticks(range(0, len(tick_labels), step))
    ax.set_yticklabels(tick_labels[::step])
    ax.set_title('Wavenumber Correlation Matrix')
    fig.tight_layout(); savefig(fig, out)

def fig_pairwise_diff(wn, groups, out):
    labels = [g[0] for g in groups]; pairs = list(combinations(range(len(groups)), 2))
    n = len(pairs)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.8*n), sharex=True)
    if n == 1: axes = [axes]
    for ax, (i, j) in zip(axes, pairs):
        la, lb = labels[i], labels[j]
        diff = groups[i][1].mean(axis=0) - groups[j][1].mean(axis=0)
        ax.plot(wn, diff, color='#2166AC', lw=1.2)
        ax.axhline(0, color='grey', lw=0.6, ls='--')
        ax.fill_between(wn, 0, diff, where=diff>0, color='#2166AC', alpha=0.15)
        ax.fill_between(wn, 0, diff, where=diff<0, color='#D6604D', alpha=0.15)
        ax.set_ylabel('\\u0394 (' + la + ' \\u2212 ' + lb + ')', fontsize=9)
    axes[-1].set_xlabel('Wavenumber (cm\\u207b\\u00b9)')
    axes[0].set_title('Pairwise Mean Spectral Difference')
    fig.tight_layout(); savefig(fig, out)

# ── Classifier helpers ────────────────────────────────────────────────────
def _build_Xy(groups):
    X = np.vstack([g[1] for g in groups])
    y = np.concatenate([[i]*len(g[1]) for i, g in enumerate(groups)]).astype(int)
    return X, y

def _pca_reduce(X_fit, X_apply=None, n_components=50):
    n_comp   = min(n_components, X_fit.shape[0]-1, X_fit.shape[1])
    scaler   = StandardScaler()
    pca_obj  = PCA(n_components=n_comp, random_state=42)
    Xr_fit   = pca_obj.fit_transform(scaler.fit_transform(X_fit))
    if X_apply is not None:
        return Xr_fit, pca_obj.transform(scaler.transform(X_apply)), scaler, pca_obj
    return Xr_fit, scaler, pca_obj

def _plot_cm(y_true, y_pred, display_labels, title, out_path):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(5, len(display_labels)*1.3),
                                    max(4, len(display_labels)*1.1)))
    ConfusionMatrixDisplay(cm, display_labels=display_labels).plot(
        ax=ax, colorbar=True, cmap='Blues', values_format='d')
    ax.set_title(title); fig.tight_layout(); savefig(fig, out_path)

def _plot_roc(y_true, y_prob, display_labels, title, out_path):
    n_cls = len(display_labels)
    y_bin = label_binarize(y_true, classes=list(range(n_cls)))
    fig, ax = plt.subplots(figsize=(7, 6))
    if n_cls == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        ax.plot(fpr, tpr, color=PALETTE[0], lw=2,
                label='AUC = {:.3f}'.format(auc(fpr, tpr)))
    else:
        for i, lbl in enumerate(display_labels):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], lw=1.5,
                    label='{} (AUC={:.3f})'.format(lbl, auc(fpr, tpr)))
        macro = roc_auc_score(y_bin, y_prob, multi_class='ovr', average='macro')
        ax.plot([], [], ' ', label='Macro AUC = {:.3f}'.format(macro))
    ax.plot([0,1],[0,1], 'k--', lw=0.8)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    fig.tight_layout(); savefig(fig, out_path)

def run_classifiers(groups, out_dir, test_groups=None, n_splits=5):
    labels = [g[0] for g in groups]
    n_cls  = len(labels)
    X_tr, y_tr = _build_Xy(groups)
    has_test    = test_groups is not None

    if has_test:
        X_te, y_te = _build_Xy(test_groups)
        Xr_tr, Xr_te, scaler, pca_obj = _pca_reduce(X_tr, X_te)
    else:
        Xr_tr, scaler, pca_obj = _pca_reduce(X_tr)

    cv = StratifiedKFold(n_splits=min(n_splits, min(len(g[1]) for g in groups)),
                         shuffle=True, random_state=42)

    if n_cls == 2:
        clf_list = [
            ('LDA', LinearDiscriminantAnalysis(),
             'solver=svd, no shrinkage; PCA-preprocessed (50 PCs)'),
            ('SVM', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
             'kernel=rbf, C=1.0, gamma=scale, probability=True'),
            ('Random Forest',
             RandomForestClassifier(n_estimators=500, max_features='sqrt',
                                    min_samples_leaf=1, random_state=42, n_jobs=-1),
             'n_estimators=500, max_features=sqrt, min_samples_leaf=1'),
        ]
    else:
        clf_list = [
            ('LDA', LinearDiscriminantAnalysis(),
             'solver=svd, no shrinkage; PCA-preprocessed (50 PCs)'),
            ('QDA', QuadraticDiscriminantAnalysis(reg_param=0.1),
             'reg_param=0.1 (regularises covariance for high-dim stability)'),
            ('Random Forest',
             RandomForestClassifier(n_estimators=500, max_features='sqrt',
                                    min_samples_leaf=1, random_state=42, n_jobs=-1),
             'n_estimators=500, max_features=sqrt, min_samples_leaf=1'),
        ]

    print('  PCA: {} components ({:.1f}% variance explained)'.format(
        pca_obj.n_components_, pca_obj.explained_variance_ratio_.sum()*100))

    all_results = {}
    for name, clf, param_note in clf_list:
        print('\\n  [' + name + ']')
        print('    Params : ' + param_note)
        cv_acc = cross_val_score(clf, Xr_tr, y_tr, cv=cv, scoring='accuracy')
        print('    CV ({}-fold) : {:.3f} ± {:.3f}'.format(cv.n_splits, cv_acc.mean(), cv_acc.std()))

        clf.fit(Xr_tr, y_tr)
        pfx = name.lower().replace(' ', '_')

        _plot_cm(y_tr, clf.predict(Xr_tr), labels,
                 name + ' — Training Confusion Matrix',
                 os.path.join(out_dir, 'cm_' + pfx + '_train.png'))
        _plot_roc(y_tr, clf.predict_proba(Xr_tr), labels,
                  name + ' — ROC Curve (Training)',
                  os.path.join(out_dir, 'roc_' + pfx + '_train.png'))

        res = {'cv_mean': round(float(cv_acc.mean()), 3),
               'cv_std':  round(float(cv_acc.std()), 3),
               'params':  param_note}

        if has_test:
            y_pred_te = clf.predict(Xr_te)
            y_prob_te = clf.predict_proba(Xr_te)
            test_acc  = float((y_pred_te == y_te).mean())
            print('    Test acc : {:.3f}'.format(test_acc))
            _plot_cm(y_te, y_pred_te, labels,
                     name + ' — Test Set Confusion Matrix',
                     os.path.join(out_dir, 'cm_' + pfx + '_test.png'))
            _plot_roc(y_te, y_prob_te, labels,
                      name + ' — ROC Curve (Test Set)',
                      os.path.join(out_dir, 'roc_' + pfx + '_test.png'))
            res['test_acc'] = round(test_acc, 3)

        all_results[name] = res

    return all_results

# ── Stats table ───────────────────────────────────────────────────────────
def save_stats_table(wn, groups, p_vals, eff_data, out_dir, p_threshold=0.05):
    sig_col = 'Significant_p' + str(p_threshold).replace('.', '')
    rows = []
    ns = [len(g[1]) for g in groups]
    for j, w in enumerate(wn):
        row = {'Wavenumber': round(float(w), 2)}
        for lbl, X in groups:
            row['Mean_' + lbl] = round(float(X[:,j].mean()), 6)
            row['SD_'   + lbl] = round(float(X[:,j].std(ddof=1)), 6)
        row['n_per_class'] = ' / '.join(str(n) for n in ns)
        row['p_value']     = round(float(p_vals[j]), 6)
        row[sig_col]       = bool(p_vals[j] < p_threshold)
        eff_val = float(eff_data['effect'][j])
        if eff_data['type'] == 'cohens_d':
            row['Cohens_d']        = round(eff_val, 4)
            row['CI95_lo_meandiff'] = round(float(eff_data['ci_lo'][j]), 6)
            row['CI95_hi_meandiff'] = round(float(eff_data['ci_hi'][j]), 6)
        else:
            row['eta_squared'] = round(eff_val, 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'stats_summary.csv'), index=False)
    df.to_excel(os.path.join(out_dir, 'stats_summary.xlsx'), index=False)
    print('  ✓  stats_summary.csv  +  .xlsx')
    return df

# ── Report ────────────────────────────────────────────────────────────────
def save_report(out_dir, hypothesis, selected, norm_name, groups, lda_acc,
                p_vals, wn, p_threshold=0.05, clf_results=None):
    n_sig_001 = int((p_vals < 0.001).sum())
    n_sig_01  = int((p_vals < 0.01).sum())
    n_sig     = int((p_vals < p_threshold).sum())
    pct_sig   = round(n_sig / len(p_vals) * 100, 1)
    top10     = np.argsort(p_vals)[:10]

    lines = [
        'SPECTRAL ANALYSIS REPORT', '='*60,
        'Date          : ' + datetime.now().strftime('%Y-%m-%d %H:%M'),
        'Normalisation : ' + norm_name.upper(),
        'P-threshold   : ' + str(p_threshold),
        '',
        'HYPOTHESIS', '-'*60,
        textwrap.fill(hypothesis, width=70),
        '',
        'CLASSES COMPARED  (sample sizes)', '-'*60,
    ]
    for lbl, X in groups:
        lines.append('  {:40s}  n = {}'.format(lbl, len(X)))

    lines += [
        '',
        'KEY METRICS', '-'*60,
        '  LDA CV accuracy              : {:.1%}'.format(lda_acc),
        '  Sig. wavenumbers  p < 0.001  : {} / {} ({:.1f}%)'.format(
            n_sig_001, len(wn), n_sig_001/len(wn)*100),
        '  Sig. wavenumbers  p < 0.01   : {} / {} ({:.1f}%)'.format(
            n_sig_01, len(wn), n_sig_01/len(wn)*100),
        '  Sig. wavenumbers  p < {}     : {} / {} ({:.1f}%)'.format(
            p_threshold, n_sig, len(wn), pct_sig),
        '',
        'TOP 10 MOST SIGNIFICANT WAVENUMBERS', '-'*60,
    ]
    for idx in top10:
        lines.append('  {:.1f} cm\\u207b\\u00b9   p = {:.2e}   {}'.format(
            wn[idx], p_vals[idx], sig_stars(p_vals[idx])))

    if clf_results:
        lines += ['', 'CLASSIFIER RESULTS', '-'*60]
        for cname, res in clf_results.items():
            line = '  {:20s}  CV: {:.3f} \\u00b1 {:.3f}'.format(
                cname, res['cv_mean'], res['cv_std'])
            if 'test_acc' in res:
                line += '   |   Test: {:.3f}'.format(res['test_acc'])
            lines.append(line)

    rpt = '\\n'.join(lines)
    with open(os.path.join(out_dir, 'report.txt'), 'w') as f:
        f.write(rpt)
    print('\\n' + rpt)

    return {
        'n_sig': n_sig, 'n_total': len(wn), 'pct_sig': pct_sig,
        'lda_acc': lda_acc,
        'top_wavenumbers': [(round(float(wn[i]),1), float(p_vals[i])) for i in top10],
        'clf_results': clf_results or {},
    }

# ── Figure list ───────────────────────────────────────────────────────────
ALL_FIGURES = [
    '01_mean_spectra', '02_pca_scores', '03_pca_loadings',
    '04_significance', '05_heatmap', '06_violin_pc_scores',
    '07_lda', '08_plsda', '09_correlation_heatmap', '10_pairwise_diff',
]

# ── Main runner ───────────────────────────────────────────────────────────
def run_analysis(wn, Xn, classes, selected, norm_name, out_dir, hypothesis,
                 figures_to_run=None, p_threshold=0.05, test_groups=None):
    if figures_to_run is None:
        figs = set(ALL_FIGURES)
    else:
        figs = set(figures_to_run)

    cls_arr = np.asarray(classes)
    groups  = [(lbl, Xn[cls_arr == lbl]) for lbl in selected]
    def out(name): return os.path.join(out_dir, name)

    pca_sc = pca_ev = y_arr = None
    p_vals = eff_data = lda_acc = None

    print('\\n--- Generating figures ---')

    if '01_mean_spectra' in figs:
        fig_mean_spectra(wn, groups, norm_name.upper(), out('01_mean_spectra.png'))

    need_pca = any(f in figs for f in ('02_pca_scores','03_pca_loadings','06_violin_pc_scores'))
    if need_pca:
        _, pca_sc, pca_ev, y_arr = fig_pca(
            wn, groups,
            out('02_pca_scores.png')    if '02_pca_scores'    in figs else os.devnull,
            out('03_pca_loadings.png')  if '03_pca_loadings'  in figs else os.devnull,
        )

    if '04_significance' in figs:
        p_vals, eff_data = fig_significance(wn, groups, out('04_significance.png'), p_threshold)
    else:
        eff_data = compute_effect_sizes(groups, wn)
        if len(groups) == 2:
            _, p_vals = stats.ttest_ind(groups[0][1], groups[1][1], axis=0, equal_var=False)
        else:
            p_vals = np.array([f_oneway(*[g[1][:,j] for g in groups]).pvalue
                               for j in range(groups[0][1].shape[1])])

    if '05_heatmap' in figs:
        fig_heatmap(wn, groups, out('05_heatmap.png'))

    if '06_violin_pc_scores' in figs:
        if pca_sc is None:
            _, pca_sc, pca_ev, y_arr = fig_pca(wn, groups, os.devnull, os.devnull)
        fig_violin(wn, groups, pca_sc, pca_ev, y_arr, out('06_violin_pc_scores.png'))

    if '07_lda' in figs:
        lda_acc = fig_lda(groups, out('07_lda.png'))

    if '08_plsda' in figs:
        if len(selected) == 2:
            fig_plsda(groups, out('08_plsda.png'))
        else:
            print('  ⚠  PLS-DA skipped: requires exactly 2 classes, but {} selected.'.format(len(selected)))
            print('     Use LDA / QDA / RF from the classifier section instead.')

    if '09_correlation_heatmap' in figs:
        fig_correlation(wn, groups, out('09_correlation_heatmap.png'))

    if '10_pairwise_diff' in figs:
        fig_pairwise_diff(wn, groups, out('10_pairwise_diff.png'))

    print('\\n--- Running classifiers ---')
    clf_results = run_classifiers(groups, out_dir, test_groups=test_groups)

    print('\\n--- Saving statistics ---')
    save_stats_table(wn, groups, p_vals, eff_data, out_dir, p_threshold)
    results_summary = save_report(
        out_dir, hypothesis, selected, norm_name, groups,
        lda_acc if lda_acc is not None else float('nan'),
        p_vals, wn, p_threshold, clf_results)

    print('\\n✅ All outputs saved to:\\n   ' + out_dir)
    return results_summary

print('✓ Pipeline code loaded.')
"""

# ─────────────────────────────────────────────────────────────────────────────
CELL_3 = """\
hypothesis = input('State your hypothesis: ').strip()
if not hypothesis:
    raise ValueError('Hypothesis cannot be empty.')
print('\\nRecorded: "' + hypothesis + '"')
"""

CELL_4A = """\
filepath = input('Full path to your Excel file: ').strip().strip('"')
if not os.path.isfile(filepath):
    raise FileNotFoundError('File not found: ' + filepath)
print('Loading...')
try:
    df = pd.read_excel(filepath, header=0)
except Exception as e:
    raise RuntimeError('Could not read Excel file: ' + str(e))
print('Shape: {} samples x {} columns'.format(df.shape[0], df.shape[1]))
print('\\nFirst 8 columns: ' + str(list(df.columns[:8])))
print('Last  5 columns: ' + str(list(df.columns[-5:])))
non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
print('\\nNon-numeric columns (likely class labels): ' + str(non_numeric))
"""

CELL_4B = """\
class_col = input('Name of the class/label column: ').strip()
if class_col not in df.columns:
    raise KeyError('Column "' + class_col + '" not found. Available: ' + str(list(df.columns[:20])))
spectral_cols = [c for c in df.columns
                 if c != class_col and pd.api.types.is_numeric_dtype(df[c])]
if not spectral_cols:
    raise ValueError('No numeric spectral columns found after removing "' + class_col + '".')
wn      = np.array([float(c) for c in spectral_cols])
spectra = df[spectral_cols].values.astype(float)
classes = df[class_col].astype(str)
print('\\nWavenumber range : {:.1f} - {:.1f} cm\\u207b\\u00b9  ({} points)'.format(
    wn.min(), wn.max(), len(wn)))
print('\\nClasses found:')
for cls in sorted(classes.unique()):
    print('  {:40s}  n = {}'.format(cls, int((classes == cls).sum())))
"""

CELL_5 = """\
out_norm_dir = os.path.join(os.path.dirname(filepath), 'normalised')
os.makedirs(out_norm_dir, exist_ok=True)
base = os.path.splitext(os.path.basename(filepath))[0]
spectral_col_names = [str(w) for w in wn]
normalised = {}
for name in NORM_FNS:
    Xn = get_normalised(name, spectra)
    normalised[name] = Xn
    out_path = os.path.join(out_norm_dir, base + '_' + name.upper() + '.xlsx')
    df_out = pd.DataFrame(Xn, columns=spectral_col_names)
    df_out.insert(0, 'Class', classes.values)
    df_out.to_excel(out_path, index=False)
    print('  saved -> ' + os.path.basename(out_path))

print('\\n--- Replicate RSD at marker bands ---')
rsd_dir = out_norm_dir
fig_rsd_comparison(wn, spectra, classes, normalised, rsd_dir)
print('\\n✓ Normalised files + RSD report saved to: ' + out_norm_dir)
"""

CELL_6A = """\
print('Available: peak | snv | msc')
norm_name = input('Which normalisation to use for analysis: ').strip().lower()
if norm_name not in NORM_FNS:
    raise ValueError('"' + norm_name + '" is not valid. Choose: peak, snv, msc')
Xn = normalised[norm_name]
print('Using: ' + norm_name.upper())
"""

CELL_6B = """\
available_classes = sorted(classes.unique())
print('Available classes:')
for i, c in enumerate(available_classes, 1):
    print('  [{}] {}  (n={})'.format(i, c, int((classes == c).sum())))
print('\\nEnter class names or numbers (comma-separated, min 2).')
raw = input('Classes: ').strip()
parts = [s.strip() for s in raw.split(',')]
selected = []
for p in parts:
    if p.isdigit():
        idx = int(p) - 1
        if idx < 0 or idx >= len(available_classes):
            raise IndexError('Number {} out of range (1-{})'.format(p, len(available_classes)))
        selected.append(available_classes[idx])
    else:
        if p not in available_classes:
            raise ValueError('"' + p + '" not found. Available: ' + str(available_classes))
        selected.append(p)
if len(selected) < 2:
    raise ValueError('Select at least 2 classes. Got: ' + str(selected))
print('\\nSelected: ' + str(selected))
"""

CELL_7 = """\
print('P-value threshold for significance (default: 0.05):')
p_raw = input('  Threshold [0.05]: ').strip()
if p_raw == '':
    P_THRESHOLD = 0.05
else:
    try:
        P_THRESHOLD = float(p_raw)
        if not (0 < P_THRESHOLD < 1): raise ValueError
    except ValueError:
        raise ValueError('"' + p_raw + '" is not valid. Enter a number between 0 and 1.')
print('Using p-value threshold: ' + str(P_THRESHOLD))

print('\\n--- Figure selection ---')
for i, f in enumerate(ALL_FIGURES, 1):
    print('  [{:2d}] {}'.format(i, f))
print('\\nEnter figure numbers (comma-separated), or leave blank for ALL.')
fig_raw = input('Figures [all]: ').strip()
if fig_raw == '':
    FIGURES_TO_RUN = None
    print('→ Generating all figures.')
else:
    FIGURES_TO_RUN = []
    for n in [s.strip() for s in fig_raw.split(',')]:
        if not n.isdigit() or int(n) < 1 or int(n) > len(ALL_FIGURES):
            raise ValueError('"' + n + '" is not a valid figure number (1-{}).'.format(len(ALL_FIGURES)))
        FIGURES_TO_RUN.append(ALL_FIGURES[int(n)-1])
    print('→ Generating: ' + str(FIGURES_TO_RUN))
"""

CELL_8 = """\
# Optional: provide a separate test file for held-out evaluation.
# If you skip this (press Enter), classifiers use cross-validation only.
print('Optional: path to a separate TEST Excel file (same format as training file).')
print('Leave blank to skip (cross-validation only).')
test_path = input('Test file path [skip]: ').strip().strip('"')

test_groups = None
if test_path:
    if not os.path.isfile(test_path):
        raise FileNotFoundError('Test file not found: ' + test_path)
    print('Loading test file...')
    df_test = pd.read_excel(test_path, header=0)
    spectral_cols_test = [c for c in df_test.columns if c != class_col
                          and pd.api.types.is_numeric_dtype(df_test[c])]
    spectra_test = df_test[spectral_cols_test].values.astype(float)
    classes_test = df_test[class_col].astype(str)
    Xn_test      = get_normalised(norm_name, spectra_test)
    cls_arr_test = np.asarray(classes_test)
    test_groups  = [(lbl, Xn_test[cls_arr_test == lbl]) for lbl in selected
                    if lbl in cls_arr_test]
    missing = [lbl for lbl in selected if lbl not in cls_arr_test]
    if missing:
        print('  ⚠  Classes not found in test file (skipped): ' + str(missing))
    print('  Test set:')
    for lbl, X in test_groups:
        print('    {:40s}  n = {}'.format(lbl, len(X)))
else:
    print('No test file — classifiers will use cross-validation only.')
"""

CELL_9 = """\
tag     = '_vs_'.join(s.replace(' ', '') for s in selected)
out_dir = os.path.join(os.path.dirname(filepath), 'results_' + tag + '_' + norm_name)

if os.path.isdir(out_dir) and os.listdir(out_dir):
    print('⚠  Output folder already exists:\\n   ' + out_dir)
    confirm = input('   Overwrite? (yes / no): ').strip().lower()
    if confirm not in ('yes', 'y'):
        raise RuntimeError('Run cancelled. Change class selection or normalisation to use a new folder.')
    print('   → Overwriting...')

os.makedirs(out_dir, exist_ok=True)
print('Results folder: ' + out_dir + '\\n')

results_summary = run_analysis(
    wn, Xn, classes, selected, norm_name, out_dir, hypothesis,
    figures_to_run=FIGURES_TO_RUN,
    p_threshold=P_THRESHOLD,
    test_groups=test_groups,
)
"""

CELL_10 = """\
import anthropic, getpass
from IPython.display import Markdown

api_key = os.environ.get('ANTHROPIC_API_KEY', '')
if not api_key:
    try:
        from google.colab import userdata
        api_key = userdata.get('ANTHROPIC_API_KEY')
    except Exception:
        pass
if not api_key:
    api_key = getpass.getpass('Enter your Anthropic API key: ')
if not api_key:
    raise RuntimeError('No API key. Add ANTHROPIC_API_KEY to Colab Secrets.')

client = anthropic.Anthropic(api_key=api_key)

top_wn_str = ', '.join('{} cm\\u207b\\u00b9 (p={:.2e})'.format(w, p)
                        for w, p in results_summary['top_wavenumbers'])
clf_str = '\\n'.join('  {}: CV {:.1%}'.format(k, v['cv_mean'])
                      + (' | Test {:.1%}'.format(v['test_acc']) if 'test_acc' in v else '')
                      for k, v in results_summary.get('clf_results', {}).items())

prompt = (
    "You are an expert spectroscopist and cell biologist specialising in extracellular vesicle "
    "(EV) Raman spectroscopy.\\n\\n"
    "## Experimental Hypothesis\\n" + hypothesis + "\\n\\n"
    "## Analysis Results\\n"
    "- Normalisation: " + norm_name.upper() + "\\n"
    "- Classes compared: " + ', '.join(selected) + "\\n"
    "- LDA CV accuracy: {:.1%}\\n".format(results_summary['lda_acc']) +
    "- Classifiers:\\n" + (clf_str or '  (none run)') + "\\n"
    "- Significant wavenumbers (p<{p}): {n}/{t} ({pct}%)\\n".format(
        p=P_THRESHOLD, n=results_summary['n_sig'],
        t=results_summary['n_total'], pct=results_summary['pct_sig']) +
    "- Top 10 wavenumbers: " + top_wn_str + "\\n\\n"
    "## Your Task\\n"
    "1. Interpret the spectral results relative to the hypothesis (support / partial / contradict).\\n"
    "2. Annotate the top significant wavenumbers with biomolecule assignments "
    "(proteins, lipids, nucleic acids, carbohydrates).\\n"
    "3. Search for and cite latest research (2020-2025) on EV Raman spectroscopy, "
    "B-cell EV biology, and tumour EV intercommunication.\\n"
    "4. Suggest mechanistic explanations for the spectral shifts between classes.\\n"
    "5. Propose follow-up experiments to strengthen or refute the hypothesis.\\n\\n"
    "Format as a structured scientific commentary with clear headings. "
    "Cite references with authors, journal, year, and DOI."
)

print('🔍 Searching latest research and generating interpretation...\\n')
print('─' * 70)

full_text = ''
with client.messages.stream(
    model='claude-opus-4-8',
    max_tokens=16000,
    thinking={'type': 'adaptive'},
    tools=[{'type': 'web_search_20260209', 'name': 'web_search', 'max_uses': 8}],
    messages=[{'role': 'user', 'content': prompt}],
    betas=['interleaved-thinking-2025-05-14'],
) as stream:
    for event in stream:
        if hasattr(event, 'type'):
            if event.type == 'content_block_delta' and hasattr(event.delta, 'text'):
                print(event.delta.text, end='', flush=True)
                full_text += event.delta.text
            elif event.type == 'content_block_start':
                blk = event.content_block
                if getattr(blk, 'type', '') == 'tool_use' and blk.name == 'web_search':
                    q = getattr(blk, 'input', {}).get('query', '')
                    if q:
                        print('\\n  🔎 Searching: "' + q + '"\\n', flush=True)

print('\\n' + '─' * 70)
interp_path = os.path.join(out_dir, 'ai_hypothesis_interpretation.md')
with open(interp_path, 'w') as f:
    f.write('# AI Hypothesis Interpretation\\n\\n')
    f.write('**Model:** claude-opus-4-8  \\n')
    f.write('**Date:** ' + datetime.now().strftime('%Y-%m-%d %H:%M') + '  \\n\\n---\\n\\n')
    f.write(full_text)
print('\\n✅ Saved: ' + interp_path)
"""

CELL_11 = """\
import shutil
from google.colab import files
zip_name = 'results_' + tag + '_' + norm_name
shutil.make_archive(zip_name, 'zip', out_dir)
files.download(zip_name + '.zip')
print('✓ Downloaded: ' + zip_name + '.zip')
"""

# ─────────────────────────────────────────────────────────────────────────────
cells = [
    md_cell('md-0', '# 🔬 Spectral Analysis Pipeline\n**Raman / IR — Hypothesis-driven, interactive**\n\n### Steps\n1. Mount Google Drive & install packages\n2. State your hypothesis\n3. Load your Excel file\n4. Normalise (Peak · SNV · MSC) — RSD comparison auto-generated\n5. Choose normalisation for analysis\n6. Select classes to compare (2 or more)\n7. Configure p-threshold & figure selection\n8. Optional: provide a separate test file for held-out evaluation\n9. Run full analysis (figures + classifiers + stats)\n10. AI Hypothesis Interpretation (Claude + web search)\n11. Download results ZIP\n\n---\n▶ **Run cells top to bottom.**'),
    md_cell('md-1', '## Cell 1 — Mount Google Drive & install packages'),
    code_cell('c1', CELL_1),
    md_cell('md-2', '## Cell 2 — Pipeline code (run once, do not edit)'),
    code_cell('c2', CELL_2),
    md_cell('md-3', '## Cell 3 — State your hypothesis'),
    code_cell('c3', CELL_3),
    md_cell('md-4', '## Cell 4 — Load your Excel file\n\nYour Google Drive is mounted at `/content/drive/MyDrive/`'),
    code_cell('c4a', CELL_4A),
    code_cell('c4b', CELL_4B),
    md_cell('md-5', '## Cell 5 — Normalise & RSD comparison\n\nAll three normalisation methods are computed and cached. A replicate RSD plot at key Raman marker bands is generated automatically — good normalisation should substantially reduce RSD compared to raw.'),
    code_cell('c5', CELL_5),
    md_cell('md-6', '## Cell 6 — Choose normalisation & select classes'),
    code_cell('c6a', CELL_6A),
    code_cell('c6b', CELL_6B),
    md_cell('md-7', '## Cell 7 — Configure analysis options\n\nSet your significance threshold and choose which figures to generate.'),
    code_cell('c7', CELL_7),
    md_cell('md-8', '## Cell 8 — Optional test file\n\nProvide a separate held-out test Excel file for independent classifier evaluation.\nLeave blank to use cross-validation only.'),
    code_cell('c8', CELL_8),
    md_cell('md-9', '## Cell 9 — Run full analysis\n\nGenerates all selected figures, runs classifiers (LDA/SVM/RF for 2-class; LDA/QDA/RF for 3+), saves stats table with p-values, effect sizes, and 95% CIs.'),
    code_cell('c9', CELL_9),
    md_cell('md-10', '## Cell 10 — AI Hypothesis Interpretation\n\nUses **Claude claude-opus-4-8** with live web search to relate spectral results to your hypothesis.\n\n> **Requires:** `ANTHROPIC_API_KEY` in Colab Secrets (left panel → 🔑) or entered at prompt.'),
    code_cell('c10', CELL_10),
    md_cell('md-11', '## Cell 11 — Download results ZIP'),
    code_cell('c11', CELL_11),
    md_cell('md-12', '## Cell 12 — Re-run with different settings (optional)\n\nRe-run **Cells 6 → 7 → 8 → 9 → 10 → 11** with different classes, normalisation, threshold, or figure selection.\nNo need to reload data — normalisations are cached.'),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": []}
    },
    "cells": cells,
}

out_path = '/home/user/flowprotocol/spectral_pipeline_colab.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print('Written:', out_path)
