#!/usr/bin/env python3
"""
Análisis Especial: Año 2022 - VIX vs Forward Points
Enfoque: Comportamiento único del mercado en 2022
Autor: Claude
Fecha: 2025-11-21
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("ANÁLISIS ESPECIAL: AÑO 2022 - VIX vs FORWARD POINTS")
print("="*80)

# Cargar datos
df = pd.read_csv('VIX_combined_mediana.csv')
df['dia'] = pd.to_datetime(df['dia'])
df['year'] = df['dia'].dt.year
df['month'] = df['dia'].dt.month
df['quarter'] = df['dia'].dt.quarter

fwd_cols = ['PnL_fwd_pts_01', 'PnL_fwd_pts_05', 'PnL_fwd_pts_25', 'PnL_fwd_pts_50', 'PnL_fwd_pts_90']
fwd_labels = ['FWD_01', 'FWD_05', 'FWD_25', 'FWD_50', 'FWD_90']

# Separar 2022 vs resto
df_2022 = df[df['year'] == 2022].copy()
df_others = df[df['year'] != 2022].copy()

print(f"\n[DATOS CARGADOS]")
print(f"   Total registros: {len(df):,}")
print(f"   Año 2022: {len(df_2022):,} registros ({len(df_2022)/len(df)*100:.1f}%)")
print(f"   Otros años: {len(df_others):,} registros")
print(f"   Años disponibles: {sorted(df['year'].unique())}")

# ============================================================================
# SECCIÓN 1: COMPARATIVA 2022 vs RESTO
# ============================================================================
print("\n" + "="*80)
print("SECCIÓN 1: COMPARATIVA 2022 vs RESTO DE AÑOS")
print("="*80)

print("\n--- ESTADÍSTICAS VIX ---")
print(f"\n2022:")
print(f"   VIX promedio: {df_2022['VIX'].mean():.2f}")
print(f"   VIX mediana: {df_2022['VIX'].median():.2f}")
print(f"   VIX min/max: [{df_2022['VIX'].min():.2f}, {df_2022['VIX'].max():.2f}]")
print(f"   VIX std: {df_2022['VIX'].std():.2f}")

print(f"\nOTROS AÑOS:")
print(f"   VIX promedio: {df_others['VIX'].mean():.2f}")
print(f"   VIX mediana: {df_others['VIX'].median():.2f}")
print(f"   VIX min/max: [{df_others['VIX'].min():.2f}, {df_others['VIX'].max():.2f}]")
print(f"   VIX std: {df_others['VIX'].std():.2f}")

# Test estadístico
t_stat, p_value = stats.ttest_ind(df_2022['VIX'].dropna(), df_others['VIX'].dropna())
print(f"\nTest t-student: t={t_stat:.3f}, p-value={p_value:.6f}")
print(f"   {'⚠️  VIX significativamente diferente en 2022' if p_value < 0.05 else '✓ VIX similar entre períodos'}")

print("\n--- PnL FORWARD POINTS ---")
comparison_table = []
for col, label in zip(fwd_cols, fwd_labels):
    mean_2022 = df_2022[col].mean()
    mean_others = df_others[col].mean()
    median_2022 = df_2022[col].median()
    median_others = df_others[col].median()
    wr_2022 = (df_2022[col] > 0).sum() / len(df_2022[col].dropna()) * 100
    wr_others = (df_others[col] > 0).sum() / len(df_others[col].dropna()) * 100

    diff_mean = mean_2022 - mean_others
    diff_pct = (diff_mean / mean_others * 100) if mean_others != 0 else 0

    comparison_table.append({
        'Forward': label,
        '2022_mean': mean_2022,
        'Others_mean': mean_others,
        'Diff': diff_mean,
        'Diff_%': diff_pct,
        '2022_WR': wr_2022,
        'Others_WR': wr_others
    })

    print(f"\n{label}:")
    print(f"   2022:   μ={mean_2022:+8.2f} | Med={median_2022:+8.2f} | WR={wr_2022:5.1f}%")
    print(f"   Otros:  μ={mean_others:+8.2f} | Med={median_others:+8.2f} | WR={wr_others:5.1f}%")
    print(f"   Δ:      {diff_mean:+8.2f} ({diff_pct:+.1f}%)")

df_comparison = pd.DataFrame(comparison_table)

# ============================================================================
# SECCIÓN 2: CORRELACIONES 2022 vs RESTO
# ============================================================================
print("\n" + "="*80)
print("SECCIÓN 2: CORRELACIONES VIX vs FWD PTS - 2022 vs RESTO")
print("="*80)

print("\n2022:")
corr_2022 = {}
for col, label in zip(fwd_cols, fwd_labels):
    valid = df_2022[['VIX', col]].dropna()
    if len(valid) > 10:
        corr_p = valid['VIX'].corr(valid[col])
        corr_s = valid['VIX'].corr(valid[col], method='spearman')
        corr_2022[label] = {'pearson': corr_p, 'spearman': corr_s}
        print(f"   {label}: Pearson={corr_p:+.4f} | Spearman={corr_s:+.4f}")

print("\nOTROS AÑOS:")
corr_others = {}
for col, label in zip(fwd_cols, fwd_labels):
    valid = df_others[['VIX', col]].dropna()
    if len(valid) > 10:
        corr_p = valid['VIX'].corr(valid[col])
        corr_s = valid['VIX'].corr(valid[col], method='spearman')
        corr_others[label] = {'pearson': corr_p, 'spearman': corr_s}
        print(f"   {label}: Pearson={corr_p:+.4f} | Spearman={corr_s:+.4f}")

print("\nCAMBIO EN CORRELACIÓN (2022 vs Otros):")
for label in fwd_labels:
    if label in corr_2022 and label in corr_others:
        diff = corr_2022[label]['pearson'] - corr_others[label]['pearson']
        print(f"   {label}: Δ Pearson = {diff:+.4f}")

# ============================================================================
# SECCIÓN 3: DISTRIBUCIÓN DE VIX EN 2022
# ============================================================================
print("\n" + "="*80)
print("SECCIÓN 3: DISTRIBUCIÓN DE VIX EN 2022")
print("="*80)

vix_ranges = [
    (0, 15, 'MUY BAJO (0-15)'),
    (15, 20, 'BAJO (15-20)'),
    (20, 30, 'MEDIO (20-30)'),
    (30, 40, 'ALTO (30-40)'),
    (40, 60, 'MUY ALTO (40-60)'),
    (60, 200, 'EXTREMO (>60)')
]

print("\n2022:")
for vmin, vmax, label in vix_ranges:
    mask_2022 = (df_2022['VIX'] >= vmin) & (df_2022['VIX'] < vmax)
    count_2022 = mask_2022.sum()
    pct_2022 = count_2022 / len(df_2022) * 100 if len(df_2022) > 0 else 0

    mask_others = (df_others['VIX'] >= vmin) & (df_others['VIX'] < vmax)
    count_others = mask_others.sum()
    pct_others = count_others / len(df_others) * 100 if len(df_others) > 0 else 0

    print(f"   {label}: {count_2022:4d} ({pct_2022:5.1f}%) | Otros: {count_others:5d} ({pct_others:5.1f}%)")

# ============================================================================
# SECCIÓN 4: ANÁLISIS MENSUAL 2022
# ============================================================================
print("\n" + "="*80)
print("SECCIÓN 4: ANÁLISIS MENSUAL 2022")
print("="*80)

monthly_stats = []
for month in range(1, 13):
    df_month = df_2022[df_2022['month'] == month]

    if len(df_month) > 0:
        month_name = ['', 'Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                      'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'][month]

        print(f"\n{month_name} 2022 ({len(df_month)} trades):")
        print(f"   VIX: μ={df_month['VIX'].mean():.2f} | Med={df_month['VIX'].median():.2f}")

        stats_row = {
            'Mes': month_name,
            'Month_num': month,
            'Trades': len(df_month),
            'VIX_mean': df_month['VIX'].mean(),
            'VIX_median': df_month['VIX'].median()
        }

        for col, label in zip(fwd_cols, fwd_labels):
            if col in df_month.columns:
                mean_pnl = df_month[col].mean()
                wr = (df_month[col] > 0).sum() / len(df_month[col].dropna()) * 100 if len(df_month[col].dropna()) > 0 else 0
                stats_row[f'{label}_mean'] = mean_pnl
                stats_row[f'{label}_WR'] = wr
                print(f"   {label}: μ={mean_pnl:+8.2f} | WR={wr:5.1f}%")

        monthly_stats.append(stats_row)

df_monthly = pd.DataFrame(monthly_stats)

# ============================================================================
# SECCIÓN 5: ANÁLISIS POR TRIMESTRE 2022
# ============================================================================
print("\n" + "="*80)
print("SECCIÓN 5: ANÁLISIS POR TRIMESTRE 2022")
print("="*80)

quarterly_stats = []
for q in range(1, 5):
    df_q = df_2022[df_2022['quarter'] == q]

    if len(df_q) > 0:
        print(f"\nQ{q} 2022 ({len(df_q)} trades):")
        print(f"   VIX: μ={df_q['VIX'].mean():.2f} | σ={df_q['VIX'].std():.2f}")

        q_row = {
            'Quarter': f'Q{q}',
            'Trades': len(df_q),
            'VIX_mean': df_q['VIX'].mean(),
            'VIX_std': df_q['VIX'].std()
        }

        for col, label in zip(fwd_cols, fwd_labels):
            if col in df_q.columns:
                mean_pnl = df_q[col].mean()
                sharpe = mean_pnl / df_q[col].std() if df_q[col].std() > 0 else 0
                wr = (df_q[col] > 0).sum() / len(df_q[col].dropna()) * 100 if len(df_q[col].dropna()) > 0 else 0
                q_row[f'{label}_mean'] = mean_pnl
                q_row[f'{label}_sharpe'] = sharpe
                q_row[f'{label}_WR'] = wr
                print(f"   {label}: μ={mean_pnl:+8.2f} | Sharpe={sharpe:+.3f} | WR={wr:5.1f}%")

        quarterly_stats.append(q_row)

df_quarterly = pd.DataFrame(quarterly_stats)

# ============================================================================
# SECCIÓN 6: RANGOS DE VIX EN 2022 - PERFORMANCE DETALLADA
# ============================================================================
print("\n" + "="*80)
print("SECCIÓN 6: PERFORMANCE POR RANGO VIX EN 2022")
print("="*80)

range_2022_stats = []
for vmin, vmax, label in vix_ranges:
    mask = (df_2022['VIX'] >= vmin) & (df_2022['VIX'] < vmax)
    subset = df_2022[mask]

    if len(subset) > 5:
        print(f"\n{label} ({len(subset)} trades):")

        stats_row = {
            'Rango': label,
            'Count': len(subset),
            'VIX_mean': subset['VIX'].mean()
        }

        for col, label_fwd in zip(fwd_cols, fwd_labels):
            if col in subset.columns:
                mean_pnl = subset[col].mean()
                median_pnl = subset[col].median()
                wr = (subset[col] > 0).sum() / len(subset[col].dropna()) * 100 if len(subset[col].dropna()) > 0 else 0

                stats_row[f'{label_fwd}_mean'] = mean_pnl
                stats_row[f'{label_fwd}_median'] = median_pnl
                stats_row[f'{label_fwd}_WR'] = wr

                print(f"   {label_fwd}: μ={mean_pnl:+8.2f} | Med={median_pnl:+8.2f} | WR={wr:5.1f}%")

        range_2022_stats.append(stats_row)

df_range_2022 = pd.DataFrame(range_2022_stats)

# ============================================================================
# SECCIÓN 7: EVENTOS ESPECIALES 2022
# ============================================================================
print("\n" + "="*80)
print("SECCIÓN 7: ANÁLISIS DE EVENTOS ESPECIALES 2022")
print("="*80)

# VIX alto (>25) en 2022
high_vix_2022 = df_2022[df_2022['VIX'] > 25]
print(f"\nDías con VIX > 25 en 2022: {len(high_vix_2022)} ({len(high_vix_2022)/len(df_2022)*100:.1f}%)")
if len(high_vix_2022) > 0:
    print("PnL promedio en alta volatilidad:")
    for col, label in zip(fwd_cols, fwd_labels):
        if col in high_vix_2022.columns:
            mean_pnl = high_vix_2022[col].mean()
            wr = (high_vix_2022[col] > 0).sum() / len(high_vix_2022[col].dropna()) * 100 if len(high_vix_2022[col].dropna()) > 0 else 0
            print(f"   {label}: μ={mean_pnl:+.2f} | WR={wr:.1f}%")

# VIX bajo (<20) en 2022
low_vix_2022 = df_2022[df_2022['VIX'] < 20]
print(f"\nDías con VIX < 20 en 2022: {len(low_vix_2022)} ({len(low_vix_2022)/len(df_2022)*100:.1f}%)")
if len(low_vix_2022) > 0:
    print("PnL promedio en baja volatilidad:")
    for col, label in zip(fwd_cols, fwd_labels):
        if col in low_vix_2022.columns:
            mean_pnl = low_vix_2022[col].mean()
            wr = (low_vix_2022[col] > 0).sum() / len(low_vix_2022[col].dropna()) * 100 if len(low_vix_2022[col].dropna()) > 0 else 0
            print(f"   {label}: μ={mean_pnl:+.2f} | WR={wr:.1f}%")

# Spikes de VIX en 2022
df_2022['VIX_change'] = df_2022['VIX'].diff()
spikes_2022 = df_2022[df_2022['VIX_change'].abs() > 5]
print(f"\nSpikes VIX (|Δ|>5) en 2022: {len(spikes_2022)}")

# ============================================================================
# SECCIÓN 8: VISUALIZACIONES
# ============================================================================
print("\n" + "="*80)
print("SECCIÓN 8: GENERANDO VISUALIZACIONES")
print("="*80)

fig = plt.figure(figsize=(24, 20))

# 1. Comparación VIX: 2022 vs Otros
ax1 = plt.subplot(4, 3, 1)
data_plot = [df_2022['VIX'].dropna(), df_others['VIX'].dropna()]
bp = ax1.boxplot(data_plot, labels=['2022', 'Otros años'], patch_artist=True)
bp['boxes'][0].set_facecolor('red')
bp['boxes'][1].set_facecolor('green')
for patch in bp['boxes']:
    patch.set_alpha(0.6)
ax1.set_title('Distribución VIX: 2022 vs Otros Años', fontsize=12, fontweight='bold')
ax1.set_ylabel('VIX')
ax1.grid(axis='y', alpha=0.3)

# 2. Comparación PnL FWD_25
ax2 = plt.subplot(4, 3, 2)
x = np.arange(len(fwd_labels))
width = 0.35
bars1 = ax2.bar(x - width/2, df_comparison['2022_mean'], width, label='2022', alpha=0.8, color='red')
bars2 = ax2.bar(x + width/2, df_comparison['Others_mean'], width, label='Otros', alpha=0.8, color='green')
ax2.set_xticks(x)
ax2.set_xticklabels(fwd_labels)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_title('PnL Promedio: 2022 vs Otros Años', fontsize=12, fontweight='bold')
ax2.set_ylabel('PnL Promedio')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. Win Rates comparación
ax3 = plt.subplot(4, 3, 3)
bars1 = ax3.bar(x - width/2, df_comparison['2022_WR'], width, label='2022', alpha=0.8, color='red')
bars2 = ax3.bar(x + width/2, df_comparison['Others_WR'], width, label='Otros', alpha=0.8, color='green')
ax3.set_xticks(x)
ax3.set_xticklabels(fwd_labels)
ax3.axhline(y=50, color='black', linestyle='--', linewidth=1)
ax3.set_title('Win Rate: 2022 vs Otros Años', fontsize=12, fontweight='bold')
ax3.set_ylabel('Win Rate (%)')
ax3.set_ylim(0, 100)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 4. Serie temporal VIX 2022
ax4 = plt.subplot(4, 3, 4)
if len(df_2022) > 0:
    ax4.plot(df_2022['dia'], df_2022['VIX'], color='red', linewidth=1.5, alpha=0.7)
    ax4.axhline(y=20, color='orange', linestyle='--', linewidth=1, label='VIX=20')
    ax4.axhline(y=30, color='red', linestyle='--', linewidth=1, label='VIX=30')
    ax4.fill_between(df_2022['dia'], df_2022['VIX'], alpha=0.3, color='red')
    ax4.set_title('Evolución VIX en 2022', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Fecha')
    ax4.set_ylabel('VIX')
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)

# 5. PnL mensual 2022 (FWD_25)
ax5 = plt.subplot(4, 3, 5)
if len(df_monthly) > 0 and 'FWD_25_mean' in df_monthly.columns:
    colors_monthly = ['green' if x > 0 else 'red' for x in df_monthly['FWD_25_mean']]
    bars = ax5.bar(df_monthly['Mes'], df_monthly['FWD_25_mean'], color=colors_monthly, alpha=0.7)
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax5.set_title('PnL FWD_25 Mensual 2022', fontsize=12, fontweight='bold')
    ax5.set_ylabel('PnL Promedio')
    ax5.grid(axis='y', alpha=0.3)
    ax5.tick_params(axis='x', rotation=45)

# 6. VIX mensual 2022
ax6 = plt.subplot(4, 3, 6)
if len(df_monthly) > 0:
    ax6.plot(df_monthly['Mes'], df_monthly['VIX_mean'], marker='o', linewidth=2,
             markersize=8, color='red', label='VIX Promedio')
    ax6.fill_between(range(len(df_monthly)), df_monthly['VIX_mean'], alpha=0.3, color='red')
    ax6.set_title('VIX Promedio Mensual 2022', fontsize=12, fontweight='bold')
    ax6.set_ylabel('VIX')
    ax6.legend()
    ax6.grid(alpha=0.3)
    ax6.tick_params(axis='x', rotation=45)

# 7. Correlaciones: 2022 vs Otros
ax7 = plt.subplot(4, 3, 7)
# Usar solo las etiquetas que existen en ambos diccionarios
common_labels = [l for l in fwd_labels if l in corr_2022 and l in corr_others]
corr_2022_vals = [corr_2022[l]['pearson'] for l in common_labels]
corr_others_vals = [corr_others[l]['pearson'] for l in common_labels]
x_corr = np.arange(len(common_labels))
bars1 = ax7.bar(x_corr - width/2, corr_2022_vals, width, label='2022', alpha=0.8, color='red')
bars2 = ax7.bar(x_corr + width/2, corr_others_vals, width, label='Otros', alpha=0.8, color='green')
ax7.set_xticks(x_corr)
ax7.set_xticklabels(common_labels)
ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax7.set_title('Correlación VIX-PnL: 2022 vs Otros', fontsize=12, fontweight='bold')
ax7.set_ylabel('Correlación Pearson')
ax7.legend()
ax7.grid(axis='y', alpha=0.3)

# 8. Scatter VIX vs PnL_25 en 2022
ax8 = plt.subplot(4, 3, 8)
valid_2022 = df_2022[['VIX', 'PnL_fwd_pts_25']].dropna()
if len(valid_2022) > 0:
    scatter = ax8.scatter(valid_2022['VIX'], valid_2022['PnL_fwd_pts_25'],
                          alpha=0.5, s=30, c=valid_2022['VIX'], cmap='Reds')
    z = np.polyfit(valid_2022['VIX'], valid_2022['PnL_fwd_pts_25'], 1)
    p = np.poly1d(z)
    ax8.plot(valid_2022['VIX'].sort_values(), p(valid_2022['VIX'].sort_values()),
             "b--", alpha=0.8, linewidth=2, label=f'Tendencia: y={z[0]:.2f}x+{z[1]:.2f}')
    ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax8.set_title('VIX vs PnL_25 en 2022', fontsize=12, fontweight='bold')
    ax8.set_xlabel('VIX')
    ax8.set_ylabel('PnL_fwd_pts_25')
    ax8.legend()
    plt.colorbar(scatter, ax=ax8, label='VIX')
    ax8.grid(alpha=0.3)

# 9. Distribución rangos VIX: 2022 vs Otros
ax9 = plt.subplot(4, 3, 9)
range_labels = []
pct_2022_list = []
pct_others_list = []
for vmin, vmax, label in vix_ranges:
    mask_2022 = (df_2022['VIX'] >= vmin) & (df_2022['VIX'] < vmax)
    mask_others = (df_others['VIX'] >= vmin) & (df_others['VIX'] < vmax)
    pct_2022 = mask_2022.sum() / len(df_2022) * 100
    pct_others = mask_others.sum() / len(df_others) * 100
    range_labels.append(label.split(' ')[0])
    pct_2022_list.append(pct_2022)
    pct_others_list.append(pct_others)

x_ranges = np.arange(len(range_labels))
bars1 = ax9.bar(x_ranges - width/2, pct_2022_list, width, label='2022', alpha=0.8, color='red')
bars2 = ax9.bar(x_ranges + width/2, pct_others_list, width, label='Otros', alpha=0.8, color='green')
ax9.set_xticks(x_ranges)
ax9.set_xticklabels(range_labels, rotation=45, ha='right')
ax9.set_title('Distribución Rangos VIX: 2022 vs Otros', fontsize=12, fontweight='bold')
ax9.set_ylabel('% de días')
ax9.legend()
ax9.grid(axis='y', alpha=0.3)

# 10. Trimestral 2022 - PnL
ax10 = plt.subplot(4, 3, 10)
if len(df_quarterly) > 0:
    x_q = np.arange(len(df_quarterly))
    width_q = 0.15
    for i, label in enumerate(fwd_labels[:4]):  # Solo primeros 4 para no saturar
        col_name = f'{label}_mean'
        if col_name in df_quarterly.columns:
            ax10.bar(x_q + i*width_q, df_quarterly[col_name], width_q,
                    label=label, alpha=0.8)
    ax10.set_xticks(x_q + width_q*1.5)
    ax10.set_xticklabels(df_quarterly['Quarter'])
    ax10.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax10.set_title('PnL Trimestral 2022', fontsize=12, fontweight='bold')
    ax10.set_ylabel('PnL Promedio')
    ax10.legend(fontsize=8)
    ax10.grid(axis='y', alpha=0.3)

# 11. Win Rate trimestral
ax11 = plt.subplot(4, 3, 11)
if len(df_quarterly) > 0:
    for i, label in enumerate(fwd_labels[:4]):
        col_name = f'{label}_WR'
        if col_name in df_quarterly.columns:
            ax11.plot(df_quarterly['Quarter'], df_quarterly[col_name],
                     marker='o', linewidth=2, markersize=8, label=label, alpha=0.8)
    ax11.axhline(y=50, color='black', linestyle='--', linewidth=1)
    ax11.set_title('Win Rate Trimestral 2022', fontsize=12, fontweight='bold')
    ax11.set_ylabel('Win Rate (%)')
    ax11.set_ylim(0, 100)
    ax11.legend(fontsize=8)
    ax11.grid(alpha=0.3)

# 12. Heatmap mensual todas las ventanas
ax12 = plt.subplot(4, 3, 12)
if len(df_monthly) > 0:
    heatmap_data = []
    for label in fwd_labels[:4]:
        col_name = f'{label}_mean'
        if col_name in df_monthly.columns:
            heatmap_data.append(df_monthly[col_name].values)

    if len(heatmap_data) > 0:
        im = ax12.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
        ax12.set_yticks(range(len(fwd_labels[:4])))
        ax12.set_yticklabels(fwd_labels[:4])
        ax12.set_xticks(range(len(df_monthly)))
        ax12.set_xticklabels(df_monthly['Mes'], rotation=45)
        ax12.set_title('Heatmap PnL Mensual 2022', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax12, label='PnL')

plt.tight_layout()
plt.savefig('analisis_2022_vix_forward.png', dpi=300, bbox_inches='tight')
print("   ✓ Visualizaciones guardadas: analisis_2022_vix_forward.png")

# ============================================================================
# EXPORTAR RESULTADOS
# ============================================================================
print("\n" + "="*80)
print("EXPORTANDO RESULTADOS")
print("="*80)

df_comparison.to_csv('comparacion_2022_vs_otros.csv', index=False)
print("   ✓ comparacion_2022_vs_otros.csv")

df_monthly.to_csv('analisis_mensual_2022.csv', index=False)
print("   ✓ analisis_mensual_2022.csv")

df_quarterly.to_csv('analisis_trimestral_2022.csv', index=False)
print("   ✓ analisis_trimestral_2022.csv")

df_range_2022.to_csv('rangos_vix_2022.csv', index=False)
print("   ✓ rangos_vix_2022.csv")

# ============================================================================
# CONCLUSIONES
# ============================================================================
print("\n" + "="*80)
print("CONCLUSIONES CLAVE - AÑO 2022")
print("="*80)

print("\n1. VIX EN 2022:")
print(f"   • Promedio: {df_2022['VIX'].mean():.2f} vs {df_others['VIX'].mean():.2f} (otros años)")
vix_diff_pct = (df_2022['VIX'].mean() - df_others['VIX'].mean()) / df_others['VIX'].mean() * 100
print(f"   • Diferencia: {vix_diff_pct:+.1f}%")
print(f"   • 2022 fue un año de {'MAYOR' if vix_diff_pct > 0 else 'MENOR'} volatilidad")

print("\n2. PERFORMANCE PnL FWD_25:")
pnl_25_2022 = df_comparison[df_comparison['Forward'] == 'FWD_25']['2022_mean'].values[0]
pnl_25_others = df_comparison[df_comparison['Forward'] == 'FWD_25']['Others_mean'].values[0]
print(f"   • 2022: {pnl_25_2022:+.2f}")
print(f"   • Otros: {pnl_25_others:+.2f}")
print(f"   • 2022 fue {'MEJOR' if pnl_25_2022 > pnl_25_others else 'PEOR'} ({pnl_25_2022 - pnl_25_others:+.2f} pts)")

print("\n3. CORRELACIÓN VIX-PnL:")
if 'FWD_25' in corr_2022 and 'FWD_25' in corr_others:
    corr_25_2022 = corr_2022['FWD_25']['pearson']
    corr_25_others = corr_others['FWD_25']['pearson']
    print(f"   • 2022: {corr_25_2022:+.4f}")
    print(f"   • Otros: {corr_25_others:+.4f}")
    print(f"   • La correlación {'AUMENTÓ' if corr_25_2022 > corr_25_others else 'DISMINUYÓ'} en 2022")

print("\n4. MEJOR MES 2022:")
if len(df_monthly) > 0 and 'FWD_25_mean' in df_monthly.columns:
    best_month = df_monthly.loc[df_monthly['FWD_25_mean'].idxmax()]
    print(f"   • {best_month['Mes']}: PnL={best_month['FWD_25_mean']:+.2f}")

print("\n5. PEOR MES 2022:")
if len(df_monthly) > 0 and 'FWD_25_mean' in df_monthly.columns:
    worst_month = df_monthly.loc[df_monthly['FWD_25_mean'].idxmin()]
    print(f"   • {worst_month['Mes']}: PnL={worst_month['FWD_25_mean']:+.2f}")

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO")
print("="*80)
