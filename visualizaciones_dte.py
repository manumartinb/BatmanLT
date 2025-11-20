#!/usr/bin/env python3
"""
Visualizaciones del Análisis DTE (Vencimientos)
Gráficos específicos para combinaciones de vencimientos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Configuración
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

print("📊 Generando visualizaciones de DTE...")
print()

# Cargar datos
df = pd.read_csv('combined_mediana.csv')

# Procesar DTE
df[['DTE1', 'DTE2']] = df['DTE1/DTE2'].str.split('/', expand=True)
df['DTE1'] = pd.to_numeric(df['DTE1'], errors='coerce')
df['DTE2'] = pd.to_numeric(df['DTE2'], errors='coerce')
df['DTE_ratio'] = df['DTE2'] / df['DTE1']
df['DTE_diff'] = df['DTE2'] - df['DTE1']
df['DTE_sum'] = df['DTE1'] + df['DTE2']
df['DTE_avg'] = (df['DTE1'] + df['DTE2']) / 2

target_vars = ['PnL_fwd_pts_01', 'PnL_fwd_pts_05', 'PnL_fwd_pts_25', 'PnL_fwd_pts_50']
dte_vars = ['DTE1', 'DTE2', 'DTE_ratio', 'DTE_diff', 'DTE_sum', 'DTE_avg']

# Limpiar
analysis_vars = target_vars + dte_vars
df_clean = df[analysis_vars].copy()
df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
df_clean = df_clean.dropna()

print(f"✓ Datos cargados: {len(df_clean)} registros")
print()

# 1. HEATMAP DE CORRELACIONES DTE
print("📈 Creando heatmap de correlaciones DTE...")

fig, ax = plt.subplots(figsize=(12, 8))

# Calcular correlaciones
corr_matrix = df_clean[dte_vars + ['PnL_fwd_pts_50']].corr()

# Heatmap
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
            vmin=-1, vmax=1, ax=ax, square=True,
            cbar_kws={'label': 'Correlación de Pearson'})

ax.set_title('Matriz de Correlación: Variables DTE vs PnL_fwd_pts_50',
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('dte_heatmap_correlaciones.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: dte_heatmap_correlaciones.png")
plt.close()

# 2. SCATTER PLOTS - DTE1, DTE2, RATIO vs PnL_fwd_pts_50
print("📊 Creando scatter plots DTE...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Scatter Plots: Variables DTE vs PnL_fwd_pts_50', fontsize=16, fontweight='bold')

for idx, dte_var in enumerate(dte_vars):
    ax = axes[idx // 3, idx % 3]

    # Scatter plot
    ax.scatter(df_clean[dte_var], df_clean['PnL_fwd_pts_50'],
               alpha=0.3, s=20, edgecolors='k', linewidth=0.3)

    # Línea de tendencia
    z = np.polyfit(df_clean[dte_var], df_clean['PnL_fwd_pts_50'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_clean[dte_var].min(), df_clean[dte_var].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, label='Tendencia')

    # Correlación
    corr, pval = pearsonr(df_clean[dte_var], df_clean['PnL_fwd_pts_50'])
    ax.text(0.05, 0.95, f'r = {corr:.4f}\np < 0.001' if pval < 0.001 else f'r = {corr:.4f}\np = {pval:.4f}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel(dte_var, fontsize=12, fontweight='bold')
    ax.set_ylabel('PnL_fwd_pts_50', fontsize=12, fontweight='bold')
    ax.set_title(f'{dte_var}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig('dte_scatter_plots.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: dte_scatter_plots.png")
plt.close()

# 3. ANÁLISIS POR RANGOS
print("📊 Creando análisis por rangos...")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Rentabilidad por Rangos de DTE', fontsize=16, fontweight='bold')

# DTE1 ranges
dte1_bins = [0, 100, 200, 300, 500, 1000, 2000]
dte1_labels = ['0-100', '100-200', '200-300', '300-500', '500-1000', '1000+']
df_clean['DTE1_range'] = pd.cut(df_clean['DTE1'], bins=dte1_bins, labels=dte1_labels)

ax = axes[0, 0]
dte1_data = df_clean.groupby('DTE1_range')['PnL_fwd_pts_50'].mean().sort_values(ascending=False)
colors_1 = ['#2ca02c' if x > 0 else '#d62728' for x in dte1_data.values]
bars = ax.bar(range(len(dte1_data)), dte1_data.values, color=colors_1, edgecolor='black')
ax.set_xticks(range(len(dte1_data)))
ax.set_xticklabels(dte1_data.index, rotation=45)
ax.set_ylabel('PnL_fwd_pts_50 promedio', fontsize=11, fontweight='bold')
ax.set_title('Por Rango de DTE1', fontsize=13, fontweight='bold')
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{height:.1f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9, fontweight='bold')

# DTE2 ranges
dte2_bins = [0, 200, 400, 600, 800, 1200, 3000]
dte2_labels = ['0-200', '200-400', '400-600', '600-800', '800-1200', '1200+']
df_clean['DTE2_range'] = pd.cut(df_clean['DTE2'], bins=dte2_bins, labels=dte2_labels)

ax = axes[0, 1]
dte2_data = df_clean.groupby('DTE2_range')['PnL_fwd_pts_50'].mean().sort_values(ascending=False)
colors_2 = ['#2ca02c' if x > 0 else '#d62728' for x in dte2_data.values]
bars = ax.bar(range(len(dte2_data)), dte2_data.values, color=colors_2, edgecolor='black')
ax.set_xticks(range(len(dte2_data)))
ax.set_xticklabels(dte2_data.index, rotation=45)
ax.set_ylabel('PnL_fwd_pts_50 promedio', fontsize=11, fontweight='bold')
ax.set_title('Por Rango de DTE2', fontsize=13, fontweight='bold')
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{height:.1f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9, fontweight='bold')

# Ratio ranges
ratio_bins = [0, 1.5, 2.0, 2.5, 3.0, 4.0, 10.0]
ratio_labels = ['<1.5x', '1.5-2.0x', '2.0-2.5x', '2.5-3.0x', '3.0-4.0x', '>4.0x']
df_clean['DTE_ratio_range'] = pd.cut(df_clean['DTE_ratio'], bins=ratio_bins, labels=ratio_labels)

ax = axes[1, 0]
ratio_data = df_clean.groupby('DTE_ratio_range')['PnL_fwd_pts_50'].mean().sort_values(ascending=False)
colors_3 = ['#2ca02c' if x > 0 else '#d62728' for x in ratio_data.values]
bars = ax.bar(range(len(ratio_data)), ratio_data.values, color=colors_3, edgecolor='black')
ax.set_xticks(range(len(ratio_data)))
ax.set_xticklabels(ratio_data.index, rotation=45)
ax.set_ylabel('PnL_fwd_pts_50 promedio', fontsize=11, fontweight='bold')
ax.set_title('Por Ratio DTE2/DTE1', fontsize=13, fontweight='bold')
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{height:.1f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9, fontweight='bold')

# Diff ranges
diff_bins = [0, 200, 400, 600, 800, 1200, 3000]
diff_labels = ['0-200', '200-400', '400-600', '600-800', '800-1200', '1200+']
df_clean['DTE_diff_range'] = pd.cut(df_clean['DTE_diff'], bins=diff_bins, labels=diff_labels)

ax = axes[1, 1]
diff_data = df_clean.groupby('DTE_diff_range')['PnL_fwd_pts_50'].mean().sort_values(ascending=False)
colors_4 = ['#2ca02c' if x > 0 else '#d62728' for x in diff_data.values]
bars = ax.bar(range(len(diff_data)), diff_data.values, color=colors_4, edgecolor='black')
ax.set_xticks(range(len(diff_data)))
ax.set_xticklabels(diff_data.index, rotation=45)
ax.set_ylabel('PnL_fwd_pts_50 promedio', fontsize=11, fontweight='bold')
ax.set_title('Por Diferencia DTE2-DTE1 (días)', fontsize=13, fontweight='bold')
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{height:.1f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('dte_analisis_rangos.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: dte_analisis_rangos.png")
plt.close()

# 4. TOP Y WORST COMBINACIONES
print("📊 Creando gráfico de top/worst combinaciones...")

# Agrupar por combinación
df_clean['DTE_combo'] = df_clean['DTE1'].astype(int).astype(str) + '/' + df_clean['DTE2'].astype(int).astype(str)
combo_stats = df_clean.groupby('DTE_combo').agg({
    'PnL_fwd_pts_50': ['mean', 'count']
})

# Filtrar combinaciones con al menos 30 muestras
combo_stats_filtered = combo_stats[combo_stats[('PnL_fwd_pts_50', 'count')] >= 30].copy()
combo_stats_filtered.columns = ['mean', 'count']
combo_stats_filtered = combo_stats_filtered.sort_values('mean', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Top 10 y Peores 10 Combinaciones DTE1/DTE2\n(Mínimo 30 muestras)',
             fontsize=16, fontweight='bold')

# Top 10
ax = axes[0]
top_10 = combo_stats_filtered.head(10).sort_values('mean')
colors_top = ['#2ca02c' for _ in range(len(top_10))]
bars = ax.barh(range(len(top_10)), top_10['mean'], color=colors_top, edgecolor='black')
ax.set_yticks(range(len(top_10)))
ax.set_yticklabels(top_10.index, fontsize=10)
ax.set_xlabel('PnL_fwd_pts_50 promedio', fontsize=12, fontweight='bold')
ax.set_title('✅ TOP 10 Combinaciones', fontsize=14, fontweight='bold', color='green')
ax.grid(True, alpha=0.3, axis='x')
for i, (bar, count) in enumerate(zip(bars, top_10['count'])):
    width = bar.get_width()
    ax.text(width + 2, bar.get_y() + bar.get_height()/2.,
           f'{width:.1f} (n={int(count)})',
           ha='left', va='center', fontsize=9, fontweight='bold')

# Worst 10
ax = axes[1]
worst_10 = combo_stats_filtered.tail(10).sort_values('mean')
colors_worst = ['#d62728' for _ in range(len(worst_10))]
bars = ax.barh(range(len(worst_10)), worst_10['mean'], color=colors_worst, edgecolor='black')
ax.set_yticks(range(len(worst_10)))
ax.set_yticklabels(worst_10.index, fontsize=10)
ax.set_xlabel('PnL_fwd_pts_50 promedio', fontsize=12, fontweight='bold')
ax.set_title('🚫 PEORES 10 Combinaciones', fontsize=14, fontweight='bold', color='red')
ax.grid(True, alpha=0.3, axis='x')
for i, (bar, count) in enumerate(zip(bars, worst_10['count'])):
    width = bar.get_width()
    ax.text(width - 2, bar.get_y() + bar.get_height()/2.,
           f'{width:.1f} (n={int(count)})',
           ha='right', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('dte_top_worst_combos.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: dte_top_worst_combos.png")
plt.close()

# 5. DISTRIBUCIONES DTE1 y DTE2
print("📊 Creando distribuciones DTE1 y DTE2...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Distribuciones de DTE1 y DTE2', fontsize=16, fontweight='bold')

# DTE1 histogram
ax = axes[0, 0]
ax.hist(df_clean['DTE1'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
p25 = np.percentile(df_clean['DTE1'], 25)
p50 = np.percentile(df_clean['DTE1'], 50)
p75 = np.percentile(df_clean['DTE1'], 75)
ax.axvline(p25, color='red', linestyle='--', linewidth=2, label=f'P25: {p25:.0f}')
ax.axvline(p50, color='green', linestyle='--', linewidth=2, label=f'P50: {p50:.0f}')
ax.axvline(p75, color='blue', linestyle='--', linewidth=2, label=f'P75: {p75:.0f}')
ax.set_xlabel('DTE1 (días)', fontsize=12)
ax.set_ylabel('Frecuencia', fontsize=12)
ax.set_title('Distribución de DTE1', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# DTE2 histogram
ax = axes[0, 1]
ax.hist(df_clean['DTE2'], bins=50, alpha=0.7, color='coral', edgecolor='black')
p25 = np.percentile(df_clean['DTE2'], 25)
p50 = np.percentile(df_clean['DTE2'], 50)
p75 = np.percentile(df_clean['DTE2'], 75)
ax.axvline(p25, color='red', linestyle='--', linewidth=2, label=f'P25: {p25:.0f}')
ax.axvline(p50, color='green', linestyle='--', linewidth=2, label=f'P50: {p50:.0f}')
ax.axvline(p75, color='blue', linestyle='--', linewidth=2, label=f'P75: {p75:.0f}')
ax.set_xlabel('DTE2 (días)', fontsize=12)
ax.set_ylabel('Frecuencia', fontsize=12)
ax.set_title('Distribución de DTE2', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Ratio histogram
ax = axes[1, 0]
ax.hist(df_clean['DTE_ratio'], bins=50, alpha=0.7, color='purple', edgecolor='black')
p25 = np.percentile(df_clean['DTE_ratio'], 25)
p50 = np.percentile(df_clean['DTE_ratio'], 50)
p75 = np.percentile(df_clean['DTE_ratio'], 75)
ax.axvline(p25, color='red', linestyle='--', linewidth=2, label=f'P25: {p25:.2f}')
ax.axvline(p50, color='green', linestyle='--', linewidth=2, label=f'P50: {p50:.2f}')
ax.axvline(p75, color='blue', linestyle='--', linewidth=2, label=f'P75: {p75:.2f}')
ax.set_xlabel('DTE_ratio (DTE2/DTE1)', fontsize=12)
ax.set_ylabel('Frecuencia', fontsize=12)
ax.set_title('Distribución de Ratio DTE2/DTE1', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Diff histogram
ax = axes[1, 1]
ax.hist(df_clean['DTE_diff'], bins=50, alpha=0.7, color='orange', edgecolor='black')
p25 = np.percentile(df_clean['DTE_diff'], 25)
p50 = np.percentile(df_clean['DTE_diff'], 50)
p75 = np.percentile(df_clean['DTE_diff'], 75)
ax.axvline(p25, color='red', linestyle='--', linewidth=2, label=f'P25: {p25:.0f}')
ax.axvline(p50, color='green', linestyle='--', linewidth=2, label=f'P50: {p50:.0f}')
ax.axvline(p75, color='blue', linestyle='--', linewidth=2, label=f'P75: {p75:.0f}')
ax.set_xlabel('DTE_diff (DTE2-DTE1 días)', fontsize=12)
ax.set_ylabel('Frecuencia', fontsize=12)
ax.set_title('Distribución de Diferencia DTE2-DTE1', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dte_distribuciones.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: dte_distribuciones.png")
plt.close()

# 6. MAPA DE CALOR 2D: DTE1 vs DTE2
print("📊 Creando mapa de calor 2D DTE1 vs DTE2...")

fig, ax = plt.subplots(figsize=(14, 10))

# Crear bins para DTE1 y DTE2
dte1_bins = np.linspace(df_clean['DTE1'].min(), df_clean['DTE1'].quantile(0.95), 20)
dte2_bins = np.linspace(df_clean['DTE2'].min(), df_clean['DTE2'].quantile(0.95), 20)

df_clean['DTE1_bin'] = pd.cut(df_clean['DTE1'], bins=dte1_bins, duplicates='drop')
df_clean['DTE2_bin'] = pd.cut(df_clean['DTE2'], bins=dte2_bins, duplicates='drop')

# Agrupar y calcular media
heatmap_data = df_clean.groupby(['DTE1_bin', 'DTE2_bin'])['PnL_fwd_pts_50'].mean().unstack()

# Heatmap
sns.heatmap(heatmap_data, cmap='RdYlGn', center=df_clean['PnL_fwd_pts_50'].median(),
            ax=ax, cbar_kws={'label': 'PnL_fwd_pts_50 promedio'},
            xticklabels=5, yticklabels=5)

ax.set_xlabel('DTE2 (días)', fontsize=12, fontweight='bold')
ax.set_ylabel('DTE1 (días)', fontsize=12, fontweight='bold')
ax.set_title('Mapa de Calor: Rentabilidad por DTE1 vs DTE2', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('dte_heatmap_2d.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: dte_heatmap_2d.png")
plt.close()

print()
print("="*80)
print(" ✨ VISUALIZACIONES DTE COMPLETADAS")
print("="*80)
print()
print("Archivos generados:")
print("  1. dte_heatmap_correlaciones.png")
print("  2. dte_scatter_plots.png")
print("  3. dte_analisis_rangos.png")
print("  4. dte_top_worst_combos.png")
print("  5. dte_distribuciones.png")
print("  6. dte_heatmap_2d.png")
print()
