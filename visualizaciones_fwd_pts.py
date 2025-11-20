#!/usr/bin/env python3
"""
Visualizaciones del Análisis FWD PTS
Gráficos de correlación, dispersión y umbrales
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

print("📊 Generando visualizaciones...")
print()

# Cargar datos
df = pd.read_csv('combined_mediana.csv')

# Variables
target_vars = [
    'PnL_fwd_pts_01',
    'PnL_fwd_pts_05',
    'PnL_fwd_pts_25',
    'PnL_fwd_pts_50',
]

predictor_vars = [
    'BQI_ABS',
    'BQI_V2_ABS',
    'delta_total',
    'theta_total',
    'PnLDV',
    'EarScore',
    'RATIO_BATMAN',
    'RATIO_UEL_EARS',
]

# Limpiar datos
analysis_vars = target_vars + predictor_vars
df_clean = df[analysis_vars].copy()
df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
df_clean = df_clean.dropna()

print(f"✓ Datos cargados: {len(df_clean)} registros")
print()

# 1. HEATMAP DE CORRELACIONES
print("📈 Creando heatmap de correlaciones...")

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('Matriz de Correlación: FWD PTS vs Variables Predictoras', fontsize=16, fontweight='bold')

for idx, target in enumerate(target_vars):
    ax = axes[idx // 2, idx % 2]

    # Calcular correlaciones
    correlations = []
    for predictor in predictor_vars:
        corr, _ = pearsonr(df_clean[predictor], df_clean[target])
        correlations.append(corr)

    # Crear dataframe para el heatmap
    corr_data = pd.DataFrame({
        'Variable': predictor_vars,
        'Correlación': correlations
    }).set_index('Variable')

    # Heatmap
    sns.heatmap(corr_data.T, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Correlación de Pearson'})
    ax.set_title(f'{target}', fontsize=14, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')

plt.tight_layout()
plt.savefig('heatmap_correlaciones.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: heatmap_correlaciones.png")
plt.close()

# 2. SCATTER PLOTS - TOP 3 PREDICTORES vs FWD PTS 50%
print("📊 Creando scatter plots...")

# Calcular correlaciones para PnL_fwd_pts_50
correlations_50 = []
for predictor in predictor_vars:
    corr, _ = pearsonr(df_clean[predictor], df_clean['PnL_fwd_pts_50'])
    correlations_50.append((predictor, abs(corr)))

top_3 = sorted(correlations_50, key=lambda x: x[1], reverse=True)[:3]
top_3_vars = [x[0] for x in top_3]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Scatter Plots: Top 3 Predictores vs PnL_fwd_pts_50', fontsize=16, fontweight='bold')

for idx, predictor in enumerate(top_3_vars):
    ax = axes[idx]

    # Scatter plot con línea de tendencia
    ax.scatter(df_clean[predictor], df_clean['PnL_fwd_pts_50'],
               alpha=0.5, s=30, edgecolors='k', linewidth=0.5)

    # Línea de tendencia
    z = np.polyfit(df_clean[predictor], df_clean['PnL_fwd_pts_50'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_clean[predictor].min(), df_clean[predictor].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, label='Tendencia')

    # Correlación
    corr, pval = pearsonr(df_clean[predictor], df_clean['PnL_fwd_pts_50'])
    ax.text(0.05, 0.95, f'r = {corr:.4f}\np = {pval:.4e}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel(predictor, fontsize=12, fontweight='bold')
    ax.set_ylabel('PnL_fwd_pts_50', fontsize=12, fontweight='bold')
    ax.set_title(f'{predictor}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig('scatter_top3_predictores.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: scatter_top3_predictores.png")
plt.close()

# 3. ANÁLISIS POR QUARTILES - TOP 3 PREDICTORES
print("📊 Creando análisis por quartiles...")

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle('Rentabilidad por Quartil: Top 3 Predictores', fontsize=16, fontweight='bold')

for pred_idx, predictor in enumerate(top_3_vars):
    # Crear quartiles
    df_clean['quartile'] = pd.qcut(df_clean[predictor], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')

    for target_idx, target in enumerate(target_vars):
        ax = axes[pred_idx, target_idx]

        # Datos por quartil
        quartile_data = df_clean.groupby('quartile')[target].mean()

        # Bar plot
        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
        bars = ax.bar(quartile_data.index, quartile_data.values, color=colors, edgecolor='black')

        # Añadir valores
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xlabel('Quartil', fontsize=10)
        ax.set_ylabel('PnL promedio (pts)', fontsize=10)
        ax.set_title(f'{predictor}\n{target}', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        # Línea horizontal en 0
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig('analisis_quartiles.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: analisis_quartiles.png")
plt.close()

# 4. DISTRIBUCIONES DE FWD PTS
print("📊 Creando distribuciones...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Distribuciones de FWD PTS', fontsize=16, fontweight='bold')

for idx, target in enumerate(target_vars):
    ax = axes[idx // 2, idx % 2]

    # Histogram
    ax.hist(df_clean[target], bins=50, alpha=0.7, color='steelblue', edgecolor='black')

    # Líneas verticales para percentiles
    p25 = np.percentile(df_clean[target], 25)
    p50 = np.percentile(df_clean[target], 50)
    p75 = np.percentile(df_clean[target], 75)

    ax.axvline(p25, color='red', linestyle='--', linewidth=2, label=f'P25: {p25:.1f}')
    ax.axvline(p50, color='green', linestyle='--', linewidth=2, label=f'P50: {p50:.1f}')
    ax.axvline(p75, color='blue', linestyle='--', linewidth=2, label=f'P75: {p75:.1f}')

    ax.set_xlabel('PnL (pts)', fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.set_title(f'{target}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Estadísticas
    stats_text = f'Media: {df_clean[target].mean():.2f}\nMediana: {df_clean[target].median():.2f}\nStd: {df_clean[target].std():.2f}'
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('distribuciones_fwd_pts.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: distribuciones_fwd_pts.png")
plt.close()

# 5. BOXPLOTS COMPARATIVOS
print("📊 Creando boxplots comparativos...")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Distribución de Predictores por Rendimiento (PnL_fwd_pts_50)', fontsize=16, fontweight='bold')

# Crear categoría de rendimiento
median_50 = df_clean['PnL_fwd_pts_50'].median()
df_clean['rendimiento'] = df_clean['PnL_fwd_pts_50'].apply(
    lambda x: 'Alto' if x > median_50 else 'Bajo'
)

for idx, predictor in enumerate(predictor_vars):
    ax = axes[idx // 4, idx % 4]

    # Boxplot
    df_clean.boxplot(column=predictor, by='rendimiento', ax=ax)
    ax.set_title(predictor, fontsize=12, fontweight='bold')
    ax.set_xlabel('Rendimiento', fontsize=10)
    ax.set_ylabel('Valor', fontsize=10)
    plt.sca(ax)
    plt.xticks(rotation=0)

    # Remover título automático
    ax.get_figure().suptitle('')

plt.suptitle('Distribución de Predictores por Rendimiento (PnL_fwd_pts_50)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('boxplots_rendimiento.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: boxplots_rendimiento.png")
plt.close()

# 6. CORRELACIÓN GENERAL - TODAS LAS VARIABLES
print("📊 Creando matriz de correlación general...")

fig, ax = plt.subplots(figsize=(14, 12))

# Matriz de correlación
corr_matrix = df_clean[analysis_vars].corr()

# Heatmap
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1, ax=ax, square=True,
            cbar_kws={'label': 'Correlación de Pearson'})

ax.set_title('Matriz de Correlación Completa', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('matriz_correlacion_completa.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: matriz_correlacion_completa.png")
plt.close()

# 7. EVOLUTION ACROSS TIME - FWD PTS
print("📊 Creando evolución temporal...")

fig, ax = plt.subplots(figsize=(12, 6))

# Promedios por tiempo de vida
time_points = ['01', '05', '25', '50']
means = [df_clean[f'PnL_fwd_pts_{t}'].mean() for t in time_points]
stds = [df_clean[f'PnL_fwd_pts_{t}'].std() for t in time_points]

# Plot con barras de error
ax.plot(time_points, means, 'o-', linewidth=3, markersize=10, color='steelblue', label='Media')
ax.fill_between(range(len(time_points)),
                 [m - s for m, s in zip(means, stds)],
                 [m + s for m, s in zip(means, stds)],
                 alpha=0.3, color='steelblue', label='±1 Std Dev')

ax.set_xlabel('% Tiempo de Vida', fontsize=14, fontweight='bold')
ax.set_ylabel('PnL promedio (pts)', fontsize=14, fontweight='bold')
ax.set_title('Evolución del PnL a lo largo del tiempo de vida', fontsize=16, fontweight='bold')
ax.set_xticklabels(['1%', '5%', '25%', '50%'])
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)

# Añadir valores
for i, (t, m) in enumerate(zip(time_points, means)):
    ax.text(i, m + 10, f'{m:.1f}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('evolucion_temporal_pnl.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: evolucion_temporal_pnl.png")
plt.close()

print()
print("="*80)
print(" ✨ VISUALIZACIONES COMPLETADAS")
print("="*80)
print()
print("Archivos generados:")
print("  1. heatmap_correlaciones.png")
print("  2. scatter_top3_predictores.png")
print("  3. analisis_quartiles.png")
print("  4. distribuciones_fwd_pts.png")
print("  5. boxplots_rendimiento.png")
print("  6. matriz_correlacion_completa.png")
print("  7. evolucion_temporal_pnl.png")
print()
