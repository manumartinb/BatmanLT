#!/usr/bin/env python3
"""
Análisis exhaustivo de correlación entre VIX y Forward Points
Autor: Claude
Fecha: 2025-11-21
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("ANÁLISIS EXHAUSTIVO: VIX vs FORWARD POINTS")
print("="*80)

# Cargar datos
print("\n[1/6] Cargando datos...")
df = pd.read_csv('VIX_combined_mediana.csv')
print(f"   ✓ Datos cargados: {len(df):,} registros")
print(f"   ✓ Periodo: {df['dia'].min()} a {df['dia'].max()}")

# Columnas de análisis
fwd_cols = ['PnL_fwd_pts_01', 'PnL_fwd_pts_05', 'PnL_fwd_pts_25', 'PnL_fwd_pts_50', 'PnL_fwd_pts_90']
fwd_labels = ['FWD_01', 'FWD_05', 'FWD_25', 'FWD_50', 'FWD_90']

# ============================================================================
# SECCIÓN 1: ANÁLISIS DE CORRELACIÓN BÁSICO
# ============================================================================
print("\n" + "="*80)
print("SECCIÓN 1: CORRELACIONES BÁSICAS VIX vs FORWARD POINTS")
print("="*80)

correlations = {}
for col, label in zip(fwd_cols, fwd_labels):
    valid_data = df[['VIX', col]].dropna()
    if len(valid_data) > 0:
        corr_pearson = valid_data['VIX'].corr(valid_data[col])
        corr_spearman = valid_data['VIX'].corr(valid_data[col], method='spearman')
        correlations[label] = {
            'pearson': corr_pearson,
            'spearman': corr_spearman,
            'n_samples': len(valid_data)
        }
        print(f"\n{label}:")
        print(f"   Correlación Pearson:  {corr_pearson:+.4f}")
        print(f"   Correlación Spearman: {corr_spearman:+.4f}")
        print(f"   Muestras: {len(valid_data):,}")

# ============================================================================
# SECCIÓN 2: ANÁLISIS POR RANGOS DE VIX
# ============================================================================
print("\n" + "="*80)
print("SECCIÓN 2: ANÁLISIS POR RANGOS DE VIX")
print("="*80)

# Definir rangos de VIX
vix_ranges = [
    (0, 15, 'MUY BAJO (0-15)'),
    (15, 20, 'BAJO (15-20)'),
    (20, 30, 'MEDIO (20-30)'),
    (30, 40, 'ALTO (30-40)'),
    (40, 60, 'MUY ALTO (40-60)'),
    (60, 200, 'EXTREMO (>60)')
]

range_stats = []
for vmin, vmax, label in vix_ranges:
    mask = (df['VIX'] >= vmin) & (df['VIX'] < vmax)
    subset = df[mask]

    if len(subset) > 0:
        print(f"\n{label}:")
        print(f"   Registros: {len(subset):,} ({len(subset)/len(df)*100:.1f}%)")
        print(f"   VIX promedio: {subset['VIX'].mean():.2f}")

        stats_row = {
            'Rango': label,
            'VIX_min': vmin,
            'VIX_max': vmax,
            'Count': len(subset),
            'Pct': len(subset)/len(df)*100
        }

        for col, flabel in zip(fwd_cols, fwd_labels):
            if col in subset.columns:
                mean_pnl = subset[col].mean()
                median_pnl = subset[col].median()
                std_pnl = subset[col].std()
                win_rate = (subset[col] > 0).sum() / len(subset) * 100

                stats_row[f'{flabel}_mean'] = mean_pnl
                stats_row[f'{flabel}_median'] = median_pnl
                stats_row[f'{flabel}_std'] = std_pnl
                stats_row[f'{flabel}_winrate'] = win_rate

                print(f"   {flabel}: μ={mean_pnl:+8.2f} | Med={median_pnl:+8.2f} | σ={std_pnl:7.2f} | WR={win_rate:5.1f}%")

        range_stats.append(stats_row)

df_range_stats = pd.DataFrame(range_stats)

# ============================================================================
# SECCIÓN 3: ANÁLISIS PROSPECTIVO - CAMBIOS EN VIX
# ============================================================================
print("\n" + "="*80)
print("SECCIÓN 3: ANÁLISIS PROSPECTIVO - CAMBIOS EN VIX A FUTURO")
print("="*80)

# Calcular cambios en VIX
df['VIX_change_1d'] = df['VIX'].shift(-1) - df['VIX']
df['VIX_change_5d'] = df['VIX'].shift(-5) - df['VIX']
df['VIX_change_pct_1d'] = (df['VIX'].shift(-1) - df['VIX']) / df['VIX'] * 100
df['VIX_change_pct_5d'] = (df['VIX'].shift(-5) - df['VIX']) / df['VIX'] * 100

# Categorizar cambios en VIX
def categorize_vix_change(change_pct):
    if pd.isna(change_pct):
        return 'Unknown'
    elif change_pct < -10:
        return 'CAÍDA FUERTE (>10%)'
    elif change_pct < -5:
        return 'CAÍDA MODERADA (5-10%)'
    elif change_pct < 5:
        return 'ESTABLE (±5%)'
    elif change_pct < 10:
        return 'SUBE MODERADO (5-10%)'
    else:
        return 'SUBE FUERTE (>10%)'

df['VIX_future_1d'] = df['VIX_change_pct_1d'].apply(categorize_vix_change)
df['VIX_future_5d'] = df['VIX_change_pct_5d'].apply(categorize_vix_change)

# Análisis por cambio futuro de VIX (5 días)
print("\n--- Cuando el VIX cambia en los próximos 5 días ---")
vix_change_order = ['CAÍDA FUERTE (>10%)', 'CAÍDA MODERADA (5-10%)', 'ESTABLE (±5%)',
                    'SUBE MODERADO (5-10%)', 'SUBE FUERTE (>10%)']

future_stats = []
for category in vix_change_order:
    subset = df[df['VIX_future_5d'] == category]

    if len(subset) > 10:  # Mínimo 10 muestras
        print(f"\n{category}:")
        print(f"   Registros: {len(subset):,} ({len(subset)/len(df)*100:.1f}%)")
        print(f"   VIX promedio inicial: {subset['VIX'].mean():.2f}")

        stats_row = {
            'Escenario': category,
            'Count': len(subset),
            'VIX_inicial': subset['VIX'].mean()
        }

        for col, flabel in zip(fwd_cols, fwd_labels):
            if col in subset.columns:
                mean_pnl = subset[col].mean()
                median_pnl = subset[col].median()
                win_rate = (subset[col] > 0).sum() / len(subset) * 100

                stats_row[f'{flabel}_mean'] = mean_pnl
                stats_row[f'{flabel}_winrate'] = win_rate

                print(f"   {flabel}: μ={mean_pnl:+8.2f} | Med={median_pnl:+8.2f} | WR={win_rate:5.1f}%")

        future_stats.append(stats_row)

df_future_stats = pd.DataFrame(future_stats)

# ============================================================================
# SECCIÓN 4: ANÁLISIS DE REGÍMENES DE VOLATILIDAD
# ============================================================================
print("\n" + "="*80)
print("SECCIÓN 4: REGÍMENES DE VOLATILIDAD")
print("="*80)

# Calcular percentiles de VIX
vix_p25 = df['VIX'].quantile(0.25)
vix_p50 = df['VIX'].quantile(0.50)
vix_p75 = df['VIX'].quantile(0.75)
vix_p90 = df['VIX'].quantile(0.90)

print(f"\nPercentiles VIX:")
print(f"   P25: {vix_p25:.2f}")
print(f"   P50 (Mediana): {vix_p50:.2f}")
print(f"   P75: {vix_p75:.2f}")
print(f"   P90: {vix_p90:.2f}")

regime_labels = []
for _, row in df.iterrows():
    vix = row['VIX']
    if pd.isna(vix):
        regime_labels.append('Unknown')
    elif vix < vix_p25:
        regime_labels.append('Q1-MUY_BAJO')
    elif vix < vix_p50:
        regime_labels.append('Q2-BAJO')
    elif vix < vix_p75:
        regime_labels.append('Q3-MEDIO')
    elif vix < vix_p90:
        regime_labels.append('Q4-ALTO')
    else:
        regime_labels.append('Q5-EXTREMO')

df['VIX_regime'] = regime_labels

print("\n--- Análisis por Régimen ---")
regime_order = ['Q1-MUY_BAJO', 'Q2-BAJO', 'Q3-MEDIO', 'Q4-ALTO', 'Q5-EXTREMO']
for regime in regime_order:
    subset = df[df['VIX_regime'] == regime]

    if len(subset) > 0:
        print(f"\n{regime}:")
        print(f"   Registros: {len(subset):,}")
        print(f"   Rango VIX: [{subset['VIX'].min():.2f}, {subset['VIX'].max():.2f}]")

        for col, flabel in zip(fwd_cols, fwd_labels):
            if col in subset.columns:
                mean_pnl = subset[col].mean()
                sharpe = mean_pnl / subset[col].std() if subset[col].std() > 0 else 0
                print(f"   {flabel}: μ={mean_pnl:+8.2f} | Sharpe={sharpe:+.3f}")

# ============================================================================
# SECCIÓN 5: ANÁLISIS DE MOMENTOS ÓPTIMOS
# ============================================================================
print("\n" + "="*80)
print("SECCIÓN 5: ¿CUÁNDO ES MEJOR OPERAR? (MEJORES CONDICIONES)")
print("="*80)

# Para cada ventana forward, encontrar condiciones óptimas
for col, flabel in zip(fwd_cols, fwd_labels):
    print(f"\n{flabel}:")

    # Top 5 mejores rangos de VIX
    best_ranges = []
    for vmin, vmax, label in vix_ranges:
        mask = (df['VIX'] >= vmin) & (df['VIX'] < vmax)
        subset = df[mask]
        if len(subset) > 20:  # Mínimo 20 muestras
            mean_pnl = subset[col].mean()
            best_ranges.append((label, mean_pnl, len(subset)))

    best_ranges.sort(key=lambda x: x[1], reverse=True)

    print("   🏆 Mejores rangos VIX:")
    for i, (label, mean_pnl, count) in enumerate(best_ranges[:3], 1):
        print(f"      {i}. {label}: μ={mean_pnl:+.2f} (n={count})")

    print("   ⚠️  Peores rangos VIX:")
    for i, (label, mean_pnl, count) in enumerate(best_ranges[-3:], 1):
        print(f"      {i}. {label}: μ={mean_pnl:+.2f} (n={count})")

# ============================================================================
# SECCIÓN 6: ANÁLISIS CREATIVO - VIX MOMENTUM
# ============================================================================
print("\n" + "="*80)
print("SECCIÓN 6: ANÁLISIS CREATIVO - VIX MOMENTUM Y TENDENCIAS")
print("="*80)

# Calcular momentum de VIX (cambio en 3 días)
df['VIX_momentum_3d'] = df['VIX'] - df['VIX'].shift(3)
df['VIX_trend'] = df['VIX_momentum_3d'].apply(
    lambda x: 'BAJISTA' if x < -3 else ('ALCISTA' if x > 3 else 'LATERAL')
)

print("\n--- Análisis por Tendencia del VIX ---")
for trend in ['BAJISTA', 'LATERAL', 'ALCISTA']:
    subset = df[df['VIX_trend'] == trend]

    if len(subset) > 0:
        print(f"\n{trend} (VIX en tendencia {trend.lower()}):")
        print(f"   Registros: {len(subset):,}")

        for col, flabel in zip(fwd_cols, fwd_labels):
            if col in subset.columns:
                mean_pnl = subset[col].mean()
                win_rate = (subset[col] > 0).sum() / len(subset) * 100
                print(f"   {flabel}: μ={mean_pnl:+8.2f} | WR={win_rate:5.1f}%")

# VIX Spike Analysis
print("\n--- Análisis de SPIKES de VIX ---")
df['VIX_spike'] = df['VIX_change_pct_1d'] > 20  # Spike = aumento >20% en 1 día

spike_days = df[df['VIX_spike'] == True]
print(f"\nDías con VIX Spike (>20% en 1 día): {len(spike_days)}")

if len(spike_days) > 0:
    print("\nPnL promedio DESPUÉS de un VIX Spike:")
    for col, flabel in zip(fwd_cols, fwd_labels):
        if col in spike_days.columns:
            mean_pnl = spike_days[col].mean()
            print(f"   {flabel}: {mean_pnl:+.2f}")

# ============================================================================
# SECCIÓN 7: VISUALIZACIONES
# ============================================================================
print("\n" + "="*80)
print("SECCIÓN 7: GENERANDO VISUALIZACIONES")
print("="*80)

# Crear figura con múltiples subplots
fig = plt.figure(figsize=(20, 24))

# 1. Correlación VIX vs Forward Points
ax1 = plt.subplot(6, 2, 1)
corr_data = [correlations[label]['pearson'] for label in fwd_labels]
bars = ax1.bar(fwd_labels, corr_data, color=['green' if x > 0 else 'red' for x in corr_data])
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax1.set_title('Correlación Pearson: VIX vs Forward Points', fontsize=12, fontweight='bold')
ax1.set_ylabel('Correlación')
ax1.set_ylim(-1, 1)
ax1.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, corr_data):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.3f}', ha='center', va='bottom' if val > 0 else 'top', fontsize=9)

# 2. PnL promedio por rango de VIX (FWD_25)
ax2 = plt.subplot(6, 2, 2)
if len(df_range_stats) > 0:
    x_labels = df_range_stats['Rango'].values
    y_values = df_range_stats['FWD_25_mean'].values
    colors = ['green' if y > 0 else 'red' for y in y_values]
    bars = ax2.barh(x_labels, y_values, color=colors, alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title('PnL Promedio por Rango VIX (FWD_25)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('PnL Promedio')
    ax2.grid(axis='x', alpha=0.3)

# 3. Scatter: VIX vs FWD_25
ax3 = plt.subplot(6, 2, 3)
valid_data = df[['VIX', 'PnL_fwd_pts_25']].dropna()
if len(valid_data) > 0:
    sample = valid_data.sample(min(5000, len(valid_data)))
    scatter = ax3.scatter(sample['VIX'], sample['PnL_fwd_pts_25'],
                          alpha=0.3, s=20, c=sample['VIX'], cmap='coolwarm')
    z = np.polyfit(sample['VIX'], sample['PnL_fwd_pts_25'], 1)
    p = np.poly1d(z)
    ax3.plot(sample['VIX'].sort_values(), p(sample['VIX'].sort_values()),
             "r--", alpha=0.8, linewidth=2, label=f'Tendencia: y={z[0]:.2f}x+{z[1]:.2f}')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_title('VIX vs PnL Forward 25 días', fontsize=12, fontweight='bold')
    ax3.set_xlabel('VIX')
    ax3.set_ylabel('PnL_fwd_pts_25')
    ax3.legend()
    plt.colorbar(scatter, ax=ax3, label='VIX Level')

# 4. Win Rate por rango de VIX (FWD_25)
ax4 = plt.subplot(6, 2, 4)
if len(df_range_stats) > 0:
    x_labels = df_range_stats['Rango'].values
    y_values = df_range_stats['FWD_25_winrate'].values
    bars = ax4.barh(x_labels, y_values, color='skyblue', alpha=0.7)
    ax4.axvline(x=50, color='red', linestyle='--', linewidth=1, label='50%')
    ax4.set_title('Win Rate por Rango VIX (FWD_25)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Win Rate (%)')
    ax4.set_xlim(0, 100)
    ax4.legend()
    ax4.grid(axis='x', alpha=0.3)

# 5. PnL por cambio futuro de VIX
ax5 = plt.subplot(6, 2, 5)
if len(df_future_stats) > 0:
    x_pos = np.arange(len(df_future_stats))
    width = 0.15
    for i, flabel in enumerate(fwd_labels):
        col_name = f'{flabel}_mean'
        if col_name in df_future_stats.columns:
            values = df_future_stats[col_name].values
            ax5.bar(x_pos + i*width, values, width, label=flabel, alpha=0.8)

    ax5.set_xticks(x_pos + width * 2)
    ax5.set_xticklabels(df_future_stats['Escenario'].values, rotation=45, ha='right')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax5.set_title('PnL según Cambio Futuro del VIX (5 días)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('PnL Promedio')
    ax5.legend(loc='upper left', fontsize=8)
    ax5.grid(axis='y', alpha=0.3)

# 6. Heatmap: Correlaciones entre todas las ventanas forward
ax6 = plt.subplot(6, 2, 6)
corr_matrix = df[fwd_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            xticklabels=fwd_labels, yticklabels=fwd_labels, ax=ax6,
            vmin=-1, vmax=1, cbar_kws={'label': 'Correlación'})
ax6.set_title('Correlación entre Ventanas Forward', fontsize=12, fontweight='bold')

# 7. Boxplot: Distribución PnL por régimen VIX
ax7 = plt.subplot(6, 2, 7)
regime_data = []
regime_labels_plot = []
for regime in regime_order:
    subset = df[df['VIX_regime'] == regime]['PnL_fwd_pts_25'].dropna()
    if len(subset) > 0:
        regime_data.append(subset)
        regime_labels_plot.append(regime)

if len(regime_data) > 0:
    bp = ax7.boxplot(regime_data, labels=regime_labels_plot, patch_artist=True,
                     showfliers=False, medianprops=dict(color='red', linewidth=2))
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax7.set_title('Distribución PnL_25 por Régimen VIX', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Régimen VIX')
    ax7.set_ylabel('PnL_fwd_pts_25')
    ax7.tick_params(axis='x', rotation=45)
    ax7.grid(axis='y', alpha=0.3)

# 8. Serie temporal: VIX y PnL_25 (sample)
ax8 = plt.subplot(6, 2, 8)
sample_df = df.sample(min(500, len(df))).sort_values('dia')
ax8_twin = ax8.twinx()
ax8.plot(range(len(sample_df)), sample_df['VIX'].values,
         color='orange', alpha=0.7, label='VIX', linewidth=1.5)
ax8_twin.plot(range(len(sample_df)), sample_df['PnL_fwd_pts_25'].values,
              color='blue', alpha=0.7, label='PnL_25', linewidth=1.5)
ax8.set_title('Serie Temporal: VIX y PnL_25 (muestra)', fontsize=12, fontweight='bold')
ax8.set_xlabel('Índice temporal')
ax8.set_ylabel('VIX', color='orange')
ax8_twin.set_ylabel('PnL_fwd_pts_25', color='blue')
ax8.tick_params(axis='y', labelcolor='orange')
ax8_twin.tick_params(axis='y', labelcolor='blue')
ax8.grid(alpha=0.3)

# 9. PnL acumulado por régimen
ax9 = plt.subplot(6, 2, 9)
for regime in regime_order:
    subset = df[df['VIX_regime'] == regime]['PnL_fwd_pts_25'].dropna()
    if len(subset) > 100:
        cumsum = subset.cumsum()
        ax9.plot(range(len(cumsum)), cumsum.values, label=regime, linewidth=2, alpha=0.7)

ax9.set_title('PnL Acumulado por Régimen VIX (FWD_25)', fontsize=12, fontweight='bold')
ax9.set_xlabel('Trades')
ax9.set_ylabel('PnL Acumulado')
ax9.legend(loc='best', fontsize=8)
ax9.grid(alpha=0.3)
ax9.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# 10. Scatter: VIX vs FWD_50
ax10 = plt.subplot(6, 2, 10)
valid_data = df[['VIX', 'PnL_fwd_pts_50']].dropna()
if len(valid_data) > 0:
    sample = valid_data.sample(min(5000, len(valid_data)))
    scatter = ax10.scatter(sample['VIX'], sample['PnL_fwd_pts_50'],
                           alpha=0.3, s=20, c=sample['VIX'], cmap='viridis')
    z = np.polyfit(sample['VIX'], sample['PnL_fwd_pts_50'], 1)
    p = np.poly1d(z)
    ax10.plot(sample['VIX'].sort_values(), p(sample['VIX'].sort_values()),
              "r--", alpha=0.8, linewidth=2, label=f'Tendencia: y={z[0]:.2f}x+{z[1]:.2f}')
    ax10.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax10.set_title('VIX vs PnL Forward 50 días', fontsize=12, fontweight='bold')
    ax10.set_xlabel('VIX')
    ax10.set_ylabel('PnL_fwd_pts_50')
    ax10.legend()
    plt.colorbar(scatter, ax=ax10, label='VIX Level')

# 11. Histograma de VIX
ax11 = plt.subplot(6, 2, 11)
ax11.hist(df['VIX'].dropna(), bins=50, color='purple', alpha=0.7, edgecolor='black')
ax11.axvline(vix_p25, color='green', linestyle='--', linewidth=2, label=f'P25={vix_p25:.1f}')
ax11.axvline(vix_p50, color='yellow', linestyle='--', linewidth=2, label=f'P50={vix_p50:.1f}')
ax11.axvline(vix_p75, color='orange', linestyle='--', linewidth=2, label=f'P75={vix_p75:.1f}')
ax11.axvline(vix_p90, color='red', linestyle='--', linewidth=2, label=f'P90={vix_p90:.1f}')
ax11.set_title('Distribución del VIX', fontsize=12, fontweight='bold')
ax11.set_xlabel('VIX')
ax11.set_ylabel('Frecuencia')
ax11.legend()
ax11.grid(axis='y', alpha=0.3)

# 12. PnL por tendencia VIX
ax12 = plt.subplot(6, 2, 12)
trend_stats = []
for trend in ['BAJISTA', 'LATERAL', 'ALCISTA']:
    subset = df[df['VIX_trend'] == trend]
    if len(subset) > 0:
        for col, flabel in zip(fwd_cols, fwd_labels):
            if col in subset.columns:
                mean_pnl = subset[col].mean()
                trend_stats.append({
                    'Trend': trend,
                    'Forward': flabel,
                    'PnL': mean_pnl
                })

df_trend_stats = pd.DataFrame(trend_stats)
if len(df_trend_stats) > 0:
    pivot_trend = df_trend_stats.pivot(index='Trend', columns='Forward', values='PnL')
    pivot_trend = pivot_trend.reindex(['BAJISTA', 'LATERAL', 'ALCISTA'])
    pivot_trend.plot(kind='bar', ax=ax12, width=0.8, alpha=0.8)
    ax12.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax12.set_title('PnL por Tendencia del VIX', fontsize=12, fontweight='bold')
    ax12.set_xlabel('Tendencia VIX')
    ax12.set_ylabel('PnL Promedio')
    ax12.legend(title='Forward', loc='best', fontsize=8)
    ax12.grid(axis='y', alpha=0.3)
    ax12.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('analisis_vix_forward_correlacion.png', dpi=300, bbox_inches='tight')
print("\n   ✓ Visualizaciones guardadas en: analisis_vix_forward_correlacion.png")

# ============================================================================
# SECCIÓN 8: EXPORTAR RESULTADOS
# ============================================================================
print("\n" + "="*80)
print("SECCIÓN 8: EXPORTANDO RESULTADOS")
print("="*80)

# Guardar estadísticas en CSV
df_range_stats.to_csv('vix_range_statistics.csv', index=False)
print("   ✓ Estadísticas por rango guardadas: vix_range_statistics.csv")

df_future_stats.to_csv('vix_future_change_statistics.csv', index=False)
print("   ✓ Estadísticas por cambio futuro guardadas: vix_future_change_statistics.csv")

# Guardar correlaciones
corr_df = pd.DataFrame(correlations).T
corr_df.to_csv('vix_correlations.csv')
print("   ✓ Correlaciones guardadas: vix_correlations.csv")

# ============================================================================
# RESUMEN EJECUTIVO
# ============================================================================
print("\n" + "="*80)
print("RESUMEN EJECUTIVO - HALLAZGOS CLAVE")
print("="*80)

print("\n1. CORRELACIONES:")
for label in fwd_labels:
    corr = correlations[label]['pearson']
    direction = "NEGATIVA" if corr < 0 else "POSITIVA"
    strength = "FUERTE" if abs(corr) > 0.5 else ("MODERADA" if abs(corr) > 0.3 else "DÉBIL")
    print(f"   • {label}: Correlación {direction} {strength} ({corr:+.3f})")

print("\n2. MEJOR RANGO VIX PARA OPERAR (FWD_25):")
if len(df_range_stats) > 0:
    best_range = df_range_stats.loc[df_range_stats['FWD_25_mean'].idxmax()]
    print(f"   • {best_range['Rango']}: μ={best_range['FWD_25_mean']:+.2f}")

print("\n3. PEOR RANGO VIX (FWD_25):")
if len(df_range_stats) > 0:
    worst_range = df_range_stats.loc[df_range_stats['FWD_25_mean'].idxmin()]
    print(f"   • {worst_range['Rango']}: μ={worst_range['FWD_25_mean']:+.2f}")

print("\n4. MEJOR ESCENARIO FUTURO (cuando el VIX a 5 días...):")
if len(df_future_stats) > 0:
    best_scenario = df_future_stats.loc[df_future_stats['FWD_25_mean'].idxmax()]
    print(f"   • {best_scenario['Escenario']}: μ={best_scenario['FWD_25_mean']:+.2f}")

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO")
print("="*80)
print(f"\nTiempo de ejecución: {datetime.now()}")
print("\nArchivos generados:")
print("   1. analisis_vix_forward_correlacion.png")
print("   2. vix_range_statistics.csv")
print("   3. vix_future_change_statistics.csv")
print("   4. vix_correlations.csv")
print("\n" + "="*80)
