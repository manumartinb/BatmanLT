#!/usr/bin/env python3
"""
Análisis Hipótesis Q4 2022: ¿Caída del VIX causó las pérdidas?
Investigación: VIX alto en apertura → VIX baja posteriormente → pérdidas
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
print("HIPÓTESIS: ¿Q4 2022 colapsó por CAÍDA SUBSIGUIENTE del VIX?")
print("="*80)

# Cargar datos
df = pd.read_csv('VIX_combined_mediana.csv')
df['dia'] = pd.to_datetime(df['dia'])
df['year'] = df['dia'].dt.year
df['quarter'] = df['dia'].dt.quarter
df['month'] = df['dia'].dt.month

# Calcular cambios futuros del VIX
df['VIX_change_1d'] = df['VIX'].shift(-1) - df['VIX']
df['VIX_change_5d'] = df['VIX'].shift(-5) - df['VIX']
df['VIX_change_25d'] = df['VIX'].shift(-25) - df['VIX']
df['VIX_change_50d'] = df['VIX'].shift(-50) - df['VIX']

df['VIX_pct_change_1d'] = (df['VIX'].shift(-1) - df['VIX']) / df['VIX'] * 100
df['VIX_pct_change_5d'] = (df['VIX'].shift(-5) - df['VIX']) / df['VIX'] * 100
df['VIX_pct_change_25d'] = (df['VIX'].shift(-25) - df['VIX']) / df['VIX'] * 100
df['VIX_pct_change_50d'] = (df['VIX'].shift(-50) - df['VIX']) / df['VIX'] * 100

# VIX futuro
df['VIX_future_5d'] = df['VIX'].shift(-5)
df['VIX_future_25d'] = df['VIX'].shift(-25)
df['VIX_future_50d'] = df['VIX'].shift(-50)

# Separar datasets
df_2022 = df[df['year'] == 2022].copy()
df_q4_2022 = df[(df['year'] == 2022) & (df['quarter'] == 4)].copy()
df_q1_2022 = df[(df['year'] == 2022) & (df['quarter'] == 1)].copy()
df_others = df[df['year'] != 2022].copy()

print(f"\n[DATOS]")
print(f"   Q4 2022: {len(df_q4_2022):,} registros")
print(f"   Q1 2022: {len(df_q1_2022):,} registros")
print(f"   Resto 2022: {len(df_2022):,} registros")

# ============================================================================
# SECCIÓN 1: EVOLUCIÓN DEL VIX EN Q4 2022
# ============================================================================
print("\n" + "="*80)
print("SECCIÓN 1: EVOLUCIÓN DEL VIX EN Q4 2022")
print("="*80)

print("\nEstadísticas VIX Q4 2022:")
print(f"   VIX promedio: {df_q4_2022['VIX'].mean():.2f}")
print(f"   VIX min/max: [{df_q4_2022['VIX'].min():.2f}, {df_q4_2022['VIX'].max():.2f}]")
print(f"   VIX inicio Q4: {df_q4_2022['VIX'].iloc[0]:.2f} (Oct 1)")
print(f"   VIX fin Q4: {df_q4_2022['VIX'].iloc[-1]:.2f} (Dec 31)")

# Evolución mensual dentro de Q4
for month in [10, 11, 12]:
    df_month = df_q4_2022[df_q4_2022['month'] == month]
    if len(df_month) > 0:
        month_name = ['', '', '', '', '', '', '', '', '', '', 'Oct', 'Nov', 'Dic'][month]
        print(f"\n{month_name}:")
        print(f"   VIX promedio: {df_month['VIX'].mean():.2f}")
        print(f"   VIX inicio: {df_month['VIX'].iloc[0]:.2f}")
        print(f"   VIX fin: {df_month['VIX'].iloc[-1]:.2f}")
        print(f"   Cambio: {df_month['VIX'].iloc[-1] - df_month['VIX'].iloc[0]:.2f}")

# ============================================================================
# SECCIÓN 2: CAMBIOS FUTUROS DEL VIX - Q4 2022 vs Q1 2022 vs RESTO
# ============================================================================
print("\n" + "="*80)
print("SECCIÓN 2: ¿EL VIX CAYÓ DESPUÉS DE ABRIR POSICIONES?")
print("="*80)

datasets = {
    'Q4 2022': df_q4_2022,
    'Q1 2022': df_q1_2022,
    'Otros años': df_others
}

comparison_vix_changes = []

for name, dataset in datasets.items():
    print(f"\n{name}:")
    print(f"   VIX promedio en apertura: {dataset['VIX'].mean():.2f}")

    stats_row = {
        'Dataset': name,
        'VIX_apertura': dataset['VIX'].mean(),
        'N': len(dataset)
    }

    # Cambios en VIX
    for days, col in [(1, 'VIX_pct_change_1d'), (5, 'VIX_pct_change_5d'),
                       (25, 'VIX_pct_change_25d'), (50, 'VIX_pct_change_50d')]:
        if col in dataset.columns:
            change = dataset[col].mean()
            pct_falling = (dataset[col] < 0).sum() / len(dataset[col].dropna()) * 100
            stats_row[f'VIX_change_{days}d'] = change
            stats_row[f'VIX_falling_{days}d_pct'] = pct_falling
            print(f"   Cambio VIX {days}d: {change:+.2f}% | VIX cayó en {pct_falling:.1f}% de casos")

    comparison_vix_changes.append(stats_row)

df_vix_comparison = pd.DataFrame(comparison_vix_changes)

# ============================================================================
# SECCIÓN 3: CORRELACIÓN ENTRE CAÍDA DE VIX Y PnL
# ============================================================================
print("\n" + "="*80)
print("SECCIÓN 3: ¿CAÍDA DEL VIX → PÉRDIDAS?")
print("="*80)

print("\nQ4 2022 - Correlación entre cambio VIX y PnL:")
for vix_col, pnl_col, label in [
    ('VIX_pct_change_5d', 'PnL_fwd_pts_05', 'FWD_05'),
    ('VIX_pct_change_25d', 'PnL_fwd_pts_25', 'FWD_25'),
    ('VIX_pct_change_50d', 'PnL_fwd_pts_50', 'FWD_50')
]:
    valid = df_q4_2022[[vix_col, pnl_col]].dropna()
    if len(valid) > 10:
        corr = valid[vix_col].corr(valid[pnl_col])
        print(f"   {label}: Corr(Δ%VIX, PnL) = {corr:+.4f}")

print("\nQ1 2022 - Correlación entre cambio VIX y PnL:")
for vix_col, pnl_col, label in [
    ('VIX_pct_change_5d', 'PnL_fwd_pts_05', 'FWD_05'),
    ('VIX_pct_change_25d', 'PnL_fwd_pts_25', 'FWD_25'),
    ('VIX_pct_change_50d', 'PnL_fwd_pts_50', 'FWD_50')
]:
    valid = df_q1_2022[[vix_col, pnl_col]].dropna()
    if len(valid) > 10:
        corr = valid[vix_col].corr(valid[pnl_col])
        print(f"   {label}: Corr(Δ%VIX, PnL) = {corr:+.4f}")

print("\nOTROS AÑOS - Correlación entre cambio VIX y PnL:")
for vix_col, pnl_col, label in [
    ('VIX_pct_change_5d', 'PnL_fwd_pts_05', 'FWD_05'),
    ('VIX_pct_change_25d', 'PnL_fwd_pts_25', 'FWD_25'),
    ('VIX_pct_change_50d', 'PnL_fwd_pts_50', 'FWD_50')
]:
    valid = df_others[[vix_col, pnl_col]].dropna()
    if len(valid) > 10:
        corr = valid[vix_col].corr(valid[pnl_col])
        print(f"   {label}: Corr(Δ%VIX, PnL) = {corr:+.4f}")

# ============================================================================
# SECCIÓN 4: ANÁLISIS POR DIRECCIÓN DEL VIX
# ============================================================================
print("\n" + "="*80)
print("SECCIÓN 4: PnL SEGÚN SI VIX SUBE O BAJA")
print("="*80)

# Q4 2022 - cuando VIX cae vs sube
print("\nQ4 2022:")
print("\n--- Cuando VIX CAE en próximos 25 días ---")
vix_falling_q4 = df_q4_2022[df_q4_2022['VIX_change_25d'] < 0]
print(f"Casos: {len(vix_falling_q4)} ({len(vix_falling_q4)/len(df_q4_2022)*100:.1f}%)")
print(f"VIX cambio promedio: {vix_falling_q4['VIX_pct_change_25d'].mean():.2f}%")
print(f"PnL_fwd_pts_25: {vix_falling_q4['PnL_fwd_pts_25'].mean():+.2f}")
print(f"PnL_fwd_pts_50: {vix_falling_q4['PnL_fwd_pts_50'].mean():+.2f}")

print("\n--- Cuando VIX SUBE en próximos 25 días ---")
vix_rising_q4 = df_q4_2022[df_q4_2022['VIX_change_25d'] > 0]
print(f"Casos: {len(vix_rising_q4)} ({len(vix_rising_q4)/len(df_q4_2022)*100:.1f}%)")
print(f"VIX cambio promedio: {vix_rising_q4['VIX_pct_change_25d'].mean():.2f}%")
print(f"PnL_fwd_pts_25: {vix_rising_q4['PnL_fwd_pts_25'].mean():+.2f}")
print(f"PnL_fwd_pts_50: {vix_rising_q4['PnL_fwd_pts_50'].mean():+.2f}")

# Q1 2022 para comparación
print("\n\nQ1 2022:")
print("\n--- Cuando VIX CAE en próximos 25 días ---")
vix_falling_q1 = df_q1_2022[df_q1_2022['VIX_change_25d'] < 0]
print(f"Casos: {len(vix_falling_q1)} ({len(vix_falling_q1)/len(df_q1_2022)*100:.1f}%)")
print(f"VIX cambio promedio: {vix_falling_q1['VIX_pct_change_25d'].mean():.2f}%")
print(f"PnL_fwd_pts_25: {vix_falling_q1['PnL_fwd_pts_25'].mean():+.2f}")
print(f"PnL_fwd_pts_50: {vix_falling_q1['PnL_fwd_pts_50'].mean():+.2f}")

print("\n--- Cuando VIX SUBE en próximos 25 días ---")
vix_rising_q1 = df_q1_2022[df_q1_2022['VIX_change_25d'] > 0]
print(f"Casos: {len(vix_rising_q1)} ({len(vix_rising_q1)/len(df_q1_2022)*100:.1f}%)")
print(f"VIX cambio promedio: {vix_rising_q1['VIX_pct_change_25d'].mean():.2f}%")
print(f"PnL_fwd_pts_25: {vix_rising_q1['PnL_fwd_pts_25'].mean():+.2f}")
print(f"PnL_fwd_pts_50: {vix_rising_q1['PnL_fwd_pts_50'].mean():+.2f}")

# ============================================================================
# SECCIÓN 5: SEGMENTACIÓN POR NIVEL INICIAL DE VIX Y CAMBIO
# ============================================================================
print("\n" + "="*80)
print("SECCIÓN 5: VIX ALTO + CAÍDA = ¿DISASTER?")
print("="*80)

# Q4 2022: VIX alto (>25) y luego cae
print("\nQ4 2022 - VIX ALTO (>25) en apertura:")
high_vix_q4 = df_q4_2022[df_q4_2022['VIX'] > 25]
print(f"Total casos: {len(high_vix_q4)}")

high_vix_falling = high_vix_q4[high_vix_q4['VIX_change_25d'] < 0]
print(f"\n   → Y luego VIX CAE: {len(high_vix_falling)} ({len(high_vix_falling)/len(high_vix_q4)*100:.1f}%)")
print(f"      VIX inicial: {high_vix_falling['VIX'].mean():.2f}")
print(f"      VIX cambio: {high_vix_falling['VIX_pct_change_25d'].mean():.2f}%")
print(f"      PnL_25: {high_vix_falling['PnL_fwd_pts_25'].mean():+.2f} 🔴")
print(f"      PnL_50: {high_vix_falling['PnL_fwd_pts_50'].mean():+.2f}")

high_vix_rising = high_vix_q4[high_vix_q4['VIX_change_25d'] > 0]
print(f"\n   → Y luego VIX SUBE: {len(high_vix_rising)} ({len(high_vix_rising)/len(high_vix_q4)*100:.1f}%)")
print(f"      VIX inicial: {high_vix_rising['VIX'].mean():.2f}")
print(f"      VIX cambio: {high_vix_rising['VIX_pct_change_25d'].mean():.2f}%")
print(f"      PnL_25: {high_vix_rising['PnL_fwd_pts_25'].mean():+.2f}")
print(f"      PnL_50: {high_vix_rising['PnL_fwd_pts_50'].mean():+.2f}")

# Comparar con Q1
print("\n\nQ1 2022 - VIX ALTO (>25) en apertura:")
high_vix_q1 = df_q1_2022[df_q1_2022['VIX'] > 25]
print(f"Total casos: {len(high_vix_q1)}")

high_vix_falling_q1 = high_vix_q1[high_vix_q1['VIX_change_25d'] < 0]
print(f"\n   → Y luego VIX CAE: {len(high_vix_falling_q1)} ({len(high_vix_falling_q1)/len(high_vix_q1)*100:.1f}%)")
print(f"      VIX inicial: {high_vix_falling_q1['VIX'].mean():.2f}")
print(f"      VIX cambio: {high_vix_falling_q1['VIX_pct_change_25d'].mean():.2f}%")
print(f"      PnL_25: {high_vix_falling_q1['PnL_fwd_pts_25'].mean():+.2f}")
print(f"      PnL_50: {high_vix_falling_q1['PnL_fwd_pts_50'].mean():+.2f}")

high_vix_rising_q1 = high_vix_q1[high_vix_q1['VIX_change_25d'] > 0]
print(f"\n   → Y luego VIX SUBE: {len(high_vix_rising_q1)} ({len(high_vix_rising_q1)/len(high_vix_q1)*100:.1f}%)")
print(f"      VIX inicial: {high_vix_rising_q1['VIX'].mean():.2f}")
print(f"      VIX cambio: {high_vix_rising_q1['VIX_pct_change_25d'].mean():.2f}%")
print(f"      PnL_25: {high_vix_rising_q1['PnL_fwd_pts_25'].mean():+.2f}")
print(f"      PnL_50: {high_vix_rising_q1['PnL_fwd_pts_50'].mean():+.2f}")

# ============================================================================
# SECCIÓN 6: TIMELINE - VIX Y PnL A LO LARGO DE Q4
# ============================================================================
print("\n" + "="*80)
print("SECCIÓN 6: EVOLUCIÓN TEMPORAL Q4 2022")
print("="*80)

# Dividir Q4 en 3 partes
q4_len = len(df_q4_2022)
tercio = q4_len // 3

df_q4_early = df_q4_2022.iloc[:tercio]
df_q4_mid = df_q4_2022.iloc[tercio:2*tercio]
df_q4_late = df_q4_2022.iloc[2*tercio:]

print("\nInicio Q4 (Oct):")
print(f"   VIX promedio: {df_q4_early['VIX'].mean():.2f}")
print(f"   PnL_25: {df_q4_early['PnL_fwd_pts_25'].mean():+.2f}")
print(f"   VIX cambio 25d: {df_q4_early['VIX_pct_change_25d'].mean():+.2f}%")

print("\nMedio Q4 (Nov):")
print(f"   VIX promedio: {df_q4_mid['VIX'].mean():.2f}")
print(f"   PnL_25: {df_q4_mid['PnL_fwd_pts_25'].mean():+.2f}")
print(f"   VIX cambio 25d: {df_q4_mid['VIX_pct_change_25d'].mean():+.2f}%")

print("\nFinal Q4 (Dec):")
print(f"   VIX promedio: {df_q4_late['VIX'].mean():.2f}")
print(f"   PnL_25: {df_q4_late['PnL_fwd_pts_25'].mean():+.2f}")
print(f"   VIX cambio 25d: {df_q4_late['VIX_pct_change_25d'].mean():+.2f}%")

# ============================================================================
# SECCIÓN 7: VISUALIZACIONES
# ============================================================================
print("\n" + "="*80)
print("SECCIÓN 7: GENERANDO VISUALIZACIONES")
print("="*80)

fig = plt.figure(figsize=(24, 18))

# 1. VIX Timeline Q4 2022 con forward
ax1 = plt.subplot(3, 3, 1)
if len(df_q4_2022) > 0:
    ax1.plot(df_q4_2022['dia'], df_q4_2022['VIX'],
             color='red', linewidth=2, label='VIX Actual', marker='o', markersize=2)
    ax1.plot(df_q4_2022['dia'], df_q4_2022['VIX_future_25d'],
             color='blue', linewidth=2, alpha=0.7, label='VIX +25d', linestyle='--')
    ax1.fill_between(df_q4_2022['dia'], df_q4_2022['VIX'],
                     df_q4_2022['VIX_future_25d'], alpha=0.2)
    ax1.set_title('Q4 2022: VIX Actual vs VIX +25 días', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('VIX')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

# 2. Cambio % VIX 25d - Q4 vs Q1
ax2 = plt.subplot(3, 3, 2)
data_changes = [
    df_q4_2022['VIX_pct_change_25d'].dropna(),
    df_q1_2022['VIX_pct_change_25d'].dropna()
]
bp = ax2.boxplot(data_changes, labels=['Q4 2022', 'Q1 2022'], patch_artist=True)
bp['boxes'][0].set_facecolor('red')
bp['boxes'][1].set_facecolor('green')
for patch in bp['boxes']:
    patch.set_alpha(0.6)
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax2.set_title('Cambio % VIX en 25 días: Q4 vs Q1 2022', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cambio % VIX')
ax2.grid(axis='y', alpha=0.3)

# 3. Scatter: VIX change vs PnL_25 Q4
ax3 = plt.subplot(3, 3, 3)
valid = df_q4_2022[['VIX_pct_change_25d', 'PnL_fwd_pts_25']].dropna()
if len(valid) > 0:
    scatter = ax3.scatter(valid['VIX_pct_change_25d'], valid['PnL_fwd_pts_25'],
                          alpha=0.6, s=50, c=valid['VIX_pct_change_25d'], cmap='RdYlGn_r')
    z = np.polyfit(valid['VIX_pct_change_25d'], valid['PnL_fwd_pts_25'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid['VIX_pct_change_25d'].min(),
                         valid['VIX_pct_change_25d'].max(), 100)
    ax3.plot(x_line, p(x_line), "b--", alpha=0.8, linewidth=2,
             label=f'y={z[0]:.2f}x+{z[1]:.2f}')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_title('Q4 2022: Cambio VIX vs PnL_25', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Cambio % VIX en 25 días')
    ax3.set_ylabel('PnL_fwd_pts_25')
    ax3.legend()
    plt.colorbar(scatter, ax=ax3, label='Cambio % VIX')
    ax3.grid(alpha=0.3)

# 4. PnL cuando VIX cae vs sube - Q4
ax4 = plt.subplot(3, 3, 4)
categories = ['VIX Cae', 'VIX Sube']
pnl_25_vals = [
    vix_falling_q4['PnL_fwd_pts_25'].mean(),
    vix_rising_q4['PnL_fwd_pts_25'].mean()
]
pnl_50_vals = [
    vix_falling_q4['PnL_fwd_pts_50'].mean(),
    vix_rising_q4['PnL_fwd_pts_50'].mean()
]
x = np.arange(len(categories))
width = 0.35
bars1 = ax4.bar(x - width/2, pnl_25_vals, width, label='FWD_25',
                color=['red', 'green'], alpha=0.7)
bars2 = ax4.bar(x + width/2, pnl_50_vals, width, label='FWD_50',
                color=['darkred', 'darkgreen'], alpha=0.7)
ax4.set_xticks(x)
ax4.set_xticklabels(categories)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax4.set_title('Q4 2022: PnL según dirección VIX', fontsize=12, fontweight='bold')
ax4.set_ylabel('PnL Promedio')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center',
                va='bottom' if height > 0 else 'top', fontsize=9)

# 5. Comparación Q1 vs Q4
ax5 = plt.subplot(3, 3, 5)
x = np.arange(2)
width = 0.2
q1_falling = [vix_falling_q1['PnL_fwd_pts_25'].mean(),
              vix_falling_q1['PnL_fwd_pts_50'].mean()]
q1_rising = [vix_rising_q1['PnL_fwd_pts_25'].mean(),
             vix_rising_q1['PnL_fwd_pts_50'].mean()]
q4_falling = [vix_falling_q4['PnL_fwd_pts_25'].mean(),
              vix_falling_q4['PnL_fwd_pts_50'].mean()]
q4_rising = [vix_rising_q4['PnL_fwd_pts_25'].mean(),
             vix_rising_q4['PnL_fwd_pts_50'].mean()]

ax5.bar(x - 1.5*width, q1_falling, width, label='Q1 VIX↓', color='lightcoral', alpha=0.8)
ax5.bar(x - 0.5*width, q1_rising, width, label='Q1 VIX↑', color='lightgreen', alpha=0.8)
ax5.bar(x + 0.5*width, q4_falling, width, label='Q4 VIX↓', color='darkred', alpha=0.8)
ax5.bar(x + 1.5*width, q4_rising, width, label='Q4 VIX↑', color='darkgreen', alpha=0.8)
ax5.set_xticks(x)
ax5.set_xticklabels(['FWD_25', 'FWD_50'])
ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax5.set_title('PnL: Q1 vs Q4 según dirección VIX', fontsize=12, fontweight='bold')
ax5.set_ylabel('PnL Promedio')
ax5.legend(fontsize=8)
ax5.grid(axis='y', alpha=0.3)

# 6. Frecuencia de caída VIX
ax6 = plt.subplot(3, 3, 6)
freq_data = df_vix_comparison[['Dataset', 'VIX_falling_25d_pct']].dropna()
if len(freq_data) > 0:
    bars = ax6.bar(freq_data['Dataset'], freq_data['VIX_falling_25d_pct'],
                   color=['red', 'orange', 'blue'], alpha=0.7)
    ax6.axhline(y=50, color='black', linestyle='--', linewidth=1, label='50%')
    ax6.set_title('% de veces que VIX cayó en 25d', fontsize=12, fontweight='bold')
    ax6.set_ylabel('% de casos')
    ax6.set_ylim(0, 100)
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

# 7. VIX alto + caída scenario
ax7 = plt.subplot(3, 3, 7)
scenarios_q4 = ['VIX Alto\n+ Cae', 'VIX Alto\n+ Sube']
pnl_scenarios_q4 = [
    high_vix_falling['PnL_fwd_pts_25'].mean(),
    high_vix_rising['PnL_fwd_pts_25'].mean()
]
colors_scen = ['darkred' if x < 0 else 'green' for x in pnl_scenarios_q4]
bars = ax7.bar(scenarios_q4, pnl_scenarios_q4, color=colors_scen, alpha=0.7, edgecolor='black', linewidth=2)
ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax7.set_title('Q4 2022: VIX>25 en apertura', fontsize=12, fontweight='bold')
ax7.set_ylabel('PnL_fwd_pts_25')
ax7.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center',
            va='bottom' if height > 0 else 'top', fontsize=11, fontweight='bold')

# 8. VIX alto + caída Q1 vs Q4
ax8 = plt.subplot(3, 3, 8)
scenarios = ['Q1\nVIX Alto+Cae', 'Q4\nVIX Alto+Cae', 'Q1\nVIX Alto+Sube', 'Q4\nVIX Alto+Sube']
pnl_comparison = [
    high_vix_falling_q1['PnL_fwd_pts_25'].mean(),
    high_vix_falling['PnL_fwd_pts_25'].mean(),
    high_vix_rising_q1['PnL_fwd_pts_25'].mean(),
    high_vix_rising['PnL_fwd_pts_25'].mean()
]
colors = ['orange', 'darkred', 'lightgreen', 'green']
bars = ax8.bar(scenarios, pnl_comparison, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax8.set_title('VIX>25: Q1 vs Q4 2022', fontsize=12, fontweight='bold')
ax8.set_ylabel('PnL_fwd_pts_25')
ax8.grid(axis='y', alpha=0.3)
ax8.tick_params(axis='x', rotation=45)
for bar in bars:
    height = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0f}', ha='center',
            va='bottom' if height > 0 else 'top', fontsize=9, fontweight='bold')

# 9. Timeline PnL_25 Q4
ax9 = plt.subplot(3, 3, 9)
if len(df_q4_2022) > 0:
    cumulative_pnl = df_q4_2022['PnL_fwd_pts_25'].cumsum()
    ax9.plot(df_q4_2022['dia'], cumulative_pnl,
             color='red', linewidth=2, label='PnL Acumulado')
    ax9.fill_between(df_q4_2022['dia'], 0, cumulative_pnl, alpha=0.3, color='red')
    ax9.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax9.set_title('Q4 2022: PnL_25 Acumulado', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Fecha')
    ax9.set_ylabel('PnL Acumulado')
    ax9.legend()
    ax9.grid(alpha=0.3)
    ax9.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('q4_2022_vix_drop_hypothesis.png', dpi=300, bbox_inches='tight')
print("   ✓ q4_2022_vix_drop_hypothesis.png")

# ============================================================================
# EXPORTAR
# ============================================================================
df_vix_comparison.to_csv('vix_change_comparison.csv', index=False)
print("   ✓ vix_change_comparison.csv")

# ============================================================================
# DIAGNÓSTICO Y CONCLUSIONES
# ============================================================================
print("\n" + "="*80)
print("DIAGNÓSTICO FINAL")
print("="*80)

corr_q4_vix_pnl = df_q4_2022[['VIX_pct_change_25d', 'PnL_fwd_pts_25']].dropna().corr().iloc[0,1]
corr_q1_vix_pnl = df_q1_2022[['VIX_pct_change_25d', 'PnL_fwd_pts_25']].dropna().corr().iloc[0,1]

print(f"\n1. CORRELACIÓN Δ%VIX vs PnL_25:")
print(f"   Q4 2022: {corr_q4_vix_pnl:+.4f}")
print(f"   Q1 2022: {corr_q1_vix_pnl:+.4f}")
print(f"   → {'POSITIVA' if corr_q4_vix_pnl > 0 else 'NEGATIVA'} en Q4")

vix_fell_pct_q4 = (df_q4_2022['VIX_change_25d'] < 0).sum() / len(df_q4_2022['VIX_change_25d'].dropna()) * 100
vix_fell_pct_q1 = (df_q1_2022['VIX_change_25d'] < 0).sum() / len(df_q1_2022['VIX_change_25d'].dropna()) * 100

print(f"\n2. FRECUENCIA CAÍDA VIX:")
print(f"   Q4 2022: {vix_fell_pct_q4:.1f}% de casos VIX cayó")
print(f"   Q1 2022: {vix_fell_pct_q1:.1f}% de casos VIX cayó")

print(f"\n3. IMPACTO EN PnL:")
print(f"   Q4 cuando VIX cae: PnL_25 = {vix_falling_q4['PnL_fwd_pts_25'].mean():+.2f}")
print(f"   Q4 cuando VIX sube: PnL_25 = {vix_rising_q4['PnL_fwd_pts_25'].mean():+.2f}")
print(f"   Δ = {vix_rising_q4['PnL_fwd_pts_25'].mean() - vix_falling_q4['PnL_fwd_pts_25'].mean():+.2f} pts")

print(f"\n4. ESCENARIO CRÍTICO (VIX >25 + CAÍDA):")
print(f"   Q4 2022: PnL_25 = {high_vix_falling['PnL_fwd_pts_25'].mean():+.2f} 🔴")
print(f"   Q1 2022: PnL_25 = {high_vix_falling_q1['PnL_fwd_pts_25'].mean():+.2f}")
print(f"   Δ = {high_vix_falling['PnL_fwd_pts_25'].mean() - high_vix_falling_q1['PnL_fwd_pts_25'].mean():+.2f} pts")

print("\n" + "="*80)
print("HIPÓTESIS: ", end="")
if abs(corr_q4_vix_pnl) > 0.3 and vix_falling_q4['PnL_fwd_pts_25'].mean() < -50:
    print("✅ CONFIRMADA")
    print("\nLa caída del VIX EXPLICA las pérdidas masivas en Q4 2022.")
    print("Cuando VIX alto cae → pérdidas catastróficas.")
elif abs(vix_rising_q4['PnL_fwd_pts_25'].mean() - vix_falling_q4['PnL_fwd_pts_25'].mean()) > 30:
    print("✅ PARCIALMENTE CONFIRMADA")
    print("\nLa dirección del VIX tiene IMPACTO SIGNIFICATIVO, pero no explica todo.")
else:
    print("❌ NO CONFIRMADA o PARCIAL")
    print("\nLa caída del VIX no es el único factor. Otros elementos en juego.")
print("="*80)
