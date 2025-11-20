#!/usr/bin/env python3
"""
PARTE 2: ANÁLISIS PROFUNDO DE WINNERS VS LOSERS
Comparación detallada y umbrales de alerta temprana
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, ttest_ind
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" PARTE 2: ANÁLISIS WINNERS VS LOSERS")
print("="*80)
print()

# Cargar correlaciones
corr_25 = pd.read_csv('loss_drivers_correlations_25.csv')
corr_50 = pd.read_csv('loss_drivers_correlations_50.csv')

# Cargar datos limpios (reconstruir desde scratch)
df = pd.read_csv('combined_mediana.csv')

# Recrear indicadores derivados (copiar lógica de part1)
exec(open('analisis_loss_drivers_part1.py').read().split('# 6. ANÁLISIS DE CORRELACIONES')[0].split('# 3. CREAR INDICADORES DERIVADOS')[1])

# Definir categorías
df['winner_25'] = df['PnL_fwd_pts_25'] > 50  # Ganadores significativos
df['loser_25'] = df['PnL_fwd_pts_25'] < -20  # Perdedores
df['winner_50'] = df['PnL_fwd_pts_50'] > 80
df['loser_50'] = df['PnL_fwd_pts_50'] < -20

print(f"📊 Clasificación de Trades:")
print(f"   Winners 25%: {df['winner_25'].sum():,} ({df['winner_25'].sum()/len(df)*100:.1f}%)")
print(f"   Losers 25%: {df['loser_25'].sum():,} ({df['loser_25'].sum()/len(df)*100:.1f}%)")
print(f"   Winners 50%: {df['winner_50'].sum():,} ({df['winner_50'].sum()/len(df)*100:.1f}%)")
print(f"   Losers 50%: {df['loser_50'].sum():,} ({df['loser_50'].sum()/len(df)*100:.1f}%)")
print()

# TOP INDICADORES
top_indicators_25 = corr_25.head(20)['Indicator'].tolist()
top_indicators_50 = corr_50.head(20)['Indicator'].tolist()
top_indicators_combined = list(set(top_indicators_25 + top_indicators_50))

print("="*80)
print(" COMPARACIÓN WINNERS VS LOSERS - TOP INDICADORES")
print("="*80)
print()

# Análisis FWD PTS 25%
print("🎯 Análisis FWD PTS 25%:")
print("-" * 80)

comparison_25 = []
for indicator in top_indicators_combined:
    if indicator not in df.columns:
        continue

    winners = df[df['winner_25']][indicator].dropna()
    losers = df[df['loser_25']][indicator].dropna()

    if len(winners) > 0 and len(losers) > 0:
        # T-test
        t_stat, p_value = ttest_ind(winners, losers, equal_var=False)

        comparison_25.append({
            'Indicator': indicator,
            'Winners_Mean': winners.mean(),
            'Winners_Median': winners.median(),
            'Losers_Mean': losers.mean(),
            'Losers_Median': losers.median(),
            'Diff_Mean': winners.mean() - losers.mean(),
            'Diff_Median': winners.median() - losers.median(),
            'Diff_Pct': ((winners.mean() - losers.mean()) / abs(losers.mean()) * 100) if losers.mean() != 0 else np.nan,
            'T_Stat': t_stat,
            'P_Value': p_value,
            'Significant': p_value < 0.001
        })

comp_df_25 = pd.DataFrame(comparison_25).sort_values('Diff_Pct', key=abs, ascending=False)
print(comp_df_25.head(20).to_string(index=False))
print()

# Análisis FWD PTS 50%
print("🎯 Análisis FWD PTS 50%:")
print("-" * 80)

comparison_50 = []
for indicator in top_indicators_combined:
    if indicator not in df.columns:
        continue

    winners = df[df['winner_50']][indicator].dropna()
    losers = df[df['loser_50']][indicator].dropna()

    if len(winners) > 0 and len(losers) > 0:
        t_stat, p_value = ttest_ind(winners, losers, equal_var=False)

        comparison_50.append({
            'Indicator': indicator,
            'Winners_Mean': winners.mean(),
            'Winners_Median': winners.median(),
            'Losers_Mean': losers.mean(),
            'Losers_Median': losers.median(),
            'Diff_Mean': winners.mean() - losers.mean(),
            'Diff_Median': winners.median() - losers.median(),
            'Diff_Pct': ((winners.mean() - losers.mean()) / abs(losers.mean()) * 100) if losers.mean() != 0 else np.nan,
            'T_Stat': t_stat,
            'P_Value': p_value,
            'Significant': p_value < 0.001
        })

comp_df_50 = pd.DataFrame(comparison_50).sort_values('Diff_Pct', key=abs, ascending=False)
print(comp_df_50.head(20).to_string(index=False))
print()

# UMBRALES DE ALERTA
print("="*80)
print(" UMBRALES DE ALERTA TEMPRANA")
print("="*80)
print()

print("🚨 Identificando umbrales críticos para los TOP 10 indicadores...")
print()

# Top 10 para análisis de umbrales
top_10_indicators = corr_50.head(10)['Indicator'].tolist()

thresholds_analysis = []

for indicator in top_10_indicators:
    if indicator not in df.columns:
        continue

    # Percentiles de losers
    losers_data = df[df['loser_50']][indicator].dropna()
    winners_data = df[df['winner_50']][indicator].dropna()

    if len(losers_data) > 10 and len(winners_data) > 10:
        # Umbrales basados en percentiles de losers
        p75_losers = losers_data.quantile(0.75)
        p90_losers = losers_data.quantile(0.90)

        # Percentiles de winners
        p25_winners = winners_data.quantile(0.25)
        p10_winners = winners_data.quantile(0.10)

        # Umbrales críticos
        threshold_danger = p75_losers  # 75% de losers están por encima/debajo
        threshold_safe = p25_winners   # 75% de winners están por encima/debajo

        thresholds_analysis.append({
            'Indicator': indicator,
            'Losers_P75': p75_losers,
            'Losers_P90': p90_losers,
            'Winners_P25': p25_winners,
            'Winners_P10': p10_winners,
            'Threshold_Danger': threshold_danger,
            'Threshold_Safe': threshold_safe,
            'Direction': 'ABOVE' if losers_data.mean() > winners_data.mean() else 'BELOW',
        })

thresh_df = pd.DataFrame(thresholds_analysis)
print(thresh_df.to_string(index=False))
print()

# Guardar resultados
comp_df_25.to_csv('winners_vs_losers_25.csv', index=False)
comp_df_50.to_csv('winners_vs_losers_50.csv', index=False)
thresh_df.to_csv('thresholds_danger_zones.csv', index=False)

print("="*80)
print(" ✓ Análisis completado")
print("="*80)
print()
print("Archivos generados:")
print("  - winners_vs_losers_25.csv")
print("  - winners_vs_losers_50.csv")
print("  - thresholds_danger_zones.csv")
print()
