#!/usr/bin/env python3
"""
ANÁLISIS COMPLETO: DRIVERS DE PÉRDIDAS
Sistema completo de identificación de predictores tempranos
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, ttest_ind
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" ANÁLISIS COMPLETO - DRIVERS DE PÉRDIDAS EN T+0")
print("="*80)
print()

# Cargar datos
df = pd.read_csv('combined_mediana.csv')
print(f"✓ Datos cargados: {len(df):,} registros\n")

# Parsear DTE
df[['DTE1', 'DTE2']] = df['DTE1/DTE2'].str.split('/', expand=True)
df['DTE1'] = pd.to_numeric(df['DTE1'], errors='coerce')
df['DTE2'] = pd.to_numeric(df['DTE2'], errors='coerce')

# ============================================================================
# GENERAR TODOS LOS INDICADORES DERIVADOS
# ============================================================================

print("Generando indicadores derivados...")

# IV
df['iv_spread_total'] = df['iv_k1'] - df['iv_k3']
df['iv_ratio_k1_k2'] = df['iv_k1'] / (df['iv_k2'] + 0.0001)
df['iv_avg'] = (df['iv_k1'] + df['iv_k2'] + df['iv_k3']) / 3
df['iv_std'] = df[['iv_k1', 'iv_k2', 'iv_k3']].std(axis=1)
df['iv_cv'] = df['iv_std'] / (df['iv_avg'] + 0.0001)
df['iv_weighted_dte'] = (df['iv_k1'] * df['DTE1'] + df['iv_k3'] * df['DTE1']) / (2 * df['DTE1'] + 0.0001)
df['iv_spread_per_dte'] = df['iv_spread_total'] / (df['DTE1'] + 1)
df['iv_skew'] = (df['iv_k1'] - df['iv_k3']) / (df['iv_k2'] + 0.0001)

# Griegas
df['theta_delta_ratio'] = df['theta_total'] / (df['delta_total'].abs() + 0.0001)
df['theta_per_dte1'] = df['theta_total'] / (df['DTE1'] + 1)
df['theta_delta_product'] = df['theta_total'] * df['delta_total'].abs()

# Precio/Crédito
df['total_premium'] = df['price_mid_short1'] + df['price_mid_long2'] + df['price_mid_short3']
df['credit_per_premium'] = df['net_credit'] / (df['total_premium'] + 0.0001)
df['short_to_long_ratio'] = (df['price_mid_short1'] + df['price_mid_short3']) / (df['price_mid_long2'] + 0.0001)

# Strikes
df['strike_spread_total'] = df['k3'] - df['k1']
df['strike_width_normalized'] = (df['k3'] - df['k1']) / (df['SPX'] + 0.0001)
df['k1_otm'] = (df['SPX'] - df['k1']) / (df['SPX'] + 0.0001)
df['k2_otm'] = (df['SPX'] - df['k2']) / (df['SPX'] + 0.0001)
df['k2_center_ratio'] = (df['k2'] - df['k1']) / (df['k3'] - df['k1'] + 0.0001)

# Compuestos
df['iv_theta_product'] = df['iv_avg'] * df['theta_total'].abs()
df['iv_delta_product'] = df['iv_avg'] * df['delta_total'].abs()
df['iv_dte_product'] = df['iv_avg'] * df['DTE1']
df['price_iv_total'] = (df['price_mid_short1'] * df['iv_k1'] +
                         df['price_mid_long2'] * df['iv_k2'] +
                         df['price_mid_short3'] * df['iv_k3'])

# Eficiencia/Riesgo
df['theta_per_credit'] = df['theta_total'] / (df['net_credit'].abs() + 0.0001)
df['pnldv_per_credit'] = df['PnLDV'] / (df['net_credit'].abs() + 0.0001)
df['risk_reward_ratio'] = df['PnLDV'].abs() / (df['net_credit'].abs() + 0.0001)

# DTE
df['dte_ratio'] = df['DTE2'] / (df['DTE1'] + 0.0001)
df['dte_diff'] = df['DTE2'] - df['DTE1']

# Creativos/Complejos
df['iv_theta_delta_combo'] = (df['iv_avg'] * df['theta_total'].abs()) / (df['delta_total'].abs() + 0.0001)
df['strike_iv_efficiency'] = df['strike_spread_total'] / (df['iv_avg'] * 100 + 0.0001)
df['dte_theta_efficiency'] = df['DTE1'] * df['theta_total'].abs()
df['iv_time_weighted'] = (df['iv_k1'] * df['DTE1'] + df['iv_k3'] * df['DTE1']) / (df['iv_k2'] * df['DTE2'] + 0.0001)
df['theta_delta_iv_adjusted'] = (df['theta_total'] / (df['delta_total'].abs() + 0.0001)) * df['iv_avg']
df['structure_stress'] = (df['iv_spread_total'].abs() * df['delta_total'].abs()) / (df['theta_total'].abs() + 0.0001)
df['structure_balance'] = (df['price_mid_short1'] + df['price_mid_short3']) / (df['price_mid_long2'] + 0.0001)

print(f"✓ Indicadores creados\n")

# ============================================================================
# IDENTIFICAR WINNERS VS LOSERS
# ============================================================================

df['is_loser_50'] = df['PnL_fwd_pts_50'] < -20
df['is_winner_50'] = df['PnL_fwd_pts_50'] > 80

n_losers = df['is_loser_50'].sum()
n_winners = df['is_winner_50'].sum()

print(f"📊 Clasificación:")
print(f"   Losers (PnL_50 < -20): {n_losers:,} ({n_losers/len(df)*100:.1f}%)")
print(f"   Winners (PnL_50 > 80): {n_winners:,} ({n_winners/len(df)*100:.1f}%)")
print()

# ============================================================================
# ANÁLISIS DE CORRELACIONES
# ============================================================================

print("="*80)
print(" ANÁLISIS DE CORRELACIONES - PREDICTORES DE PÉRDIDAS")
print("="*80)
print()

# Lista de todos los indicadores
all_indicators = [
    # Base
    'delta_total', 'theta_total', 'iv_k1', 'iv_k2', 'iv_k3',
    'price_mid_short1', 'price_mid_long2', 'price_mid_short3', 'net_credit',
    'k1', 'k2', 'k3', 'SPX', 'BQI_ABS', 'BQI_V2_ABS', 'PnLDV', 'EarScore',
    'RATIO_BATMAN', 'RATIO_UEL_EARS', 'DTE1', 'DTE2', 'FF_ATM', 'FF_BAT',
    # Derivados
    'iv_spread_total', 'iv_ratio_k1_k2', 'iv_avg', 'iv_std', 'iv_cv',
    'iv_weighted_dte', 'iv_spread_per_dte', 'iv_skew',
    'theta_delta_ratio', 'theta_per_dte1', 'theta_delta_product',
    'total_premium', 'credit_per_premium', 'short_to_long_ratio',
    'strike_spread_total', 'strike_width_normalized', 'k1_otm', 'k2_otm', 'k2_center_ratio',
    'iv_theta_product', 'iv_delta_product', 'iv_dte_product', 'price_iv_total',
    'theta_per_credit', 'pnldv_per_credit', 'risk_reward_ratio',
    'dte_ratio', 'dte_diff',
    'iv_theta_delta_combo', 'strike_iv_efficiency', 'dte_theta_efficiency',
    'iv_time_weighted', 'theta_delta_iv_adjusted', 'structure_stress', 'structure_balance',
]

# Filtrar solo indicadores existentes
all_indicators = [ind for ind in all_indicators if ind in df.columns]

# Limpiar datos
df_clean = df[all_indicators + ['PnL_fwd_pts_25', 'PnL_fwd_pts_50', 'is_loser_50', 'is_winner_50']].copy()
df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()

print(f"✓ Datos limpios: {len(df_clean):,} registros\n")

# Calcular correlaciones con PnL_fwd_pts_50
correlations = []

for indicator in all_indicators:
    try:
        pearson_r, pearson_p = pearsonr(df_clean[indicator], df_clean['PnL_fwd_pts_50'])
        spearman_r, spearman_p = spearmanr(df_clean[indicator], df_clean['PnL_fwd_pts_50'])

        correlations.append({
            'Indicator': indicator,
            'Pearson_r': pearson_r,
            'Pearson_p': pearson_p,
            'Spearman_r': spearman_r,
            'Spearman_p': spearman_p,
            'Abs_Corr': (abs(pearson_r) + abs(spearman_r)) / 2
        })
    except:
        pass

corr_df = pd.DataFrame(correlations).sort_values('Abs_Corr', ascending=False)

print("🏆 TOP 40 INDICADORES MÁS CORRELACIONADOS CON PnL_fwd_pts_50:")
print("="*80)
print(corr_df.head(40).to_string(index=False))
print()

# ============================================================================
# COMPARACIÓN WINNERS VS LOSERS
# ============================================================================

print("="*80)
print(" COMPARACIÓN WINNERS VS LOSERS - TOP 30 INDICADORES")
print("="*80)
print()

top_30_indicators = corr_df.head(30)['Indicator'].tolist()

comparison = []
for indicator in top_30_indicators:
    winners = df_clean[df_clean['is_winner_50']][indicator]
    losers = df_clean[df_clean['is_loser_50']][indicator]

    if len(winners) > 10 and len(losers) > 10:
        t_stat, p_val = ttest_ind(winners, losers, equal_var=False)

        comparison.append({
            'Indicator': indicator,
            'Winners_Mean': winners.mean(),
            'Losers_Mean': losers.mean(),
            'Diff': winners.mean() - losers.mean(),
            'Diff_Pct': ((winners.mean() - losers.mean()) / abs(losers.mean()) * 100) if losers.mean() != 0 else np.nan,
            'T_Stat': t_stat,
            'P_Value': p_val,
            'Significant': p_val < 0.001
        })

comp_df = pd.DataFrame(comparison).sort_values('Diff_Pct', key=abs, ascending=False)
print(comp_df.to_string(index=False))
print()

# ============================================================================
# UMBRALES DE PELIGRO
# ============================================================================

print("="*80)
print(" UMBRALES DE ALERTA TEMPRANA - TOP 20 INDICADORES")
print("="*80)
print()

thresholds = []
for indicator in corr_df.head(20)['Indicator'].tolist():
    losers = df_clean[df_clean['is_loser_50']][indicator]
    winners = df_clean[df_clean['is_winner_50']][indicator]

    if len(losers) > 10 and len(winners) > 10:
        # Percentiles
        loser_p75 = losers.quantile(0.75)
        loser_p90 = losers.quantile(0.90)
        winner_p25 = winners.quantile(0.25)
        winner_p10 = winners.quantile(0.10)

        # Dirección
        direction = 'HIGH' if losers.mean() > winners.mean() else 'LOW'

        thresholds.append({
            'Indicator': indicator,
            'Loser_P75': loser_p75,
            'Winner_P25': winner_p25,
            'Threshold_Danger': loser_p75,
            'Threshold_Safe': winner_p25,
            'Direction': direction,
            'Losers_Avg': losers.mean(),
            'Winners_Avg': winners.mean()
        })

thresh_df = pd.DataFrame(thresholds)
print(thresh_df.to_string(index=False))
print()

# ============================================================================
# GUARDAR RESULTADOS
# ============================================================================

print("="*80)
print(" GUARDANDO RESULTADOS")
print("="*80)
print()

corr_df.to_csv('loss_drivers_correlations_final.csv', index=False)
comp_df.to_csv('loss_drivers_winners_vs_losers.csv', index=False)
thresh_df.to_csv('loss_drivers_danger_thresholds.csv', index=False)

print("✓ loss_drivers_correlations_final.csv")
print("✓ loss_drivers_winners_vs_losers.csv")
print("✓ loss_drivers_danger_thresholds.csv")
print()

print("="*80)
print(" ✨ ANÁLISIS COMPLETADO")
print("="*80)
