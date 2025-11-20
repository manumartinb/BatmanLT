#!/usr/bin/env python3
"""
SCRIPT MAESTRO: INFRAESTRUCTURA COMPLETA DE BÚSQUEDA DE DRIVERS DE PÉRDIDAS
Ejecuta análisis iterativo completo para identificar predictores de pérdidas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, ttest_ind
import warnings
warnings.filterwarnings('ignore')

# Configuración
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print(" INFRAESTRUCTURA COMPLETA - BÚSQUEDA DE DRIVERS DE PÉRDIDAS")
print(" Análisis Iterativo T+0 → Predicción de Pérdidas Futuras")
print("="*80)
print()

# =============================================================================
# PASO 1: CARGA Y PREPARACIÓN DE DATOS
# =============================================================================

print("PASO 1: Cargando y preparando datos...")
df = pd.read_csv('combined_mediana.csv')
print(f"✓ Dataset cargado: {len(df):,} registros")
print()

# =============================================================================
# PASO 2: GENERACIÓN MASIVA DE INDICADORES DERIVADOS
# =============================================================================

print("PASO 2: Generando indicadores derivados (T+0)...")

# Parsear DTE
df[['DTE1', 'DTE2']] = df['DTE1/DTE2'].str.split('/', expand=True)
df['DTE1'] = pd.to_numeric(df['DTE1'], errors='coerce')
df['DTE2'] = pd.to_numeric(df['DTE2'], errors='coerce')

# CATEGORÍA 1: VOLATILIDAD IMPLÍCITA (IV)
# -----------------------------------------
print("  → Categoría 1: IV (Volatilidad Implícita)")
df['iv_spread_k1_k2'] = df['iv_k1'] - df['iv_k2']
df['iv_spread_k2_k3'] = df['iv_k2'] - df['iv_k3']
df['iv_spread_total'] = df['iv_k1'] - df['iv_k3']
df['iv_ratio_k1_k2'] = df['iv_k1'] / (df['iv_k2'] + 0.0001)
df['iv_ratio_k2_k3'] = df['iv_k2'] / (df['iv_k3'] + 0.0001)
df['iv_ratio_k1_k3'] = df['iv_k1'] / (df['iv_k3'] + 0.0001)
df['iv_avg'] = (df['iv_k1'] + df['iv_k2'] + df['iv_k3']) / 3
df['iv_std'] = df[['iv_k1', 'iv_k2', 'iv_k3']].std(axis=1)
df['iv_cv'] = df['iv_std'] / (df['iv_avg'] + 0.0001)
df['iv_range'] = df['iv_k1'] - df['iv_k3']
df['iv_normalized_spread'] = (df['iv_k1'] - df['iv_k3']) / (df['iv_avg'] + 0.0001)

# IV ponderadas
df['iv_weighted_price'] = (df['iv_k1'] * df['price_mid_short1'] +
                            df['iv_k2'] * df['price_mid_long2'] +
                            df['iv_k3'] * df['price_mid_short3']) / \
                           (df['price_mid_short1'] + df['price_mid_long2'] + df['price_mid_short3'] + 0.0001)

df['iv_weighted_dte'] = (df['iv_k1'] * df['DTE1'] + df['iv_k3'] * df['DTE1']) / (2 * df['DTE1'] + 0.0001)

# CATEGORÍA 2: GRIEGAS
# ---------------------
print("  → Categoría 2: Griegas (Theta, Delta)")
df['theta_delta_ratio'] = df['theta_total'] / (df['delta_total'].abs() + 0.0001)
df['theta_delta_abs_ratio'] = df['theta_total'].abs() / (df['delta_total'].abs() + 0.0001)
df['theta_per_day'] = df['theta_total'] * 365
df['delta_per_dte1'] = df['delta_total'] / (df['DTE1'] + 1)
df['theta_per_dte1'] = df['theta_total'] / (df['DTE1'] + 1)
df['theta_delta_product'] = df['theta_total'] * df['delta_total'].abs()
df['theta_squared'] = df['theta_total'] ** 2
df['delta_squared'] = df['delta_total'] ** 2

# CATEGORÍA 3: PRECIO Y CRÉDITO
# -------------------------------
print("  → Categoría 3: Precio y Crédito")
df['net_credit_abs'] = df['net_credit'].abs()
df['net_credit_ratio_long'] = df['net_credit'] / (df['price_mid_long2'] + 0.0001)
df['price_ratio_k1_k2'] = df['price_mid_short1'] / (df['price_mid_long2'] + 0.0001)
df['price_ratio_k2_k3'] = df['price_mid_long2'] / (df['price_mid_short3'] + 0.0001)
df['price_ratio_k1_k3'] = df['price_mid_short1'] / (df['price_mid_short3'] + 0.0001)
df['total_premium'] = df['price_mid_short1'] + df['price_mid_long2'] + df['price_mid_short3']
df['credit_per_total_premium'] = df['net_credit'] / (df['total_premium'] + 0.0001)
df['short_premium_total'] = df['price_mid_short1'] + df['price_mid_short3']
df['short_to_long_ratio'] = df['short_premium_total'] / (df['price_mid_long2'] + 0.0001)

# CATEGORÍA 4: STRIKES Y MONEYNESS
# ----------------------------------
print("  → Categoría 4: Strikes y Moneyness")
df['strike_spread_k1_k2'] = df['k2'] - df['k1']
df['strike_spread_k2_k3'] = df['k3'] - df['k2']
df['strike_spread_total'] = df['k3'] - df['k1']
df['strike_ratio_k1_k2'] = df['k1'] / (df['k2'] + 0.0001)
df['strike_width_normalized'] = (df['k3'] - df['k1']) / (df['SPX'] + 0.0001)

# Distancia al mercado (OTM/ITM)
df['k1_otm'] = (df['SPX'] - df['k1']) / (df['SPX'] + 0.0001)
df['k2_otm'] = (df['SPX'] - df['k2']) / (df['SPX'] + 0.0001)
df['k3_otm'] = (df['k3'] - df['SPX']) / (df['SPX'] + 0.0001)  # Nota: k3 es call
df['k1_moneyness'] = df['k1'] / (df['SPX'] + 0.0001)
df['k2_moneyness'] = df['k2'] / (df['SPX'] + 0.0001)
df['k3_moneyness'] = df['k3'] / (df['SPX'] + 0.0001)

# Simetría de estructura
df['k2_center_ratio'] = (df['k2'] - df['k1']) / (df['k3'] - df['k1'] + 0.0001)
df['k2_centered_pct'] = abs(df['k2_center_ratio'] - 0.5)  # Qué tan centrado está k2

# CATEGORÍA 5: INDICADORES COMPUESTOS
# -------------------------------------
print("  → Categoría 5: Indicadores Compuestos")
# IV × Griegas
df['iv_theta_product'] = df['iv_avg'] * df['theta_total'].abs()
df['iv_delta_product'] = df['iv_avg'] * df['delta_total'].abs()
df['iv_theta_ratio'] = df['iv_avg'] / (df['theta_total'].abs() + 0.0001)

# IV × DTE
df['iv_dte_product'] = df['iv_avg'] * df['DTE1']
df['iv_spread_per_dte'] = df['iv_spread_total'] / (df['DTE1'] + 1)
df['iv_spread_per_dte2'] = df['iv_spread_total'] / (df['DTE2'] + 1)

# Precio × IV
df['price_iv_k1'] = df['price_mid_short1'] * df['iv_k1']
df['price_iv_k2'] = df['price_mid_long2'] * df['iv_k2']
df['price_iv_k3'] = df['price_mid_short3'] * df['iv_k3']
df['price_iv_total'] = df['price_iv_k1'] + df['price_iv_k2'] + df['price_iv_k3']

# CATEGORÍA 6: SKEW Y ASIMETRÍA
# -------------------------------
print("  → Categoría 6: Skew y Asimetría")
df['iv_skew_k1_k3'] = (df['iv_k1'] - df['iv_k3']) / (df['iv_k2'] + 0.0001)
df['iv_skew_normalized'] = (df['iv_k1'] - df['iv_k3']) / (df['iv_avg'] + 0.0001)
df['price_skew'] = (df['price_mid_short1'] - df['price_mid_short3']) / (df['price_mid_long2'] + 0.0001)
df['iv_asymmetry'] = (df['iv_k1'] + df['iv_k3']) / (2 * df['iv_k2'] + 0.0001) - 1

# CATEGORÍA 7: EFICIENCIA Y RIESGO/RECOMPENSA
# ---------------------------------------------
print("  → Categoría 7: Eficiencia")
df['theta_per_credit'] = df['theta_total'] / (df['net_credit'].abs() + 0.0001)
df['delta_per_credit'] = df['delta_total'].abs() / (df['net_credit'].abs() + 0.0001)
df['pnldv_per_credit'] = df['PnLDV'] / (df['net_credit'].abs() + 0.0001)
df['risk_reward_ratio'] = df['PnLDV'].abs() / (df['net_credit'].abs() + 0.0001)
df['max_loss_potential'] = df['PnLDV'] / (df['net_credit'] + 0.0001)

# CATEGORÍA 8: DTE Y VENCIMIENTOS
# ---------------------------------
print("  → Categoría 8: DTE")
df['dte_ratio'] = df['DTE2'] / (df['DTE1'] + 0.0001)
df['dte_diff'] = df['DTE2'] - df['DTE1']
df['dte_sum'] = df['DTE1'] + df['DTE2']
df['dte_avg'] = (df['DTE1'] + df['DTE2']) / 2
df['dte1_normalized'] = df['DTE1'] / 365  # En años
df['dte2_normalized'] = df['DTE2'] / 365

# CATEGORÍA 9: RATIOS COMPLEJOS (Creatividad máxima)
# ----------------------------------------------------
print("  → Categoría 9: Ratios Complejos")
df['iv_theta_delta_combo'] = (df['iv_avg'] * df['theta_total'].abs()) / (df['delta_total'].abs() + 0.0001)
df['strike_iv_efficiency'] = df['strike_spread_total'] / (df['iv_avg'] * 100 + 0.0001)
df['dte_theta_efficiency'] = df['DTE1'] * df['theta_total'].abs()
df['credit_strike_efficiency'] = df['net_credit'].abs() / (df['strike_spread_total'] + 0.0001)

# Ratios ponderados por tiempo
df['iv_time_decay'] = df['iv_avg'] * np.sqrt(df['DTE1'] / 365)
df['theta_time_normalized'] = df['theta_total'] * np.sqrt(df['DTE1'])

# Ratios de calidad estructural
df['batman_per_iv'] = df['RATIO_BATMAN'] / (df['iv_avg'] * 100 + 0.0001)
df['earScore_per_theta'] = df['EarScore'] / (df['theta_total'].abs() + 0.0001)

# CATEGORÍA 10: INTERACCIONES ESPECÍFICAS (Usuario mencionó)
# ------------------------------------------------------------
print("  → Categoría 10: Interacciones específicas")
# IV ponderadas por tiempo de vencimiento
df['iv_time_weighted'] = (df['iv_k1'] * df['DTE1'] + df['iv_k3'] * df['DTE1']) / \
                          (df['iv_k2'] * df['DTE2'] + 0.0001)

# Diferencias de IV ponderadas
df['iv_spread_time_weighted'] = (df['iv_k1'] - df['iv_k2']) / np.sqrt(df['DTE1'] + 1) + \
                                  (df['iv_k2'] - df['iv_k3']) / np.sqrt(df['DTE1'] + 1)

# Theta/Delta ponderado por IV
df['theta_delta_iv_adjusted'] = (df['theta_total'] / (df['delta_total'].abs() + 0.0001)) * df['iv_avg']

# Indicador de "stress"
df['structure_stress'] = (df['iv_spread_total'].abs() * df['delta_total'].abs()) / \
                         (df['theta_total'].abs() + 0.0001)

# Indicador de "balance"
df['structure_balance'] = (df['price_mid_short1'] + df['price_mid_short3']) / \
                          (df['price_mid_long2'] + 0.0001)

# TOTAL DE INDICADORES CREADOS
indicator_count = len([col for col in df.columns if any(x in col for x in
                      ['iv_', 'theta_', 'delta_', 'price_', 'strike_', 'credit_',
                       'dte_', 'otm', 'moneyness', 'skew', 'efficiency', 'stress', 'balance'])])

print(f"\n✓ {indicator_count} indicadores derivados creados")
print()

# =============================================================================
# PASO 3: DEFINIR WINNERS Y LOSERS
# =============================================================================

print("PASO 3: Definiendo Winners y Losers...")

# Definiciones
df['is_loss_25'] = df['PnL_fwd_pts_25'] < 0
df['is_loss_50'] = df['PnL_fwd_pts_50'] < 0
df['is_severe_loss_25'] = df['PnL_fwd_pts_25'] < -50
df['is_severe_loss_50'] = df['PnL_fwd_pts_50'] < -50
df['is_winner_25'] = df['PnL_fwd_pts_25'] > 50
df['is_winner_50'] = df['PnL_fwd_pts_50'] > 80

print(f"📊 Distribución:")
print(f"   Pérdidas 25%: {df['is_loss_25'].sum():,} ({df['is_loss_25'].sum()/len(df)*100:.1f}%)")
print(f"   Pérdidas severas 25%: {df['is_severe_loss_25'].sum():,} ({df['is_severe_loss_25'].sum()/len(df)*100:.1f}%)")
print(f"   Pérdidas 50%: {df['is_loss_50'].sum():,} ({df['is_loss_50'].sum()/len(df)*100:.1f}%)")
print(f"   Pérdidas severas 50%: {df['is_severe_loss_50'].sum():,} ({df['is_severe_loss_50'].sum()/len(df)*100:.1f}%)")
print(f"   Winners 25%: {df['is_winner_25'].sum():,} ({df['is_winner_25'].sum()/len(df)*100:.1f}%)")
print(f"   Winners 50%: {df['is_winner_50'].sum():,} ({df['is_winner_50'].sum()/len(df)*100:.1f}%)")
print()

# =============================================================================
# PASO 4: COMPILAR TODOS LOS INDICADORES
# =============================================================================

print("PASO 4: Compilando indicadores para análisis...")

# Indicadores base (T+0)
base_indicators = [
    'delta_total', 'theta_total', 'iv_k1', 'iv_k2', 'iv_k3',
    'price_mid_short1', 'price_mid_long2', 'price_mid_short3',
    'net_credit', 'k1', 'k2', 'k3', 'SPX',
    'BQI_ABS', 'BQI_V2_ABS', 'PnLDV', 'EarScore', 'RATIO_BATMAN', 'RATIO_UEL_EARS',
    'DTE1', 'DTE2', 'FF_ATM', 'FF_BAT',
]

# Todos los indicadores derivados
derived_indicators = [col for col in df.columns if any(x in col for x in
                     ['iv_', 'theta_', 'delta_', 'price_', 'strike_', 'credit_',
                      'dte_', 'otm', 'moneyness', 'skew', 'efficiency', 'stress',
                      'balance', 'ratio_', 'product', 'weighted'])]

all_indicators = list(set([ind for ind in base_indicators + derived_indicators if ind in df.columns]))

print(f"✓ Total indicadores: {len(all_indicators)}")
print()

# Continúa en la ejecución...
print("="*80)
print(" Ejecutando análisis de correlaciones...")
print("="*80)
print()
