#!/usr/bin/env python3
"""
INFRAESTRUCTURA DE BÚSQUEDA DE DRIVERS DE PÉRDIDAS
Análisis exhaustivo para identificar predictores tempranos (T+0) de pérdidas futuras

Objetivo: Encontrar qué factores en el momento inicial predicen FWD PTS negativos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" INFRAESTRUCTURA DE BÚSQUEDA DE DRIVERS DE PÉRDIDAS")
print(" Análisis T+0 → Predicción de pérdidas futuras")
print("="*80)
print()

# 1. CARGA DE DATOS
print("📊 Cargando datos...")
df = pd.read_csv('combined_mediana.csv')
print(f"✓ Dataset cargado: {len(df)} registros")
print()

# 2. IDENTIFICAR WINNERS Y LOSERS
print("🔍 Identificando Winners vs Losers...")
print()

# Definir umbrales de pérdida
df['is_loss_25'] = df['PnL_fwd_pts_25'] < 0
df['is_loss_50'] = df['PnL_fwd_pts_50'] < 0
df['is_severe_loss_25'] = df['PnL_fwd_pts_25'] < -50  # Pérdidas severas
df['is_severe_loss_50'] = df['PnL_fwd_pts_50'] < -50

print(f"📊 Análisis de Pérdidas:")
print(f"   PnL_fwd_pts_25 < 0: {df['is_loss_25'].sum():,} ({df['is_loss_25'].sum()/len(df)*100:.1f}%)")
print(f"   PnL_fwd_pts_25 < -50: {df['is_severe_loss_25'].sum():,} ({df['is_severe_loss_25'].sum()/len(df)*100:.1f}%)")
print(f"   PnL_fwd_pts_50 < 0: {df['is_loss_50'].sum():,} ({df['is_loss_50'].sum()/len(df)*100:.1f}%)")
print(f"   PnL_fwd_pts_50 < -50: {df['is_severe_loss_50'].sum():,} ({df['is_severe_loss_50'].sum()/len(df)*100:.1f}%)")
print()

# 3. CREAR INDICADORES DERIVADOS (T+0)
print("="*80)
print(" GENERACIÓN DE INDICADORES DERIVADOS (T+0)")
print("="*80)
print()

print("🔬 Creando indicadores derivados de datos iniciales...")

# 3.1 INDICADORES DE VOLATILIDAD IMPLÍCITA
print("   → Indicadores de IV...")
df['iv_spread_short_long'] = df['iv_k1'] - df['iv_k2']  # Diferencia IV corto-largo
df['iv_spread_long_far'] = df['iv_k2'] - df['iv_k3']     # Diferencia IV largo-lejos
df['iv_spread_total'] = df['iv_k1'] - df['iv_k3']        # Diferencia IV total
df['iv_ratio_k1_k2'] = df['iv_k1'] / df['iv_k2']         # Ratio IV
df['iv_ratio_k2_k3'] = df['iv_k2'] / df['iv_k3']
df['iv_ratio_k1_k3'] = df['iv_k1'] / df['iv_k3']
df['iv_avg'] = (df['iv_k1'] + df['iv_k2'] + df['iv_k3']) / 3
df['iv_std'] = df[['iv_k1', 'iv_k2', 'iv_k3']].std(axis=1)
df['iv_cv'] = df['iv_std'] / df['iv_avg']  # Coeficiente de variación

# IV ponderadas por precio
df['iv_weighted_price'] = (df['iv_k1'] * df['price_mid_short1'] +
                            df['iv_k2'] * df['price_mid_long2'] +
                            df['iv_k3'] * df['price_mid_short3']) / \
                           (df['price_mid_short1'] + df['price_mid_long2'] + df['price_mid_short3'])

# 3.2 INDICADORES DE GRIEGAS
print("   → Indicadores de Griegas...")
# Parsear DTE
df[['DTE1', 'DTE2']] = df['DTE1/DTE2'].str.split('/', expand=True)
df['DTE1'] = pd.to_numeric(df['DTE1'], errors='coerce')
df['DTE2'] = pd.to_numeric(df['DTE2'], errors='coerce')

df['theta_delta_ratio'] = df['theta_total'] / (df['delta_total'].abs() + 0.0001)
df['theta_per_day'] = df['theta_total'] * 365  # Theta anualizado
df['delta_per_dte1'] = df['delta_total'] / (df['DTE1'] + 1)
df['theta_per_dte1'] = df['theta_total'] / (df['DTE1'] + 1)
df['theta_delta_product'] = df['theta_total'] * df['delta_total'].abs()

# 3.3 INDICADORES DE PRECIO Y CRÉDITO
print("   → Indicadores de Precio...")
df['net_credit_ratio'] = df['net_credit'] / (df['price_mid_long2'] + 0.0001)
df['price_ratio_k1_k2'] = df['price_mid_short1'] / (df['price_mid_long2'] + 0.0001)
df['price_ratio_k2_k3'] = df['price_mid_long2'] / (df['price_mid_short3'] + 0.0001)
df['total_premium'] = df['price_mid_short1'] + df['price_mid_long2'] + df['price_mid_short3']
df['credit_per_total_premium'] = df['net_credit'] / (df['total_premium'] + 0.0001)

# 3.4 INDICADORES DE STRIKES
print("   → Indicadores de Strikes...")
df['strike_spread_k1_k2'] = df['k2'] - df['k1']
df['strike_spread_k2_k3'] = df['k3'] - df['k2']
df['strike_spread_total'] = df['k3'] - df['k1']
df['strike_ratio_k1_k2'] = df['k1'] / (df['k2'] + 0.0001)
df['strike_width_normalized'] = df['strike_spread_total'] / df['SPX']

# Distancia al mercado
df['k1_distance'] = (df['SPX'] - df['k1']) / df['SPX']
df['k2_distance'] = (df['SPX'] - df['k2']) / df['SPX']
df['k3_distance'] = (df['SPX'] - df['k3']) / df['SPX']
df['k1_moneyness'] = df['k1'] / df['SPX']
df['k2_moneyness'] = df['k2'] / df['SPX']
df['k3_moneyness'] = df['k3'] / df['SPX']

# 3.5 INDICADORES COMPUESTOS
print("   → Indicadores Compuestos...")
# IV × Theta
df['iv_theta_product'] = df['iv_avg'] * df['theta_total'].abs()
df['iv_delta_product'] = df['iv_avg'] * df['delta_total'].abs()

# IV × DTE
df['iv_dte_ratio'] = df['iv_avg'] * df['DTE1']
df['iv_spread_per_dte'] = df['iv_spread_total'] / (df['DTE1'] + 1)

# Precio × IV
df['price_iv_k1'] = df['price_mid_short1'] * df['iv_k1']
df['price_iv_k2'] = df['price_mid_long2'] * df['iv_k2']
df['price_iv_k3'] = df['price_mid_short3'] * df['iv_k3']

# 3.6 INDICADORES DE ASIMETRÍA Y SKEW
print("   → Indicadores de Skew...")
df['iv_skew'] = (df['iv_k1'] - df['iv_k3']) / (df['iv_k2'] + 0.0001)
df['price_skew'] = (df['price_mid_short1'] - df['price_mid_short3']) / (df['price_mid_long2'] + 0.0001)

# 3.7 INDICADORES DE EFICIENCIA
print("   → Indicadores de Eficiencia...")
df['theta_per_credit'] = df['theta_total'] / (df['net_credit'].abs() + 0.0001)
df['delta_per_credit'] = df['delta_total'].abs() / (df['net_credit'].abs() + 0.0001)
df['pnldv_per_credit'] = df['PnLDV'] / (df['net_credit'].abs() + 0.0001)

# 3.8 RATIOS COMPLEJOS
print("   → Ratios Complejos...")
df['iv_theta_delta_combo'] = (df['iv_avg'] * df['theta_total'].abs()) / (df['delta_total'].abs() + 0.0001)
df['strike_iv_efficiency'] = df['strike_spread_total'] / (df['iv_avg'] * 100 + 0.0001)
df['dte_theta_efficiency'] = (df['DTE1'] * df['theta_total'].abs())

# 3.9 INDICADORES DE FORMA DE ESTRUCTURA
print("   → Indicadores de Forma...")
df['k2_center_ratio'] = (df['k2'] - df['k1']) / (df['k3'] - df['k1'] + 0.0001)  # Simetría de strikes
df['iv_k2_center_ratio'] = (df['iv_k2'] - df['iv_k1']) / (df['iv_k3'] - df['iv_k1'] + 0.0001)

# 3.10 INDICADORES DE RIESGO
print("   → Indicadores de Riesgo...")
df['risk_reward_ratio'] = df['PnLDV'].abs() / (df['net_credit'].abs() + 0.0001)
df['max_loss_potential'] = df['PnLDV'] / df['net_credit']

# 3.11 INDICADORES DE CALIDAD
print("   → Usando BQI, EarScore, etc...")
# Ya existen: BQI_ABS, BQI_V2_ABS, EarScore, RATIO_BATMAN, RATIO_UEL_EARS

# 3.12 RATIOS CON DTE
print("   → Ratios con DTE...")
df['dte_ratio'] = df['DTE2'] / (df['DTE1'] + 0.0001)
df['dte_diff'] = df['DTE2'] - df['DTE1']
df['dte_avg'] = (df['DTE1'] + df['DTE2']) / 2

# 3.13 IV ponderada por DTE
print("   → IV ponderada por DTE...")
df['iv_weighted_dte'] = (df['iv_k1'] * df['DTE1'] + df['iv_k3'] * df['DTE1']) / (2 * df['DTE1'] + 0.0001)

# 3.14 INDICADORES DE LIQUIDEZ/SPREAD
print("   → Indicadores de Spread...")
# Asumiendo que tenemos bid/ask (si no, omitir)
if 'price_bid_short1' in df.columns:
    df['spread_k1'] = df['price_ask_short1'] - df['price_bid_short1']
    df['spread_k2'] = df['price_ask_long2'] - df['price_bid_long2']
    df['spread_k3'] = df['price_ask_short3'] - df['price_bid_short3']
    df['spread_total'] = df['spread_k1'] + df['spread_k2'] + df['spread_k3']
    df['spread_ratio'] = df['spread_total'] / (df['total_premium'] + 0.0001)

print(f"✓ {len([c for c in df.columns if c.startswith(('iv_', 'theta_', 'delta_', 'price_', 'strike_', 'k1_', 'k2_', 'k3_', 'dte_', 'risk_', 'spread_'))])} indicadores derivados creados")
print()

# 4. LISTA DE TODOS LOS INDICADORES
print("="*80)
print(" COMPILACIÓN DE INDICADORES PARA ANÁLISIS")
print("="*80)
print()

# Indicadores base (T+0)
base_indicators = [
    # Griegas
    'delta_total', 'theta_total',
    # IV
    'iv_k1', 'iv_k2', 'iv_k3',
    # Precios
    'price_mid_short1', 'price_mid_long2', 'price_mid_short3',
    # Crédito
    'net_credit',
    # Strikes
    'k1', 'k2', 'k3',
    # Métricas
    'BQI_ABS', 'BQI_V2_ABS', 'PnLDV', 'EarScore', 'RATIO_BATMAN', 'RATIO_UEL_EARS',
    # DTE
    'DTE1', 'DTE2',
    # SPX
    'SPX',
    # FF
    'FF_ATM', 'FF_BAT',
]

# Indicadores derivados
derived_indicators = [
    # IV
    'iv_spread_short_long', 'iv_spread_long_far', 'iv_spread_total',
    'iv_ratio_k1_k2', 'iv_ratio_k2_k3', 'iv_ratio_k1_k3',
    'iv_avg', 'iv_std', 'iv_cv', 'iv_weighted_price',
    # Griegas
    'theta_delta_ratio', 'theta_per_day', 'delta_per_dte1', 'theta_per_dte1',
    'theta_delta_product',
    # Precio/Crédito
    'net_credit_ratio', 'price_ratio_k1_k2', 'price_ratio_k2_k3',
    'total_premium', 'credit_per_total_premium',
    # Strikes
    'strike_spread_k1_k2', 'strike_spread_k2_k3', 'strike_spread_total',
    'strike_ratio_k1_k2', 'strike_width_normalized',
    'k1_distance', 'k2_distance', 'k3_distance',
    'k1_moneyness', 'k2_moneyness', 'k3_moneyness',
    # Compuestos
    'iv_theta_product', 'iv_delta_product', 'iv_dte_ratio', 'iv_spread_per_dte',
    'price_iv_k1', 'price_iv_k2', 'price_iv_k3',
    # Skew
    'iv_skew', 'price_skew',
    # Eficiencia
    'theta_per_credit', 'delta_per_credit', 'pnldv_per_credit',
    # Ratios complejos
    'iv_theta_delta_combo', 'strike_iv_efficiency', 'dte_theta_efficiency',
    # Forma
    'k2_center_ratio', 'iv_k2_center_ratio',
    # Riesgo
    'risk_reward_ratio', 'max_loss_potential',
    # DTE
    'dte_ratio', 'dte_diff', 'dte_avg',
    # IV ponderada
    'iv_weighted_dte',
]

# Filtrar indicadores que existen
all_indicators = [ind for ind in base_indicators + derived_indicators if ind in df.columns]

print(f"📊 Total de indicadores para análisis: {len(all_indicators)}")
print()

# 5. LIMPIEZA DE DATOS
print("🧹 Limpiando datos...")
analysis_columns = all_indicators + ['PnL_fwd_pts_25', 'PnL_fwd_pts_50',
                                      'is_loss_25', 'is_loss_50',
                                      'is_severe_loss_25', 'is_severe_loss_50']
df_clean = df[analysis_columns].copy()
df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

# Contar NaN antes
nan_before = df_clean.isnull().sum().sum()
df_clean = df_clean.dropna()
nan_after = 0

print(f"✓ NaN eliminados: {nan_before:,}")
print(f"✓ Datos limpios: {len(df_clean):,} registros ({len(df_clean)/len(df)*100:.1f}%)")
print()

# 6. ANÁLISIS DE CORRELACIONES - PÉRDIDAS EN 25%
print("="*80)
print(" ANÁLISIS DE CORRELACIONES - PÉRDIDAS EN FWD PTS 25%")
print("="*80)
print()

target_25 = 'PnL_fwd_pts_25'
correlations_25 = []

print(f"🎯 Calculando correlaciones con {target_25}...")
for indicator in all_indicators:
    try:
        # Pearson
        pearson_corr, pearson_pval = pearsonr(df_clean[indicator], df_clean[target_25])
        # Spearman
        spearman_corr, spearman_pval = spearmanr(df_clean[indicator], df_clean[target_25])

        correlations_25.append({
            'Indicator': indicator,
            'Pearson_r': pearson_corr,
            'Pearson_pval': pearson_pval,
            'Spearman_r': spearman_corr,
            'Spearman_pval': spearman_pval,
            'Abs_Pearson': abs(pearson_corr),
            'Abs_Spearman': abs(spearman_corr),
            'Avg_Abs_Corr': (abs(pearson_corr) + abs(spearman_corr)) / 2
        })
    except Exception as e:
        pass

corr_df_25 = pd.DataFrame(correlations_25).sort_values('Avg_Abs_Corr', ascending=False)

print("\n🏆 TOP 30 INDICADORES MÁS CORRELACIONADOS (PnL_fwd_pts_25):")
print(corr_df_25.head(30).to_string(index=False))
print()

# 7. ANÁLISIS DE CORRELACIONES - PÉRDIDAS EN 50%
print("="*80)
print(" ANÁLISIS DE CORRELACIONES - PÉRDIDAS EN FWD PTS 50%")
print("="*80)
print()

target_50 = 'PnL_fwd_pts_50'
correlations_50 = []

print(f"🎯 Calculando correlaciones con {target_50}...")
for indicator in all_indicators:
    try:
        pearson_corr, pearson_pval = pearsonr(df_clean[indicator], df_clean[target_50])
        spearman_corr, spearman_pval = spearmanr(df_clean[indicator], df_clean[target_50])

        correlations_50.append({
            'Indicator': indicator,
            'Pearson_r': pearson_corr,
            'Pearson_pval': pearson_pval,
            'Spearman_r': spearman_corr,
            'Spearman_pval': spearman_pval,
            'Abs_Pearson': abs(pearson_corr),
            'Abs_Spearman': abs(spearman_corr),
            'Avg_Abs_Corr': (abs(pearson_corr) + abs(spearman_corr)) / 2
        })
    except Exception as e:
        pass

corr_df_50 = pd.DataFrame(correlations_50).sort_values('Avg_Abs_Corr', ascending=False)

print("\n🏆 TOP 30 INDICADORES MÁS CORRELACIONADOS (PnL_fwd_pts_50):")
print(corr_df_50.head(30).to_string(index=False))
print()

# Continúa en siguiente parte...
print("="*80)
print(" Guardando resultados intermedios...")
print("="*80)
print()

# Guardar correlaciones
corr_df_25.to_csv('loss_drivers_correlations_25.csv', index=False)
corr_df_50.to_csv('loss_drivers_correlations_50.csv', index=False)
print("✓ Correlaciones guardadas")
print()

print("="*80)
print(" Continuando con análisis detallado...")
print("="*80)
print()
