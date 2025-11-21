#!/usr/bin/env python3
"""
MACRO ESTUDIO ESTADÍSTICO: Predictores de PnL FWD Desastroso en Ventana 50

Objetivo: Identificar qué variables en T+0 o T+25 pueden predecir un resultado desastroso
         en PnL_fwd_pts_50
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 16)

print("="*100)
print("MACRO ESTUDIO ESTADÍSTICO: PREDICTORES DE PnL DESASTROSO EN VENTANA 50")
print("="*100)

# ============================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ============================================================================
print("\n" + "="*100)
print("1. CARGA Y PREPARACIÓN DE DATOS")
print("="*100)

df = pd.read_csv('/home/user/BatmanLT/PNLDV.csv')
print(f"\nRegistros totales: {len(df)}")
print(f"Columnas disponibles: {len(df.columns)}")

# Definir variable objetivo: PnL desastroso
# Usaremos el peor quintil (20% inferior) como "desastroso"
df_valid = df[df['PnL_fwd_pts_50'].notna()].copy()
print(f"Registros con PnL_fwd_pts_50 válido: {len(df_valid)}")

# Crear clasificación de PnL
percentile_20 = df_valid['PnL_fwd_pts_50'].quantile(0.20)
percentile_10 = df_valid['PnL_fwd_pts_50'].quantile(0.10)

df_valid['PnL_desastroso'] = (df_valid['PnL_fwd_pts_50'] < percentile_20).astype(int)
df_valid['PnL_muy_desastroso'] = (df_valid['PnL_fwd_pts_50'] < percentile_10).astype(int)

print(f"\nDefinición de PnL desastroso:")
print(f"  - Umbral quintil inferior (20%): {percentile_20:.2f} pts")
print(f"  - Umbral decil inferior (10%): {percentile_10:.2f} pts")
print(f"  - Casos desastrosos: {df_valid['PnL_desastroso'].sum()} ({df_valid['PnL_desastroso'].mean()*100:.1f}%)")
print(f"  - Casos muy desastrosos: {df_valid['PnL_muy_desastroso'].sum()} ({df_valid['PnL_muy_desastroso'].mean()*100:.1f}%)")

# ============================================================================
# 2. SELECCIÓN Y CREACIÓN DE VARIABLES PREDICTORAS
# ============================================================================
print("\n" + "="*100)
print("2. CREACIÓN DE VARIABLES PREDICTORAS")
print("="*100)

# Variables disponibles en T+0
base_vars_t0 = [
    'BQI_ABS', 'net_credit_diff', 'delta_total', 'theta_total',
    'net_credit', 'SPX', 'Asym', 'BQI_V2_ABS', 'BQR_1000',
    'Death valley', 'EarL', 'EarR', 'EarScore', 'PnLDV',
    'RANK_PRE', 'RATIO_BATMAN', 'RATIO_UEL_EARS', 'UEL_inf_USD'
]

# Variables IV
iv_vars = ['iv_k1', 'iv_k2', 'iv_k3']

# Variables Greeks individuales
greek_vars = ['delta_k1', 'delta_k2', 'delta_k3', 'theta_k1', 'theta_k2', 'theta_k3']

# Variables T+25
vars_t25 = ['PnLDV_fwd_25', 'PnL_fwd_pts_25', 'SPX_chg_pct_25']

print(f"\nVariables base en T+0: {len(base_vars_t0)}")
print(f"Variables IV: {len(iv_vars)}")
print(f"Variables Greeks: {len(greek_vars)}")
print(f"Variables T+25: {len(vars_t25)}")

# Crear variables derivadas INNOVADORAS
print("\n--- CREANDO VARIABLES DERIVADAS ---")

# Ratios y métricas compuestas
df_valid['BQI_per_net_credit'] = df_valid['BQI_ABS'] / (df_valid['net_credit'].abs() + 1)
df_valid['Death_valley_per_credit'] = df_valid['Death valley'] / (df_valid['net_credit'].abs() + 1)
df_valid['PnLDV_per_credit'] = df_valid['PnLDV'] / (df_valid['net_credit'].abs() + 1)
df_valid['Ears_asymmetry'] = (df_valid['EarL'] - df_valid['EarR']).abs()
df_valid['Ears_total'] = df_valid['EarL'] + df_valid['EarR']
df_valid['Ears_ratio'] = df_valid['EarL'] / (df_valid['EarR'] + 1)

# Ratios de IV
df_valid['iv_spread_k1_k2'] = df_valid['iv_k1'] - df_valid['iv_k2']
df_valid['iv_spread_k2_k3'] = df_valid['iv_k2'] - df_valid['iv_k3']
df_valid['iv_spread_k1_k3'] = df_valid['iv_k1'] - df_valid['iv_k3']
df_valid['iv_skew'] = (df_valid['iv_k1'] + df_valid['iv_k3']) / (2 * df_valid['iv_k2'] + 0.001)

# Métricas de riesgo
df_valid['theta_delta_ratio'] = df_valid['theta_total'] / (df_valid['delta_total'].abs() + 0.001)
df_valid['risk_reward_ratio'] = df_valid['net_credit'] / (df_valid['BQI_ABS'] + 1)

# Indicadores de deterioro temprano (T+25)
df_valid['PnL_deterioration_25'] = df_valid['PnL_fwd_pts_25'] / (df_valid['net_credit'] + 0.001)
df_valid['PnLDV_deterioration_25'] = df_valid['PnLDV_fwd_25'] / (df_valid['PnLDV'].abs() + 1)

# Indicador combinado de peligro
df_valid['danger_score'] = (
    (df_valid['PnLDV'] < df_valid['PnLDV'].quantile(0.25)).astype(int) * 3 +
    (df_valid['BQI_ABS'] > df_valid['BQI_ABS'].quantile(0.75)).astype(int) * 2 +
    (df_valid['Death valley'] > df_valid['Death valley'].quantile(0.75)).astype(int) * 2 +
    (df_valid['delta_total'].abs() > df_valid['delta_total'].abs().quantile(0.75)).astype(int)
)

# Variables de interacción
df_valid['PnLDV_x_BQI'] = df_valid['PnLDV'] * df_valid['BQI_ABS']
df_valid['Death_valley_x_delta'] = df_valid['Death valley'] * df_valid['delta_total'].abs()

derived_vars = [
    'BQI_per_net_credit', 'Death_valley_per_credit', 'PnLDV_per_credit',
    'Ears_asymmetry', 'Ears_total', 'Ears_ratio',
    'iv_spread_k1_k2', 'iv_spread_k2_k3', 'iv_spread_k1_k3', 'iv_skew',
    'theta_delta_ratio', 'risk_reward_ratio',
    'PnL_deterioration_25', 'PnLDV_deterioration_25',
    'danger_score', 'PnLDV_x_BQI', 'Death_valley_x_delta'
]

print(f"Variables derivadas creadas: {len(derived_vars)}")

# Lista completa de predictores
all_predictors_t0 = base_vars_t0 + iv_vars + greek_vars + derived_vars
all_predictors_t25 = all_predictors_t0 + vars_t25

print(f"\nTotal predictores T+0: {len(all_predictors_t0)}")
print(f"Total predictores T+0 + T+25: {len(all_predictors_t25)}")

# ============================================================================
# 3. ANÁLISIS UNIVARIADO: PODER PREDICTIVO INDIVIDUAL
# ============================================================================
print("\n" + "="*100)
print("3. ANÁLISIS UNIVARIADO DE PREDICTORES")
print("="*100)

def evaluate_predictor(df, predictor, target='PnL_desastroso'):
    """Evalúa el poder predictivo de una variable individual"""
    # Limpiar datos
    data = df[[predictor, target, 'PnL_fwd_pts_50']].dropna()
    if len(data) < 50:
        return None

    X = data[predictor].values.reshape(-1, 1)
    y = data[target].values
    y_continuous = data['PnL_fwd_pts_50'].values

    results = {'variable': predictor, 'n_samples': len(data)}

    try:
        # Correlación con PnL continuo
        corr, p_val = stats.pearsonr(data[predictor], y_continuous)
        results['correlation'] = corr
        results['corr_pvalue'] = p_val

        # AUC para clasificación binaria
        if len(np.unique(data[predictor])) > 1:
            # Invertir si correlación negativa para que AUC > 0.5
            if corr < 0:
                X_norm = -X
            else:
                X_norm = X

            auc = roc_auc_score(y, X_norm)
            results['auc'] = auc
        else:
            results['auc'] = 0.5

        # Diferencia de medias entre grupos
        group_desastroso = data[data[target] == 1][predictor].mean()
        group_normal = data[data[target] == 0][predictor].mean()
        results['mean_disastrous'] = group_desastroso
        results['mean_normal'] = group_normal
        results['mean_diff'] = group_desastroso - group_normal

        # Test t
        t_stat, t_pval = stats.ttest_ind(
            data[data[target] == 1][predictor],
            data[data[target] == 0][predictor],
            equal_var=False
        )
        results['t_statistic'] = t_stat
        results['t_pvalue'] = t_pval

        # Quintiles del predictor
        data['quintil'] = pd.qcut(data[predictor], q=5, labels=False, duplicates='drop')
        quintil_means = data.groupby('quintil')[target].mean()

        # Tendencia monotónica
        if len(quintil_means) >= 3:
            slope, _, r_val, p_trend, _ = stats.linregress(
                range(len(quintil_means)), quintil_means.values
            )
            results['trend_slope'] = slope
            results['trend_r2'] = r_val**2
            results['trend_pvalue'] = p_trend

    except Exception as e:
        print(f"  Warning: Error evaluating {predictor}: {str(e)}")
        return None

    return results

print("\nEvaluando predictores T+0...")
results_t0 = []
for var in all_predictors_t0:
    if var in df_valid.columns:
        result = evaluate_predictor(df_valid, var)
        if result:
            results_t0.append(result)

print(f"Predictores T+0 evaluados: {len(results_t0)}")

print("\nEvaluando predictores T+25...")
results_t25 = []
for var in all_predictors_t25:
    if var in df_valid.columns:
        result = evaluate_predictor(df_valid, var)
        if result:
            results_t25.append(result)

print(f"Predictores T+0+T+25 evaluados: {len(results_t25)}")

# Crear DataFrames de resultados
results_df_t0 = pd.DataFrame(results_t0)
results_df_t25 = pd.DataFrame(results_t25)

# Calcular score combinado de poder predictivo
def calculate_predictive_score(row):
    """Score combinado basado en múltiples métricas"""
    score = 0

    # AUC (peso 40%)
    if not pd.isna(row['auc']):
        score += abs(row['auc'] - 0.5) * 40

    # Correlación absoluta (peso 30%)
    if not pd.isna(row['correlation']):
        score += abs(row['correlation']) * 30

    # Significancia estadística (peso 20%)
    if not pd.isna(row['t_pvalue']) and row['t_pvalue'] < 0.01:
        score += 20
    elif not pd.isna(row['t_pvalue']) and row['t_pvalue'] < 0.05:
        score += 10

    # Tendencia monotónica (peso 10%)
    if not pd.isna(row['trend_r2']):
        score += row['trend_r2'] * 10

    return score

results_df_t0['predictive_score'] = results_df_t0.apply(calculate_predictive_score, axis=1)
results_df_t25['predictive_score'] = results_df_t25.apply(calculate_predictive_score, axis=1)

# Ordenar por score
results_df_t0 = results_df_t0.sort_values('predictive_score', ascending=False)
results_df_t25 = results_df_t25.sort_values('predictive_score', ascending=False)

print("\n" + "-"*100)
print("TOP 15 PREDICTORES EN T+0:")
print("-"*100)
print(results_df_t0[['variable', 'auc', 'correlation', 't_pvalue', 'predictive_score']].head(15).to_string(index=False))

print("\n" + "-"*100)
print("TOP 15 PREDICTORES EN T+0 + T+25:")
print("-"*100)
print(results_df_t25[['variable', 'auc', 'correlation', 't_pvalue', 'predictive_score']].head(15).to_string(index=False))

# ============================================================================
# 4. ANÁLISIS MULTIVARIADO: RANDOM FOREST FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*100)
print("4. ANÁLISIS MULTIVARIADO: FEATURE IMPORTANCE")
print("="*100)

# Preparar datos para modelo
def prepare_model_data(df, predictors, target='PnL_desastroso'):
    """Prepara datos limpios para modelado"""
    cols_needed = predictors + [target]
    df_model = df[cols_needed].copy()

    # Eliminar infinitos y NaN
    df_model = df_model.replace([np.inf, -np.inf], np.nan)
    df_model = df_model.dropna()

    X = df_model[predictors]
    y = df_model[target]

    return X, y, df_model

# Seleccionar mejores predictores para modelo
top_predictors_t0 = results_df_t0.head(30)['variable'].tolist()
top_predictors_t25 = results_df_t25.head(30)['variable'].tolist()

# Filtrar los que existen
top_predictors_t0 = [v for v in top_predictors_t0 if v in df_valid.columns]
top_predictors_t25 = [v for v in top_predictors_t25 if v in df_valid.columns]

print(f"\nTop predictores para modelo T+0: {len(top_predictors_t0)}")
print(f"Top predictores para modelo T+25: {len(top_predictors_t25)}")

# Modelo T+0
print("\n--- RANDOM FOREST T+0 ---")
X_t0, y_t0, data_t0 = prepare_model_data(df_valid, top_predictors_t0)
print(f"Muestras para modelo T+0: {len(X_t0)}")

rf_t0 = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=20, random_state=42)
rf_t0.fit(X_t0, y_t0)

# Feature importance
importance_t0 = pd.DataFrame({
    'variable': top_predictors_t0,
    'importance': rf_t0.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 por Feature Importance (T+0):")
print(importance_t0.head(15).to_string(index=False))

# Modelo T+25
print("\n--- RANDOM FOREST T+0 + T+25 ---")
X_t25, y_t25, data_t25 = prepare_model_data(df_valid, top_predictors_t25)
print(f"Muestras para modelo T+25: {len(X_t25)}")

rf_t25 = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=20, random_state=42)
rf_t25.fit(X_t25, y_t25)

importance_t25 = pd.DataFrame({
    'variable': top_predictors_t25,
    'importance': rf_t25.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 por Feature Importance (T+0 + T+25):")
print(importance_t25.head(15).to_string(index=False))

# Cross-validation scores
cv_scores_t0 = cross_val_score(rf_t0, X_t0, y_t0, cv=5, scoring='roc_auc')
cv_scores_t25 = cross_val_score(rf_t25, X_t25, y_t25, cv=5, scoring='roc_auc')

print(f"\nCross-validation AUC T+0: {cv_scores_t0.mean():.4f} (+/- {cv_scores_t0.std():.4f})")
print(f"Cross-validation AUC T+25: {cv_scores_t25.mean():.4f} (+/- {cv_scores_t25.std():.4f})")

# ============================================================================
# 5. ANÁLISIS DETALLADO DE TOP PREDICTORES
# ============================================================================
print("\n" + "="*100)
print("5. ANÁLISIS DETALLADO DE TOP 5 PREDICTORES")
print("="*100)

top_5_vars = importance_t25.head(5)['variable'].tolist()

for var in top_5_vars:
    print(f"\n{'='*80}")
    print(f"PREDICTOR: {var}")
    print(f"{'='*80}")

    data = df_valid[[var, 'PnL_desastroso', 'PnL_fwd_pts_50']].dropna()

    # Crear quintiles
    data['quintil'] = pd.qcut(data[var], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')

    # Análisis por quintil
    quintil_analysis = data.groupby('quintil').agg({
        'PnL_desastroso': ['count', 'sum', 'mean'],
        'PnL_fwd_pts_50': ['mean', 'median', 'std']
    }).round(3)

    print("\nAnálisis por quintil:")
    print(quintil_analysis)

    # Tasa de desastre por quintil
    disaster_rate = data.groupby('quintil')['PnL_desastroso'].mean() * 100
    print(f"\nTasa de desastre por quintil (%):")
    for q, rate in disaster_rate.items():
        print(f"  {q}: {rate:.1f}%")

# ============================================================================
# 6. RECOMENDACIONES Y UMBRALES
# ============================================================================
print("\n" + "="*100)
print("6. RECOMENDACIONES: UMBRALES DE ALERTA")
print("="*100)

# Para cada top predictor, encontrar umbral óptimo
for var in importance_t25.head(10)['variable'].tolist():
    data = df_valid[[var, 'PnL_desastroso']].dropna()

    if len(data) < 100:
        continue

    # Encontrar umbral que maximiza diferencia
    percentiles = [10, 20, 25, 30]

    best_perc = None
    best_ratio = 0

    for perc in percentiles:
        threshold = data[var].quantile(perc/100) if data[var].mean() < 0 else data[var].quantile(1 - perc/100)

        if data[var].mean() < 0:
            high_risk = data[data[var] <= threshold]
        else:
            high_risk = data[data[var] >= threshold]

        if len(high_risk) > 0:
            disaster_rate_high = high_risk['PnL_desastroso'].mean()
            disaster_rate_low = data[~data.index.isin(high_risk.index)]['PnL_desastroso'].mean()

            ratio = disaster_rate_high / (disaster_rate_low + 0.001)

            if ratio > best_ratio:
                best_ratio = ratio
                best_perc = perc

    if best_perc:
        threshold = data[var].quantile(best_perc/100) if data[var].mean() < 0 else data[var].quantile(1 - best_perc/100)
        print(f"\n{var}:")
        print(f"  Umbral recomendado: {threshold:.4f}")
        print(f"  Tipo: {'<=' if data[var].mean() < 0 else '>='} umbral = ALTO RIESGO")
        print(f"  Incremento de riesgo: {best_ratio:.2f}x")

# Guardar resultados
results_df_t0.to_csv('/home/user/BatmanLT/predictors_analysis_t0.csv', index=False)
results_df_t25.to_csv('/home/user/BatmanLT/predictors_analysis_t25.csv', index=False)
importance_t0.to_csv('/home/user/BatmanLT/feature_importance_t0.csv', index=False)
importance_t25.to_csv('/home/user/BatmanLT/feature_importance_t25.csv', index=False)

print("\n" + "="*100)
print("ARCHIVOS GUARDADOS")
print("="*100)
print("- predictors_analysis_t0.csv")
print("- predictors_analysis_t25.csv")
print("- feature_importance_t0.csv")
print("- feature_importance_t25.csv")

print("\n" + "="*100)
print("ANÁLISIS COMPLETADO")
print("="*100)
