#!/usr/bin/env python3
"""
Análisis Estadístico de FWD PTS - Identificación de Drivers de Rentabilidad
Análisis de correlaciones y umbrales para estructuras Batman
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print(" ANÁLISIS ESTADÍSTICO - FWD PTS vs VARIABLES PREDICTORAS")
print("="*80)
print()

# 1. CARGA DE DATOS
print("📊 Cargando datos...")
df = pd.read_csv('combined_mediana.csv')
print(f"✓ Dataset cargado: {len(df)} registros, {len(df.columns)} columnas")
print()

# 2. DEFINICIÓN DE VARIABLES
# Variables objetivo (FWD PTS - NO PTC)
target_vars = [
    'PnL_fwd_pts_01',  # 1% del tiempo de vida
    'PnL_fwd_pts_05',  # 5% del tiempo de vida
    'PnL_fwd_pts_25',  # 25% del tiempo de vida
    'PnL_fwd_pts_50',  # 50% del tiempo de vida
]

# Variables predictoras
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

# 3. LIMPIEZA DE DATOS
print("🧹 Limpiando datos (eliminando NaN e infinitos)...")
# Crear subset con las variables de interés
analysis_vars = target_vars + predictor_vars
df_clean = df[analysis_vars].copy()

# Eliminar infinitos y NaN
df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
df_clean = df_clean.dropna()

print(f"✓ Datos limpios: {len(df_clean)} registros válidos ({len(df_clean)/len(df)*100:.1f}%)")
print()

# 4. ESTADÍSTICAS DESCRIPTIVAS
print("="*80)
print(" ESTADÍSTICAS DESCRIPTIVAS")
print("="*80)
print()

print("📈 Variables Objetivo (FWD PTS):")
print(df_clean[target_vars].describe().round(2))
print()

print("📊 Variables Predictoras:")
print(df_clean[predictor_vars].describe().round(2))
print()

# 5. ANÁLISIS DE CORRELACIONES
print("="*80)
print(" ANÁLISIS DE CORRELACIONES")
print("="*80)
print()

# Matriz de correlación completa
correlation_results = {}

for target in target_vars:
    print(f"\n🎯 {target} (Correlación de Pearson y Spearman)")
    print("-" * 80)

    correlations = []
    for predictor in predictor_vars:
        # Pearson correlation
        pearson_corr, pearson_pval = pearsonr(df_clean[predictor], df_clean[target])
        # Spearman correlation (para relaciones no lineales)
        spearman_corr, spearman_pval = spearmanr(df_clean[predictor], df_clean[target])

        correlations.append({
            'Variable': predictor,
            'Pearson_r': pearson_corr,
            'Pearson_pval': pearson_pval,
            'Spearman_r': spearman_corr,
            'Spearman_pval': spearman_pval,
            'Pearson_r_abs': abs(pearson_corr)
        })

    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('Pearson_r_abs', ascending=False)
    correlation_results[target] = corr_df

    print(corr_df.to_string(index=False))
    print()

# 6. RANKING GENERAL DE VARIABLES MÁS CORRELACIONADAS
print("="*80)
print(" RANKING GENERAL - VARIABLES MÁS CORRELACIONADAS")
print("="*80)
print()

# Calcular correlación promedio para cada predictor
avg_correlations = []
for predictor in predictor_vars:
    pearson_avg = np.mean([abs(pearsonr(df_clean[predictor], df_clean[target])[0])
                           for target in target_vars])
    spearman_avg = np.mean([abs(spearmanr(df_clean[predictor], df_clean[target])[0])
                            for target in target_vars])

    avg_correlations.append({
        'Variable': predictor,
        'Avg_Pearson_r': pearson_avg,
        'Avg_Spearman_r': spearman_avg,
        'Combined_Score': (pearson_avg + spearman_avg) / 2
    })

ranking_df = pd.DataFrame(avg_correlations).sort_values('Combined_Score', ascending=False)
print("🏆 Ranking por correlación promedio (absoluta) con todos los FWD PTS:")
print(ranking_df.to_string(index=False))
print()

top_predictor = ranking_df.iloc[0]['Variable']
print(f"✨ MEJOR PREDICTOR: {top_predictor}")
print(f"   Score combinado: {ranking_df.iloc[0]['Combined_Score']:.4f}")
print()

# 7. ANÁLISIS DE UMBRALES (QUARTILES Y DECILES)
print("="*80)
print(" ANÁLISIS DE UMBRALES Y RENTABILIDAD")
print("="*80)
print()

# Analizar los 3 mejores predictores
top_3_predictors = ranking_df.head(3)['Variable'].tolist()

for predictor in top_3_predictors:
    print(f"\n🔍 {predictor}")
    print("-" * 80)

    # Análisis por quartiles
    df_clean['quartile'] = pd.qcut(df_clean[predictor], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')

    print("\n📊 Rentabilidad promedio por QUARTIL:")
    quartile_analysis = df_clean.groupby('quartile')[target_vars].mean()
    print(quartile_analysis.round(2))
    print()

    # Análisis por deciles
    df_clean['decile'] = pd.qcut(df_clean[predictor], q=10, labels=False, duplicates='drop') + 1

    print("📊 Rentabilidad promedio por DECIL:")
    decile_analysis = df_clean.groupby('decile')[target_vars].mean()
    print(decile_analysis.round(2))
    print()

    # Identificar umbrales críticos
    # Percentiles clave
    percentiles = [10, 25, 50, 75, 90]
    print(f"📏 Percentiles de {predictor}:")
    for p in percentiles:
        val = np.percentile(df_clean[predictor], p)
        print(f"   P{p}: {val:.4f}")
    print()

    # Análisis de umbral óptimo
    # Dividir en grupos por encima/debajo de diferentes umbrales
    print("🎯 Análisis de UMBRALES CRÍTICOS:")
    test_percentiles = [25, 50, 75, 90]
    for p in test_percentiles:
        threshold = np.percentile(df_clean[predictor], p)
        above = df_clean[df_clean[predictor] >= threshold][target_vars].mean()
        below = df_clean[df_clean[predictor] < threshold][target_vars].mean()
        diff = above - below

        print(f"\n   Umbral: P{p} = {threshold:.4f}")
        print(f"   Encima del umbral: n={len(df_clean[df_clean[predictor] >= threshold])}")
        for target in target_vars:
            print(f"      {target}: {above[target]:.2f} pts (diff: {diff[target]:+.2f})")

    print()

# 8. IDENTIFICACIÓN DE ZONAS EVITAR
print("="*80)
print(" ⚠️  ZONAS A EVITAR - Configuraciones de BAJA rentabilidad")
print("="*80)
print()

for predictor in top_3_predictors:
    print(f"\n🚫 {predictor}")
    print("-" * 80)

    # Primer quartil (peores resultados)
    q1_threshold = df_clean[predictor].quantile(0.25)
    q1_data = df_clean[df_clean[predictor] <= q1_threshold]

    print(f"   Q1 (25% inferior): {predictor} <= {q1_threshold:.4f}")
    print(f"   Rentabilidad promedio:")
    for target in target_vars:
        print(f"      {target}: {q1_data[target].mean():.2f} pts")
    print()

    # Último quartil (mejores resultados)
    q4_threshold = df_clean[predictor].quantile(0.75)
    q4_data = df_clean[df_clean[predictor] >= q4_threshold]

    print(f"   Q4 (25% superior): {predictor} >= {q4_threshold:.4f}")
    print(f"   Rentabilidad promedio:")
    for target in target_vars:
        print(f"      {target}: {q4_data[target].mean():.2f} pts")
    print()

# 9. ANÁLISIS MULTIVARIADO
print("="*80)
print(" ANÁLISIS MULTIVARIADO - Combinaciones óptimas")
print("="*80)
print()

# Crear variable de alto rendimiento
median_50 = df_clean['PnL_fwd_pts_50'].median()
df_clean['high_performer'] = (df_clean['PnL_fwd_pts_50'] > median_50).astype(int)

print("🎯 Características de las estructuras de ALTO RENDIMIENTO (> mediana PnL_fwd_pts_50):")
print()

high_perf = df_clean[df_clean['high_performer'] == 1]
low_perf = df_clean[df_clean['high_performer'] == 0]

comparison = pd.DataFrame({
    'Variable': predictor_vars,
    'Alto_Rendimiento': [high_perf[v].mean() for v in predictor_vars],
    'Bajo_Rendimiento': [low_perf[v].mean() for v in predictor_vars],
    'Diferencia': [high_perf[v].mean() - low_perf[v].mean() for v in predictor_vars],
    'Diferencia_%': [(high_perf[v].mean() - low_perf[v].mean()) / abs(low_perf[v].mean()) * 100
                     for v in predictor_vars]
})

comparison = comparison.sort_values('Diferencia_%', ascending=False, key=abs)
print(comparison.to_string(index=False))
print()

# 10. RESUMEN EJECUTIVO
print("="*80)
print(" 📋 RESUMEN EJECUTIVO Y RECOMENDACIONES")
print("="*80)
print()

print("🏆 TOP 3 DRIVERS MÁS CORRELACIONADOS:")
for i, row in ranking_df.head(3).iterrows():
    print(f"   {i+1}. {row['Variable']}: Score = {row['Combined_Score']:.4f}")
print()

print("✅ RECOMENDACIONES:")
print()

# Identificar la mejor variable
best_var = ranking_df.iloc[0]['Variable']
best_threshold = df_clean[best_var].quantile(0.75)
print(f"1. PRIORIZAR estructuras con {best_var} >= {best_threshold:.4f} (Q4)")
print(f"   → Rentabilidad esperada significativamente superior")
print()

# Segunda mejor variable
if len(ranking_df) > 1:
    second_best = ranking_df.iloc[1]['Variable']
    second_threshold = df_clean[second_best].quantile(0.75)
    print(f"2. COMPLEMENTAR con {second_best} >= {second_threshold:.4f}")
    print()

# Variables a evitar
worst_var = ranking_df.iloc[-1]['Variable']
worst_threshold = df_clean[worst_var].quantile(0.25)
print(f"3. EVITAR estructuras con {worst_var} en el Q1 (< {worst_threshold:.4f})")
print(f"   → Asociado con menor rentabilidad")
print()

print("4. COMBINAR múltiples indicadores: no depender de un solo driver")
print()

print("5. MONITOREAR especialmente:")
for var in top_3_predictors:
    threshold = df_clean[var].quantile(0.75)
    print(f"   • {var} (óptimo > {threshold:.4f})")
print()

# 11. GUARDAR RESULTADOS
print("="*80)
print(" 💾 GUARDANDO RESULTADOS")
print("="*80)
print()

# Guardar correlaciones
with open('analisis_correlaciones.txt', 'w') as f:
    f.write("ANÁLISIS DE CORRELACIONES - FWD PTS\n")
    f.write("="*80 + "\n\n")
    for target, corr_df in correlation_results.items():
        f.write(f"\n{target}\n")
        f.write("-"*80 + "\n")
        f.write(corr_df.to_string(index=False))
        f.write("\n\n")

print("✓ Correlaciones guardadas en: analisis_correlaciones.txt")

# Guardar ranking
ranking_df.to_csv('ranking_predictores.csv', index=False)
print("✓ Ranking guardado en: ranking_predictores.csv")

# Guardar comparación alto/bajo rendimiento
comparison.to_csv('comparacion_rendimiento.csv', index=False)
print("✓ Comparación guardada en: comparacion_rendimiento.csv")

print()
print("="*80)
print(" ✨ ANÁLISIS COMPLETADO")
print("="*80)
