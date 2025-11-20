#!/usr/bin/env python3
"""
Análisis de Vencimientos (DTE1/DTE2) vs Rentabilidad FWD PTS
Identificación de combinaciones óptimas de vencimientos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Configuración
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print(" ANÁLISIS DE VENCIMIENTOS (DTE1/DTE2) vs RENTABILIDAD FWD PTS")
print("="*80)
print()

# 1. CARGA DE DATOS
print("📊 Cargando datos...")
df = pd.read_csv('combined_mediana.csv')
print(f"✓ Dataset cargado: {len(df)} registros")
print()

# 2. PROCESAR COLUMNA DTE1/DTE2
print("🔍 Procesando columna DTE1/DTE2...")

# Separar DTE1 y DTE2
df[['DTE1', 'DTE2']] = df['DTE1/DTE2'].str.split('/', expand=True)
df['DTE1'] = pd.to_numeric(df['DTE1'], errors='coerce')
df['DTE2'] = pd.to_numeric(df['DTE2'], errors='coerce')

# Calcular métricas derivadas
df['DTE_ratio'] = df['DTE2'] / df['DTE1']  # Ratio entre vencimientos
df['DTE_diff'] = df['DTE2'] - df['DTE1']   # Diferencia en días
df['DTE_sum'] = df['DTE1'] + df['DTE2']    # Suma total de días
df['DTE_avg'] = (df['DTE1'] + df['DTE2']) / 2  # Promedio

print(f"✓ Columnas procesadas: DTE1, DTE2, DTE_ratio, DTE_diff, DTE_sum, DTE_avg")
print()

# Variables objetivo
target_vars = [
    'PnL_fwd_pts_01',
    'PnL_fwd_pts_05',
    'PnL_fwd_pts_25',
    'PnL_fwd_pts_50',
]

# Variables de vencimiento
dte_vars = ['DTE1', 'DTE2', 'DTE_ratio', 'DTE_diff', 'DTE_sum', 'DTE_avg']

# 3. LIMPIEZA
print("🧹 Limpiando datos...")
analysis_vars = target_vars + dte_vars
df_clean = df[analysis_vars].copy()
df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
df_clean = df_clean.dropna()
print(f"✓ Datos limpios: {len(df_clean)} registros válidos ({len(df_clean)/len(df)*100:.1f}%)")
print()

# 4. ESTADÍSTICAS DESCRIPTIVAS DE DTE
print("="*80)
print(" ESTADÍSTICAS DESCRIPTIVAS - VENCIMIENTOS")
print("="*80)
print()

print("📊 Distribución de vencimientos:")
print(df_clean[dte_vars].describe().round(2))
print()

# 5. ANÁLISIS DE CORRELACIONES
print("="*80)
print(" CORRELACIONES: DTE vs FWD PTS")
print("="*80)
print()

correlation_results = {}

for target in target_vars:
    print(f"\n🎯 {target}")
    print("-" * 80)

    correlations = []
    for dte_var in dte_vars:
        pearson_corr, pearson_pval = pearsonr(df_clean[dte_var], df_clean[target])
        spearman_corr, spearman_pval = spearmanr(df_clean[dte_var], df_clean[target])

        correlations.append({
            'Variable': dte_var,
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

# 6. RANKING GENERAL
print("="*80)
print(" RANKING GENERAL - VARIABLES DTE MÁS CORRELACIONADAS")
print("="*80)
print()

avg_correlations = []
for dte_var in dte_vars:
    pearson_avg = np.mean([abs(pearsonr(df_clean[dte_var], df_clean[target])[0])
                           for target in target_vars])
    spearman_avg = np.mean([abs(spearmanr(df_clean[dte_var], df_clean[target])[0])
                            for target in target_vars])

    avg_correlations.append({
        'Variable': dte_var,
        'Avg_Pearson_r': pearson_avg,
        'Avg_Spearman_r': spearman_avg,
        'Combined_Score': (pearson_avg + spearman_avg) / 2
    })

ranking_df = pd.DataFrame(avg_correlations).sort_values('Combined_Score', ascending=False)
print("🏆 Ranking por correlación promedio:")
print(ranking_df.to_string(index=False))
print()

# 7. ANÁLISIS DE COMBINACIONES ESPECÍFICAS DTE1/DTE2
print("="*80)
print(" ANÁLISIS DE COMBINACIONES DTE1/DTE2 MÁS COMUNES")
print("="*80)
print()

# Agrupar por combinación DTE1/DTE2
df_clean['DTE_combo'] = df_clean['DTE1'].astype(int).astype(str) + '/' + df_clean['DTE2'].astype(int).astype(str)
combo_stats = df_clean.groupby('DTE_combo').agg({
    'PnL_fwd_pts_01': ['mean', 'std', 'count'],
    'PnL_fwd_pts_05': ['mean', 'std', 'count'],
    'PnL_fwd_pts_25': ['mean', 'std', 'count'],
    'PnL_fwd_pts_50': ['mean', 'std', 'count']
}).round(2)

# Filtrar combinaciones con al menos 30 muestras
combo_stats_filtered = combo_stats[combo_stats[('PnL_fwd_pts_50', 'count')] >= 30].copy()
combo_stats_filtered = combo_stats_filtered.sort_values(('PnL_fwd_pts_50', 'mean'), ascending=False)

print(f"📊 Top 15 combinaciones por rentabilidad (PnL_fwd_pts_50) - Mínimo 30 muestras:")
print()
print(combo_stats_filtered.head(15))
print()

print(f"📊 Bottom 15 combinaciones (peores) - Mínimo 30 muestras:")
print()
print(combo_stats_filtered.tail(15))
print()

# 8. ANÁLISIS POR RANGOS DE DTE1
print("="*80)
print(" ANÁLISIS POR RANGOS DE DTE1")
print("="*80)
print()

# Crear rangos de DTE1
dte1_bins = [0, 100, 200, 300, 500, 1000, 2000]
dte1_labels = ['0-100', '100-200', '200-300', '300-500', '500-1000', '1000+']
df_clean['DTE1_range'] = pd.cut(df_clean['DTE1'], bins=dte1_bins, labels=dte1_labels)

print("📊 Rentabilidad por rango de DTE1:")
dte1_analysis = df_clean.groupby('DTE1_range')[target_vars].agg(['mean', 'count'])
print(dte1_analysis.round(2))
print()

# 9. ANÁLISIS POR RANGOS DE DTE2
print("="*80)
print(" ANÁLISIS POR RANGOS DE DTE2")
print("="*80)
print()

# Crear rangos de DTE2
dte2_bins = [0, 200, 400, 600, 800, 1200, 3000]
dte2_labels = ['0-200', '200-400', '400-600', '600-800', '800-1200', '1200+']
df_clean['DTE2_range'] = pd.cut(df_clean['DTE2'], bins=dte2_bins, labels=dte2_labels)

print("📊 Rentabilidad por rango de DTE2:")
dte2_analysis = df_clean.groupby('DTE2_range')[target_vars].agg(['mean', 'count'])
print(dte2_analysis.round(2))
print()

# 10. ANÁLISIS POR RATIO DTE2/DTE1
print("="*80)
print(" ANÁLISIS POR RATIO DTE2/DTE1")
print("="*80)
print()

# Crear rangos de ratio
ratio_bins = [0, 1.5, 2.0, 2.5, 3.0, 4.0, 10.0]
ratio_labels = ['<1.5x', '1.5-2.0x', '2.0-2.5x', '2.5-3.0x', '3.0-4.0x', '>4.0x']
df_clean['DTE_ratio_range'] = pd.cut(df_clean['DTE_ratio'], bins=ratio_bins, labels=ratio_labels)

print("📊 Rentabilidad por ratio DTE2/DTE1:")
ratio_analysis = df_clean.groupby('DTE_ratio_range')[target_vars].agg(['mean', 'count'])
print(ratio_analysis.round(2))
print()

# 11. ANÁLISIS POR DIFERENCIA DE DÍAS
print("="*80)
print(" ANÁLISIS POR DIFERENCIA DE DÍAS (DTE2 - DTE1)")
print("="*80)
print()

# Crear rangos de diferencia
diff_bins = [0, 200, 400, 600, 800, 1200, 3000]
diff_labels = ['0-200', '200-400', '400-600', '600-800', '800-1200', '1200+']
df_clean['DTE_diff_range'] = pd.cut(df_clean['DTE_diff'], bins=diff_bins, labels=diff_labels)

print("📊 Rentabilidad por diferencia de días:")
diff_analysis = df_clean.groupby('DTE_diff_range')[target_vars].agg(['mean', 'count'])
print(diff_analysis.round(2))
print()

# 12. IDENTIFICAR COMBINACIONES ÓPTIMAS
print("="*80)
print(" 🎯 COMBINACIONES ÓPTIMAS DE VENCIMIENTOS")
print("="*80)
print()

# Top combinaciones específicas
top_combos = combo_stats_filtered.head(10)
print("✅ TOP 10 COMBINACIONES (ordenadas por PnL_fwd_pts_50):")
print()
for idx, (combo, row) in enumerate(top_combos.iterrows(), 1):
    pnl_50_mean = row[('PnL_fwd_pts_50', 'mean')]
    pnl_50_count = row[('PnL_fwd_pts_50', 'count')]
    print(f"{idx:2}. {combo:15} → {pnl_50_mean:7.2f} pts (n={int(pnl_50_count):4})")
print()

# Peores combinaciones
worst_combos = combo_stats_filtered.tail(10)
print("🚫 PEORES 10 COMBINACIONES (a evitar):")
print()
for idx, (combo, row) in enumerate(worst_combos.iterrows(), 1):
    pnl_50_mean = row[('PnL_fwd_pts_50', 'mean')]
    pnl_50_count = row[('PnL_fwd_pts_50', 'count')]
    print(f"{idx:2}. {combo:15} → {pnl_50_mean:7.2f} pts (n={int(pnl_50_count):4})")
print()

# 13. ANÁLISIS DE PERCENTILES
print("="*80)
print(" ANÁLISIS DE PERCENTILES - DTE1 y DTE2")
print("="*80)
print()

percentiles = [10, 25, 50, 75, 90]

print("📏 Percentiles de DTE1:")
for p in percentiles:
    val = np.percentile(df_clean['DTE1'], p)
    above = df_clean[df_clean['DTE1'] >= val]['PnL_fwd_pts_50'].mean()
    below = df_clean[df_clean['DTE1'] < val]['PnL_fwd_pts_50'].mean()
    print(f"   P{p:2}: {val:6.0f} días → Encima: {above:6.2f} pts | Debajo: {below:6.2f} pts | Diff: {above-below:+7.2f}")
print()

print("📏 Percentiles de DTE2:")
for p in percentiles:
    val = np.percentile(df_clean['DTE2'], p)
    above = df_clean[df_clean['DTE2'] >= val]['PnL_fwd_pts_50'].mean()
    below = df_clean[df_clean['DTE2'] < val]['PnL_fwd_pts_50'].mean()
    print(f"   P{p:2}: {val:6.0f} días → Encima: {above:6.2f} pts | Debajo: {below:6.2f} pts | Diff: {above-below:+7.2f}")
print()

print("📏 Percentiles de DTE_ratio:")
for p in percentiles:
    val = np.percentile(df_clean['DTE_ratio'], p)
    above = df_clean[df_clean['DTE_ratio'] >= val]['PnL_fwd_pts_50'].mean()
    below = df_clean[df_clean['DTE_ratio'] < val]['PnL_fwd_pts_50'].mean()
    print(f"   P{p:2}: {val:6.2f}x    → Encima: {above:6.2f} pts | Debajo: {below:6.2f} pts | Diff: {above-below:+7.2f}")
print()

# 14. RESUMEN EJECUTIVO
print("="*80)
print(" 📋 RESUMEN EJECUTIVO - VENCIMIENTOS")
print("="*80)
print()

print("🏆 VARIABLE DTE MÁS CORRELACIONADA:")
best_dte = ranking_df.iloc[0]
print(f"   {best_dte['Variable']}: Score = {best_dte['Combined_Score']:.4f}")
print()

print("✅ RECOMENDACIONES:")
print()

# Identificar rangos óptimos
best_dte1_range = dte1_analysis.loc[:, ('PnL_fwd_pts_50', 'mean')].idxmax()
best_dte2_range = dte2_analysis.loc[:, ('PnL_fwd_pts_50', 'mean')].idxmax()
best_ratio_range = ratio_analysis.loc[:, ('PnL_fwd_pts_50', 'mean')].idxmax()

print(f"1. RANGO ÓPTIMO DTE1: {best_dte1_range}")
print(f"   → Rentabilidad promedio: {dte1_analysis.loc[best_dte1_range, ('PnL_fwd_pts_50', 'mean')]:.2f} pts")
print()

print(f"2. RANGO ÓPTIMO DTE2: {best_dte2_range}")
print(f"   → Rentabilidad promedio: {dte2_analysis.loc[best_dte2_range, ('PnL_fwd_pts_50', 'mean')]:.2f} pts")
print()

print(f"3. RATIO ÓPTIMO DTE2/DTE1: {best_ratio_range}")
print(f"   → Rentabilidad promedio: {ratio_analysis.loc[best_ratio_range, ('PnL_fwd_pts_50', 'mean')]:.2f} pts")
print()

# 15. GUARDAR RESULTADOS
print("="*80)
print(" 💾 GUARDANDO RESULTADOS")
print("="*80)
print()

# Guardar ranking
ranking_df.to_csv('ranking_dte_variables.csv', index=False)
print("✓ Ranking guardado en: ranking_dte_variables.csv")

# Guardar combinaciones
combo_stats_filtered.to_csv('dte_combinaciones_stats.csv')
print("✓ Combinaciones guardadas en: dte_combinaciones_stats.csv")

# Guardar top y worst
top_combos[('PnL_fwd_pts_50', 'mean')].to_csv('dte_top_combos.csv')
worst_combos[('PnL_fwd_pts_50', 'mean')].to_csv('dte_worst_combos.csv')
print("✓ Top/Worst combos guardados")

print()
print("="*80)
print(" ✨ ANÁLISIS DE VENCIMIENTOS COMPLETADO")
print("="*80)
