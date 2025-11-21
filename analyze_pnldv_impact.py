#!/usr/bin/env python3
"""
Análisis de la relación entre PNLDV FWD y PnL FWD
Objetivo: Confirmar o desmentir que cuando PnLDV se hunde más, el PnL FWD empeora
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Leer el archivo
print("Cargando datos...")
df = pd.read_csv('/home/user/BatmanLT/PNLDV.csv')
print(f"Total de registros: {len(df)}")

# Ventanas a analizar
ventanas = [5, 25, 50]

# Crear análisis para cada ventana
resultados = {}

for ventana in ventanas:
    print(f"\n{'='*80}")
    print(f"ANÁLISIS PARA VENTANA {ventana}")
    print(f"{'='*80}")

    # Nombres de columnas
    col_pnldv = f'PnLDV_fwd_{ventana:02d}'
    col_pnl = f'PnL_fwd_pts_{ventana:02d}'

    # Filtrar datos válidos (sin NaN)
    df_clean = df[[col_pnldv, col_pnl]].dropna()
    print(f"Registros válidos: {len(df_clean)} de {len(df)}")

    # Estadísticas descriptivas
    print(f"\n--- ESTADÍSTICAS DESCRIPTIVAS ---")
    print(f"\n{col_pnldv}:")
    print(df_clean[col_pnldv].describe())
    print(f"\n{col_pnl}:")
    print(df_clean[col_pnl].describe())

    # Correlación de Pearson (lineal)
    pearson_corr, pearson_pval = stats.pearsonr(df_clean[col_pnldv], df_clean[col_pnl])

    # Correlación de Spearman (monotónica, más robusta a outliers)
    spearman_corr, spearman_pval = stats.spearmanr(df_clean[col_pnldv], df_clean[col_pnl])

    print(f"\n--- CORRELACIONES ---")
    print(f"Correlación de Pearson:  {pearson_corr:7.4f} (p-value: {pearson_pval:.2e})")
    print(f"Correlación de Spearman: {spearman_corr:7.4f} (p-value: {spearman_pval:.2e})")

    # Interpretación
    if pearson_pval < 0.001:
        significancia = "ALTAMENTE SIGNIFICATIVA"
    elif pearson_pval < 0.05:
        significancia = "SIGNIFICATIVA"
    else:
        significancia = "NO SIGNIFICATIVA"

    print(f"Significancia estadística: {significancia}")

    # Análisis por quintiles de PnLDV
    print(f"\n--- ANÁLISIS POR QUINTILES DE {col_pnldv} ---")
    df_clean['quintil_pnldv'] = pd.qcut(df_clean[col_pnldv], q=5, labels=['Q1 (Más hundido)', 'Q2', 'Q3', 'Q4', 'Q5 (Menos hundido)'])

    quintiles_stats = df_clean.groupby('quintil_pnldv')[col_pnl].agg([
        ('Count', 'count'),
        ('Media', 'mean'),
        ('Mediana', 'median'),
        ('Std', 'std'),
        ('Min', 'min'),
        ('Max', 'max')
    ])
    print(quintiles_stats)

    # Test de tendencia lineal en quintiles
    quintil_means = df_clean.groupby('quintil_pnldv')[col_pnl].mean().values
    quintil_nums = np.arange(1, 6)
    slope, intercept, r_value, p_value, std_err = stats.linregress(quintil_nums, quintil_means)

    print(f"\nTendencia lineal entre quintiles:")
    print(f"  Pendiente: {slope:.4f} (positiva = a más hundido PnLDV, mejor PnL)")
    print(f"  R²: {r_value**2:.4f}")
    print(f"  p-value: {p_value:.4f}")

    # Guardar resultados
    resultados[ventana] = {
        'n_registros': len(df_clean),
        'pearson_corr': pearson_corr,
        'pearson_pval': pearson_pval,
        'spearman_corr': spearman_corr,
        'spearman_pval': spearman_pval,
        'quintiles_stats': quintiles_stats,
        'tendencia_slope': slope,
        'tendencia_pval': p_value,
        'df_clean': df_clean
    }

# VISUALIZACIONES
print(f"\n{'='*80}")
print("GENERANDO VISUALIZACIONES...")
print(f"{'='*80}")

fig, axes = plt.subplots(3, 2, figsize=(18, 15))
fig.suptitle('Análisis de Relación entre PNLDV FWD y PnL FWD', fontsize=16, fontweight='bold')

for idx, ventana in enumerate(ventanas):
    col_pnldv = f'PnLDV_fwd_{ventana:02d}'
    col_pnl = f'PnL_fwd_pts_{ventana:02d}'
    df_clean = resultados[ventana]['df_clean']

    # Scatter plot con línea de regresión
    ax1 = axes[idx, 0]
    ax1.scatter(df_clean[col_pnldv], df_clean[col_pnl], alpha=0.3, s=10)

    # Línea de regresión
    z = np.polyfit(df_clean[col_pnldv], df_clean[col_pnl], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_clean[col_pnldv].min(), df_clean[col_pnldv].max(), 100)
    ax1.plot(x_line, p(x_line), "r-", linewidth=2, label=f'Regresión lineal')

    ax1.set_xlabel(col_pnldv, fontsize=10)
    ax1.set_ylabel(col_pnl, fontsize=10)
    ax1.set_title(f'Ventana {ventana} - Scatter Plot\nPearson r={resultados[ventana]["pearson_corr"]:.4f}', fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot por quintiles
    ax2 = axes[idx, 1]
    df_clean.boxplot(column=col_pnl, by='quintil_pnldv', ax=ax2)
    ax2.set_xlabel('Quintil de PNLDV (Q1 = más hundido)', fontsize=10)
    ax2.set_ylabel(col_pnl, fontsize=10)
    ax2.set_title(f'Ventana {ventana} - PnL FWD por Quintil de PNLDV', fontsize=11)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Remover el título automático de pandas
    ax2.get_figure().suptitle('')

plt.tight_layout()
plt.savefig('/home/user/BatmanLT/pnldv_analysis.png', dpi=150, bbox_inches='tight')
print("Gráficos guardados en: /home/user/BatmanLT/pnldv_analysis.png")

# RESUMEN EJECUTIVO
print(f"\n{'='*80}")
print("RESUMEN EJECUTIVO")
print(f"{'='*80}\n")

print("HIPÓTESIS: 'Conforme más se hunde PnLDV, peor es el PnL FWD'\n")
print("(Una correlación POSITIVA confirmaría la hipótesis, ya que valores más negativos")
print("de PnLDV deberían asociarse con valores más negativos de PnL FWD)\n")

conclusiones = []

for ventana in ventanas:
    r = resultados[ventana]
    print(f"VENTANA {ventana}:")
    print(f"  - Correlación de Pearson:  {r['pearson_corr']:7.4f} (p={r['pearson_pval']:.2e})")
    print(f"  - Correlación de Spearman: {r['spearman_corr']:7.4f} (p={r['spearman_pval']:.2e})")

    # Análisis de dirección
    if r['pearson_corr'] > 0:
        direccion = "POSITIVA"
        interpretacion = "Cuando PnLDV baja (se hunde más), PnL FWD también tiende a bajar (empeora)"
        confirma = "CONFIRMA"
    else:
        direccion = "NEGATIVA"
        interpretacion = "Cuando PnLDV baja (se hunde más), PnL FWD tiende a SUBIR (mejora)"
        confirma = "DESMIENTE"

    # Fuerza de la correlación
    abs_corr = abs(r['pearson_corr'])
    if abs_corr > 0.7:
        fuerza = "MUY FUERTE"
    elif abs_corr > 0.5:
        fuerza = "FUERTE"
    elif abs_corr > 0.3:
        fuerza = "MODERADA"
    elif abs_corr > 0.1:
        fuerza = "DÉBIL"
    else:
        fuerza = "MUY DÉBIL"

    print(f"  - Dirección: {direccion}")
    print(f"  - Fuerza: {fuerza}")
    print(f"  - Interpretación: {interpretacion}")
    print(f"  - CONCLUSIÓN: {confirma} la hipótesis")
    print()

    conclusiones.append(confirma)

# Conclusión general
if all(c == "CONFIRMA" for c in conclusiones):
    conclusion_final = "CONFIRMA COMPLETAMENTE"
elif all(c == "DESMIENTE" for c in conclusiones):
    conclusion_final = "DESMIENTE COMPLETAMENTE"
else:
    conclusion_final = "RESULTADOS MIXTOS"

print(f"\n{'='*80}")
print(f"CONCLUSIÓN FINAL: El análisis {conclusion_final} la hipótesis")
print(f"{'='*80}\n")

# Guardar resultados en CSV
resumen_df = pd.DataFrame({
    'Ventana': ventanas,
    'N_Registros': [resultados[v]['n_registros'] for v in ventanas],
    'Pearson_r': [resultados[v]['pearson_corr'] for v in ventanas],
    'Pearson_pval': [resultados[v]['pearson_pval'] for v in ventanas],
    'Spearman_r': [resultados[v]['spearman_corr'] for v in ventanas],
    'Spearman_pval': [resultados[v]['spearman_pval'] for v in ventanas],
    'Significativo': ['Sí' if resultados[v]['pearson_pval'] < 0.05 else 'No' for v in ventanas],
    'Confirma_Hipotesis': conclusiones
})

resumen_df.to_csv('/home/user/BatmanLT/pnldv_analysis_summary.csv', index=False)
print("Resumen guardado en: /home/user/BatmanLT/pnldv_analysis_summary.csv")

print("\nAnálisis completado.")
