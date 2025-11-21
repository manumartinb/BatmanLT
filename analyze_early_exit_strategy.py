#!/usr/bin/env python3
"""
Análisis: ¿Cerrar posiciones con PNLDV FWD deteriorado en ventana 25 mejora resultados vs mantener hasta ventana 50?

Hipótesis: Si PNLDV_fwd_25 está muy deteriorado (muy negativo), ¿es mejor cerrar en 25 o continuar hasta 50?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

print("="*80)
print("ANÁLISIS DE ESTRATEGIA DE SALIDA TEMPRANA")
print("="*80)

# Leer datos
df = pd.read_csv('/home/user/BatmanLT/PNLDV.csv')
print(f"\nTotal de registros: {len(df)}")

# Filtrar registros con datos válidos para ambas ventanas
cols_needed = ['PnLDV_fwd_25', 'PnL_fwd_pts_25', 'PnLDV_fwd_50', 'PnL_fwd_pts_50']
df_clean = df[cols_needed].dropna()
print(f"Registros con datos completos para ventanas 25 y 50: {len(df_clean)}")

# Crear quintiles de PNLDV_fwd_25
df_clean['quintil_pnldv_25'] = pd.qcut(
    df_clean['PnLDV_fwd_25'],
    q=5,
    labels=['Q1 (Más deteriorado)', 'Q2', 'Q3', 'Q4', 'Q5 (Mejor)']
)

# También crear categorías más extremas (deciles)
df_clean['decil_pnldv_25'] = pd.qcut(
    df_clean['PnLDV_fwd_25'],
    q=10,
    labels=[f'D{i}' for i in range(1, 11)]
)

print("\n" + "="*80)
print("1. ANÁLISIS GENERAL: ¿Qué pasa del periodo 25 al 50?")
print("="*80)

# Calcular diferencia de PnL entre ventana 50 y 25
df_clean['PnL_incremental_25_a_50'] = df_clean['PnL_fwd_pts_50'] - df_clean['PnL_fwd_pts_25']

print("\nEstadísticas de PnL incremental (ventana 50 - ventana 25):")
print(df_clean['PnL_incremental_25_a_50'].describe())

print("\n" + "="*80)
print("2. ANÁLISIS POR QUINTILES DE PNLDV_FWD_25")
print("="*80)

# Análisis por quintiles
quintil_analysis = df_clean.groupby('quintil_pnldv_25', observed=True).agg({
    'PnL_fwd_pts_25': ['count', 'mean', 'median', 'std'],
    'PnL_fwd_pts_50': ['mean', 'median', 'std'],
    'PnL_incremental_25_a_50': ['mean', 'median', 'std']
}).round(2)

print("\nComparación de PnL por quintil:")
print(quintil_analysis)

# Crear una tabla más clara
resumen_quintiles = pd.DataFrame({
    'Quintil': ['Q1 (Más deteriorado)', 'Q2', 'Q3', 'Q4', 'Q5 (Mejor)'],
    'N': df_clean.groupby('quintil_pnldv_25', observed=True).size().values,
    'PnL_25_media': df_clean.groupby('quintil_pnldv_25', observed=True)['PnL_fwd_pts_25'].mean().values,
    'PnL_50_media': df_clean.groupby('quintil_pnldv_25', observed=True)['PnL_fwd_pts_50'].mean().values,
    'Incremento_25_a_50': df_clean.groupby('quintil_pnldv_25', observed=True)['PnL_incremental_25_a_50'].mean().values
})

resumen_quintiles['Mejor_cerrar_en_25'] = resumen_quintiles['Incremento_25_a_50'] < 0

print("\n" + "-"*80)
print("RESUMEN POR QUINTILES:")
print("-"*80)
print(resumen_quintiles.to_string(index=False))

print("\n" + "="*80)
print("3. ANÁLISIS DETALLADO: QUINTIL 1 (MÁS DETERIORADO)")
print("="*80)

q1_data = df_clean[df_clean['quintil_pnldv_25'] == 'Q1 (Más deteriorado)']

print(f"\nRegistros en Q1: {len(q1_data)}")
print(f"\nPnL promedio en ventana 25: {q1_data['PnL_fwd_pts_25'].mean():.2f} pts")
print(f"PnL promedio en ventana 50: {q1_data['PnL_fwd_pts_50'].mean():.2f} pts")
print(f"Cambio incremental promedio: {q1_data['PnL_incremental_25_a_50'].mean():.2f} pts")

# Porcentaje que empeora vs mejora
empeora = (q1_data['PnL_incremental_25_a_50'] < 0).sum()
mejora = (q1_data['PnL_incremental_25_a_50'] > 0).sum()
igual = (q1_data['PnL_incremental_25_a_50'] == 0).sum()

print(f"\nDe ventana 25 a 50 en Q1:")
print(f"  - Empeora (pérdidas se amplían): {empeora} ({empeora/len(q1_data)*100:.1f}%)")
print(f"  - Mejora (se recupera): {mejora} ({mejora/len(q1_data)*100:.1f}%)")
print(f"  - Sin cambio: {igual} ({igual/len(q1_data)*100:.1f}%)")

# Test estadístico: ¿Es el cambio incremental significativamente diferente de 0?
t_stat, p_val = stats.ttest_1samp(q1_data['PnL_incremental_25_a_50'], 0)
print(f"\nTest t (H0: cambio incremental = 0):")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_val:.4e}")

if p_val < 0.05:
    if t_stat < 0:
        conclusion_q1 = "EMPEORA SIGNIFICATIVAMENTE"
    else:
        conclusion_q1 = "MEJORA SIGNIFICATIVAMENTE"
else:
    conclusion_q1 = "NO HAY CAMBIO SIGNIFICATIVO"

print(f"  Conclusión: {conclusion_q1}")

print("\n" + "="*80)
print("4. ANÁLISIS POR DECILES (MÁS GRANULAR)")
print("="*80)

decil_analysis = df_clean.groupby('decil_pnldv_25', observed=True).agg({
    'PnL_fwd_pts_25': ['mean'],
    'PnL_fwd_pts_50': ['mean'],
    'PnL_incremental_25_a_50': ['mean']
}).round(2)

print("\nCambio incremental por decil:")
decil_summary = pd.DataFrame({
    'Decil': [f'D{i}' for i in range(1, 11)],
    'PnL_25': df_clean.groupby('decil_pnldv_25', observed=True)['PnL_fwd_pts_25'].mean().values,
    'PnL_50': df_clean.groupby('decil_pnldv_25', observed=True)['PnL_fwd_pts_50'].mean().values,
    'Incremento': df_clean.groupby('decil_pnldv_25', observed=True)['PnL_incremental_25_a_50'].mean().values
})
decil_summary['Decisión'] = decil_summary['Incremento'].apply(
    lambda x: 'CERRAR en 25' if x < -5 else ('MANTENER hasta 50' if x > 5 else 'INDIFERENTE')
)

print(decil_summary.to_string(index=False))

print("\n" + "="*80)
print("5. IMPACTO FINANCIERO DE LA ESTRATEGIA")
print("="*80)

# Estrategia 1: Mantener todo hasta ventana 50
estrategia_mantener = df_clean['PnL_fwd_pts_50'].sum()

# Estrategia 2: Cerrar Q1 en ventana 25, resto en ventana 50
q1_cerrado_25 = q1_data['PnL_fwd_pts_25'].sum()
resto_50 = df_clean[df_clean['quintil_pnldv_25'] != 'Q1 (Más deteriorado)']['PnL_fwd_pts_50'].sum()
estrategia_salida_q1 = q1_cerrado_25 + resto_50

# Estrategia 3: Cerrar Q1 y Q2 en ventana 25, resto en ventana 50
q1q2_data = df_clean[df_clean['quintil_pnldv_25'].isin(['Q1 (Más deteriorado)', 'Q2'])]
q1q2_cerrado_25 = q1q2_data['PnL_fwd_pts_25'].sum()
resto_q3q4q5_50 = df_clean[~df_clean['quintil_pnldv_25'].isin(['Q1 (Más deteriorado)', 'Q2'])]['PnL_fwd_pts_50'].sum()
estrategia_salida_q1q2 = q1q2_cerrado_25 + resto_q3q4q5_50

print(f"\nComparación de estrategias (PnL total acumulado):")
print(f"\n1. MANTENER TODO hasta ventana 50:")
print(f"   PnL total: {estrategia_mantener:,.2f} pts")

print(f"\n2. CERRAR Q1 (más deteriorado) en ventana 25, resto en 50:")
print(f"   PnL total: {estrategia_salida_q1:,.2f} pts")
print(f"   Diferencia vs mantener: {estrategia_salida_q1 - estrategia_mantener:+,.2f} pts ({(estrategia_salida_q1/estrategia_mantener - 1)*100:+.2f}%)")

print(f"\n3. CERRAR Q1+Q2 (deteriorados) en ventana 25, resto en 50:")
print(f"   PnL total: {estrategia_salida_q1q2:,.2f} pts")
print(f"   Diferencia vs mantener: {estrategia_salida_q1q2 - estrategia_mantener:+,.2f} pts ({(estrategia_salida_q1q2/estrategia_mantener - 1)*100:+.2f}%)")

# Determinar mejor estrategia
mejor_estrategia = max(
    [('Mantener todo', estrategia_mantener),
     ('Cerrar Q1 en 25', estrategia_salida_q1),
     ('Cerrar Q1+Q2 en 25', estrategia_salida_q1q2)],
    key=lambda x: x[1]
)

print(f"\n{'='*80}")
print(f"MEJOR ESTRATEGIA: {mejor_estrategia[0]} (PnL: {mejor_estrategia[1]:,.2f} pts)")
print(f"{'='*80}")

# VISUALIZACIONES
print("\n" + "="*80)
print("GENERANDO VISUALIZACIONES...")
print("="*80)

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Comparación PnL 25 vs 50 por quintil
ax1 = fig.add_subplot(gs[0, :2])
x = np.arange(5)
width = 0.35
ax1.bar(x - width/2, resumen_quintiles['PnL_25_media'], width, label='PnL Ventana 25', alpha=0.8)
ax1.bar(x + width/2, resumen_quintiles['PnL_50_media'], width, label='PnL Ventana 50', alpha=0.8)
ax1.set_xlabel('Quintil de PNLDV_fwd_25')
ax1.set_ylabel('PnL Promedio (pts)')
ax1.set_title('Comparación PnL: Ventana 25 vs Ventana 50 por Quintil', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(['Q1\n(Más deteriorado)', 'Q2', 'Q3', 'Q4', 'Q5\n(Mejor)'])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# 2. Cambio incremental por quintil
ax2 = fig.add_subplot(gs[0, 2])
colors = ['red' if x < 0 else 'green' for x in resumen_quintiles['Incremento_25_a_50']]
ax2.barh(resumen_quintiles['Quintil'], resumen_quintiles['Incremento_25_a_50'], color=colors, alpha=0.7)
ax2.set_xlabel('Incremento Promedio\n(PnL_50 - PnL_25)')
ax2.set_title('Cambio de PnL\nentre ventanas', fontweight='bold', fontsize=10)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax2.grid(axis='x', alpha=0.3)

# 3. Distribución de cambio incremental por quintil
ax3 = fig.add_subplot(gs[1, :])
df_clean.boxplot(column='PnL_incremental_25_a_50', by='quintil_pnldv_25', ax=ax3)
ax3.set_xlabel('Quintil de PNLDV_fwd_25')
ax3.set_ylabel('PnL Incremental (ventana 50 - ventana 25)')
ax3.set_title('Distribución del Cambio de PnL entre Ventanas por Quintil', fontweight='bold')
ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Sin cambio')
ax3.legend()
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0, ha='center')
ax3.get_figure().suptitle('')

# 4. Scatter: PnL_25 vs PnL_50 para Q1
ax4 = fig.add_subplot(gs[2, 0])
ax4.scatter(q1_data['PnL_fwd_pts_25'], q1_data['PnL_fwd_pts_50'], alpha=0.3, s=20)
ax4.plot([q1_data['PnL_fwd_pts_25'].min(), q1_data['PnL_fwd_pts_25'].max()],
         [q1_data['PnL_fwd_pts_25'].min(), q1_data['PnL_fwd_pts_25'].max()],
         'r--', linewidth=2, label='Línea 45° (sin cambio)')
ax4.set_xlabel('PnL en Ventana 25')
ax4.set_ylabel('PnL en Ventana 50')
ax4.set_title('Q1 (Más deteriorado):\nPnL_25 vs PnL_50', fontweight='bold', fontsize=10)
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

# 5. Histograma de cambio incremental Q1
ax5 = fig.add_subplot(gs[2, 1])
ax5.hist(q1_data['PnL_incremental_25_a_50'], bins=30, alpha=0.7, edgecolor='black')
ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Sin cambio')
ax5.axvline(x=q1_data['PnL_incremental_25_a_50'].mean(), color='blue', linestyle='-', linewidth=2, label=f'Media: {q1_data["PnL_incremental_25_a_50"].mean():.1f}')
ax5.set_xlabel('Cambio incremental (pts)')
ax5.set_ylabel('Frecuencia')
ax5.set_title('Q1: Distribución del\nCambio Incremental', fontweight='bold', fontsize=10)
ax5.legend(fontsize=8)
ax5.grid(axis='y', alpha=0.3)

# 6. Comparación de estrategias
ax6 = fig.add_subplot(gs[2, 2])
estrategias = ['Mantener\ntodo', 'Cerrar Q1\nen 25', 'Cerrar Q1+Q2\nen 25']
valores = [estrategia_mantener, estrategia_salida_q1, estrategia_salida_q1q2]
colors_est = ['blue' if v == max(valores) else 'gray' for v in valores]
bars = ax6.bar(estrategias, valores, color=colors_est, alpha=0.7, edgecolor='black')
ax6.set_ylabel('PnL Total (pts)')
ax6.set_title('Comparación de\nEstrategias', fontweight='bold', fontsize=10)
ax6.grid(axis='y', alpha=0.3)
# Añadir valores en las barras
for bar, val in zip(bars, valores):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:,.0f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Análisis de Estrategia de Salida Temprana: ¿Cerrar en Ventana 25 o Mantener hasta 50?',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('/home/user/BatmanLT/early_exit_strategy_analysis.png', dpi=150, bbox_inches='tight')
print("Gráficos guardados en: early_exit_strategy_analysis.png")

# Guardar resultados
resumen_quintiles.to_csv('/home/user/BatmanLT/early_exit_quintiles_summary.csv', index=False)
decil_summary.to_csv('/home/user/BatmanLT/early_exit_deciles_summary.csv', index=False)

print("\n" + "="*80)
print("CONCLUSIÓN FINAL")
print("="*80)

print(f"\nPara posiciones con PNLDV_fwd_25 MÁS DETERIORADO (Quintil 1):")
print(f"  - PnL promedio en ventana 25: {q1_data['PnL_fwd_pts_25'].mean():.2f} pts")
print(f"  - PnL promedio en ventana 50: {q1_data['PnL_fwd_pts_50'].mean():.2f} pts")
print(f"  - Cambio incremental: {q1_data['PnL_incremental_25_a_50'].mean():.2f} pts")

if q1_data['PnL_incremental_25_a_50'].mean() < -5:
    print(f"\n✓ RECOMENDACIÓN: CERRAR posiciones deterioradas en ventana 25")
    print(f"  Las pérdidas tienden a AMPLIARSE significativamente hasta ventana 50")
elif q1_data['PnL_incremental_25_a_50'].mean() > 5:
    print(f"\n✗ RECOMENDACIÓN: MANTENER posiciones hasta ventana 50")
    print(f"  Las posiciones tienden a RECUPERARSE significativamente")
else:
    print(f"\n≈ RECOMENDACIÓN: INDIFERENTE")
    print(f"  No hay diferencia significativa entre cerrar en 25 o mantener hasta 50")

print(f"\nMejor estrategia global: {mejor_estrategia[0]}")
print(f"PnL total óptimo: {mejor_estrategia[1]:,.2f} pts")

diferencia_vs_mantener = mejor_estrategia[1] - estrategia_mantener
if diferencia_vs_mantener > 0:
    print(f"\nCerrar posiciones deterioradas anticipadamente MEJORA el resultado en {diferencia_vs_mantener:+,.2f} pts")
elif diferencia_vs_mantener < 0:
    print(f"\nMantener todas las posiciones hasta ventana 50 es MEJOR en {-diferencia_vs_mantener:,.2f} pts")
else:
    print(f"\nAmbas estrategias dan resultados similares")

print("\n" + "="*80)
print("Análisis completado.")
print("="*80)
