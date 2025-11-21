"""
Análisis de Estrategia de Cierre Anticipado en W=25
===================================================

Pregunta: ¿Hubiera mejorado el PnL general cerrar posiciones en W=25
cuando el PnLDV muestra deterioro, en lugar de dejarlas correr hasta W=50?

Estrategia Activa: Cierre anticipado en W=25 si PnLDV ha sufrido deterioro
Estrategia Pasiva: Dejar correr todas las operaciones hasta W=50
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (18, 12)

print("="*100)
print("ANÁLISIS DE ESTRATEGIA DE CIERRE ANTICIPADO EN W=25 BASADO EN DETERIORO DEL PNLDV")
print("="*100)
print(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Cargar datos
df = pd.read_csv('PNLDV.csv')
print(f"Dataset: {len(df)} operaciones\n")

# Calcular deterioro del PnLDV en W=25
df['delta_pnldv_25'] = df['PnLDV_fwd_25'] - df['PnLDV']
df['pct_change_pnldv_25'] = ((df['PnLDV_fwd_25'] - df['PnLDV']) / df['PnLDV'].abs()) * 100

# Filtrar operaciones válidas (tienen datos en ambas ventanas W=25 y W=50)
df_validas = df[df['PnL_fwd_pts_25'].notna() & df['PnL_fwd_pts_50'].notna()].copy()
print(f"Operaciones con datos válidos en W=25 y W=50: {len(df_validas)}")
print(f"(Excluidas {len(df) - len(df_validas)} operaciones por datos faltantes)\n")

# ============================================================================
# ANÁLISIS CON MÚLTIPLES UMBRALES DE DETERIORO
# ============================================================================

print("="*100)
print("1. ANÁLISIS DE MÚLTIPLES UMBRALES DE DETERIORO")
print("="*100)

# Definir diferentes umbrales de deterioro
umbrales = [
    {'nombre': 'Sin deterioro (delta >= 0)', 'condicion': lambda x: x >= 0},
    {'nombre': 'Deterioro cualquiera (delta < 0)', 'condicion': lambda x: x < 0},
    {'nombre': 'Deterioro leve (delta < -20)', 'condicion': lambda x: x < -20},
    {'nombre': 'Deterioro moderado (delta < -50)', 'condicion': lambda x: x < -50},
    {'nombre': 'Deterioro fuerte (delta < -75)', 'condicion': lambda x: x < -75},
    {'nombre': 'Deterioro muy fuerte (delta < -100)', 'condicion': lambda x: x < -100},
]

resultados_umbrales = []

for umbral_def in umbrales:
    nombre = umbral_def['nombre']
    condicion = umbral_def['condicion']

    # Marcar operaciones que se cerrarían en W=25
    df_validas['cerrar_w25'] = condicion(df_validas['delta_pnldv_25'])

    # Calcular PnL realizado según estrategia
    # Si se cierra en W=25: usar PnL_fwd_pts_25
    # Si continúa hasta W=50: usar PnL_fwd_pts_50
    df_validas['pnl_estrategia_activa'] = np.where(
        df_validas['cerrar_w25'],
        df_validas['PnL_fwd_pts_25'],
        df_validas['PnL_fwd_pts_50']
    )

    # Estrategia pasiva: siempre W=50
    df_validas['pnl_estrategia_pasiva'] = df_validas['PnL_fwd_pts_50']

    # Métricas de la estrategia
    n_cerradas_w25 = df_validas['cerrar_w25'].sum()
    pct_cerradas = (n_cerradas_w25 / len(df_validas)) * 100

    # PnL total
    pnl_total_activa = df_validas['pnl_estrategia_activa'].sum()
    pnl_total_pasiva = df_validas['pnl_estrategia_pasiva'].sum()
    mejora_absoluta = pnl_total_activa - pnl_total_pasiva
    mejora_porcentual = (mejora_absoluta / abs(pnl_total_pasiva)) * 100 if pnl_total_pasiva != 0 else 0

    # PnL promedio
    pnl_medio_activa = df_validas['pnl_estrategia_activa'].mean()
    pnl_medio_pasiva = df_validas['pnl_estrategia_pasiva'].mean()
    mejora_media = pnl_medio_activa - pnl_medio_pasiva

    # Win rate
    wr_activa = (df_validas['pnl_estrategia_activa'] > 0).sum() / len(df_validas) * 100
    wr_pasiva = (df_validas['pnl_estrategia_pasiva'] > 0).sum() / len(df_validas) * 100
    mejora_wr = wr_activa - wr_pasiva

    # Mediana
    mediana_activa = df_validas['pnl_estrategia_activa'].median()
    mediana_pasiva = df_validas['pnl_estrategia_pasiva'].median()

    # Desviación estándar (riesgo)
    std_activa = df_validas['pnl_estrategia_activa'].std()
    std_pasiva = df_validas['pnl_estrategia_pasiva'].std()
    reduccion_riesgo = ((std_pasiva - std_activa) / std_pasiva) * 100 if std_pasiva != 0 else 0

    # Sharpe ratio aproximado (asumiendo rf=0)
    sharpe_activa = pnl_medio_activa / std_activa if std_activa != 0 else 0
    sharpe_pasiva = pnl_medio_pasiva / std_pasiva if std_pasiva != 0 else 0

    # PnL de operaciones cerradas en W=25
    if n_cerradas_w25 > 0:
        pnl_cerradas = df_validas[df_validas['cerrar_w25']]['PnL_fwd_pts_25'].mean()
        pnl_cerradas_hubieran_sido = df_validas[df_validas['cerrar_w25']]['PnL_fwd_pts_50'].mean()
        ahorro_por_cierre = pnl_cerradas - pnl_cerradas_hubieran_sido
    else:
        pnl_cerradas = 0
        pnl_cerradas_hubieran_sido = 0
        ahorro_por_cierre = 0

    resultados_umbrales.append({
        'umbral': nombre,
        'n_cerradas': n_cerradas_w25,
        'pct_cerradas': pct_cerradas,
        'pnl_total_activa': pnl_total_activa,
        'pnl_total_pasiva': pnl_total_pasiva,
        'mejora_absoluta': mejora_absoluta,
        'mejora_porcentual': mejora_porcentual,
        'pnl_medio_activa': pnl_medio_activa,
        'pnl_medio_pasiva': pnl_medio_pasiva,
        'mejora_media': mejora_media,
        'wr_activa': wr_activa,
        'wr_pasiva': wr_pasiva,
        'mejora_wr': mejora_wr,
        'mediana_activa': mediana_activa,
        'mediana_pasiva': mediana_pasiva,
        'std_activa': std_activa,
        'std_pasiva': std_pasiva,
        'reduccion_riesgo': reduccion_riesgo,
        'sharpe_activa': sharpe_activa,
        'sharpe_pasiva': sharpe_pasiva,
        'pnl_cerradas': pnl_cerradas,
        'pnl_cerradas_hubieran_sido': pnl_cerradas_hubieran_sido,
        'ahorro_por_cierre': ahorro_por_cierre
    })

df_resultados = pd.DataFrame(resultados_umbrales)

print("\nRESUMEN COMPARATIVO POR UMBRAL DE DETERIORO:")
print("="*100)
print(f"{'Umbral':<35} {'N Cerr.':<10} {'%Cerr.':<8} {'PnL Total':<12} {'Mejora':<12} {'Mejora %':<10}")
print("-"*100)

for idx, row in df_resultados.iterrows():
    signo = "✅" if row['mejora_absoluta'] > 0 else "❌"
    print(f"{row['umbral']:<35} {row['n_cerradas']:<10.0f} {row['pct_cerradas']:<8.1f} "
          f"{row['pnl_total_activa']:<12.1f} {row['mejora_absoluta']:<+12.1f} {row['mejora_porcentual']:<+10.2f}% {signo}")

print("\n" + "="*100)
print("2. ANÁLISIS DETALLADO DEL MEJOR UMBRAL")
print("="*100)

# Identificar mejor umbral (mayor mejora absoluta)
idx_mejor = df_resultados['mejora_absoluta'].idxmax()
mejor_umbral = df_resultados.loc[idx_mejor]

print(f"\n🏆 MEJOR UMBRAL: {mejor_umbral['umbral']}")
print("-"*100)

print(f"\n📊 MÉTRICAS COMPARATIVAS:")
print(f"\n  Operaciones cerradas anticipadamente en W=25: {mejor_umbral['n_cerradas']:.0f} ({mejor_umbral['pct_cerradas']:.1f}%)")
print(f"  Operaciones que continuaron hasta W=50: {len(df_validas) - mejor_umbral['n_cerradas']:.0f} ({100-mejor_umbral['pct_cerradas']:.1f}%)")

print(f"\n  💰 PnL TOTAL:")
print(f"    Estrategia ACTIVA (cierre en W=25):  {mejor_umbral['pnl_total_activa']:>12.2f} pts")
print(f"    Estrategia PASIVA (hold hasta W=50): {mejor_umbral['pnl_total_pasiva']:>12.2f} pts")
print(f"    MEJORA ABSOLUTA:                      {mejor_umbral['mejora_absoluta']:>+12.2f} pts ({mejor_umbral['mejora_porcentual']:+.2f}%)")

print(f"\n  📈 PnL PROMEDIO por operación:")
print(f"    Estrategia ACTIVA:  {mejor_umbral['pnl_medio_activa']:>8.2f} pts")
print(f"    Estrategia PASIVA:  {mejor_umbral['pnl_medio_pasiva']:>8.2f} pts")
print(f"    MEJORA:             {mejor_umbral['mejora_media']:>+8.2f} pts")

print(f"\n  📊 PnL MEDIANA:")
print(f"    Estrategia ACTIVA:  {mejor_umbral['mediana_activa']:>8.2f} pts")
print(f"    Estrategia PASIVA:  {mejor_umbral['mediana_pasiva']:>8.2f} pts")

print(f"\n  🎯 WIN RATE:")
print(f"    Estrategia ACTIVA:  {mejor_umbral['wr_activa']:>6.2f}%")
print(f"    Estrategia PASIVA:  {mejor_umbral['wr_pasiva']:>6.2f}%")
print(f"    MEJORA:             {mejor_umbral['mejora_wr']:>+6.2f}%")

print(f"\n  ⚠️ RIESGO (desviación estándar):")
print(f"    Estrategia ACTIVA:  {mejor_umbral['std_activa']:>8.2f} pts")
print(f"    Estrategia PASIVA:  {mejor_umbral['std_pasiva']:>8.2f} pts")
print(f"    REDUCCIÓN DE RIESGO: {mejor_umbral['reduccion_riesgo']:>+6.2f}%")

print(f"\n  📐 SHARPE RATIO (aproximado):")
print(f"    Estrategia ACTIVA:  {mejor_umbral['sharpe_activa']:>6.3f}")
print(f"    Estrategia PASIVA:  {mejor_umbral['sharpe_pasiva']:>6.3f}")
print(f"    MEJORA:             {mejor_umbral['sharpe_activa'] - mejor_umbral['sharpe_pasiva']:>+6.3f}")

print(f"\n  💡 ANÁLISIS DE OPERACIONES CERRADAS:")
print(f"    PnL promedio al cerrar en W=25:         {mejor_umbral['pnl_cerradas']:>+8.2f} pts")
print(f"    PnL promedio si hubieran llegado a W=50: {mejor_umbral['pnl_cerradas_hubieran_sido']:>+8.2f} pts")
print(f"    AHORRO por cierre anticipado:            {mejor_umbral['ahorro_por_cierre']:>+8.2f} pts")

# ============================================================================
# ANÁLISIS DETALLADO: ¿QUÉ PASÓ CON LAS OPERACIONES CERRADAS?
# ============================================================================

print("\n" + "="*100)
print("3. ANÁLISIS DETALLADO DE OPERACIONES CERRADAS ANTICIPADAMENTE")
print("="*100)

# Recrear la estrategia con el mejor umbral
mejor_umbral_func = umbrales[idx_mejor]['condicion']
df_validas['cerrar_w25_mejor'] = mejor_umbral_func(df_validas['delta_pnldv_25'])

# Separar grupos
cerradas = df_validas[df_validas['cerrar_w25_mejor']].copy()
continuadas = df_validas[~df_validas['cerrar_w25_mejor']].copy()

if len(cerradas) > 0:
    print(f"\n📍 OPERACIONES CERRADAS EN W=25 ({len(cerradas)} operaciones):")
    print("-"*100)

    # Distribución de resultados
    cerradas['resultado_w25'] = pd.cut(
        cerradas['PnL_fwd_pts_25'],
        bins=[-np.inf, -100, -50, 0, 50, 100, np.inf],
        labels=['Pérdida fuerte (< -100)', 'Pérdida moderada (-100 a -50)',
                'Pérdida leve (-50 a 0)', 'Ganancia leve (0 a 50)',
                'Ganancia moderada (50 a 100)', 'Ganancia fuerte (> 100)']
    )

    dist_cerradas = cerradas['resultado_w25'].value_counts().sort_index()
    print("\n  Distribución de PnL en W=25 (al momento del cierre):")
    for cat, count in dist_cerradas.items():
        pct = (count / len(cerradas)) * 100
        print(f"    {cat:<35} {count:>5} ops ({pct:>5.1f}%)")

    # ¿Qué hubiera pasado si hubieran continuado?
    cerradas['diferencia_w50_w25'] = cerradas['PnL_fwd_pts_50'] - cerradas['PnL_fwd_pts_25']

    evitadas_peores = (cerradas['diferencia_w50_w25'] < 0).sum()
    hubieran_mejorado = (cerradas['diferencia_w50_w25'] > 0).sum()

    print(f"\n  ¿Qué hubiera pasado si hubieran continuado hasta W=50?")
    print(f"    Operaciones que EVITARON empeorar:     {evitadas_peores} ({evitadas_peores/len(cerradas)*100:.1f}%)")
    print(f"    Operaciones que HUBIERAN mejorado:     {hubieran_mejorado} ({hubieran_mejorado/len(cerradas)*100:.1f}%)")
    print(f"    Deterioro promedio evitado:            {-cerradas['diferencia_w50_w25'].mean():+.2f} pts")
    print(f"    Deterioro mediano evitado:             {-cerradas['diferencia_w50_w25'].median():+.2f} pts")

    # Casos extremos
    peor_decision = cerradas.loc[cerradas['diferencia_w50_w25'].idxmax()]
    mejor_decision = cerradas.loc[cerradas['diferencia_w50_w25'].idxmin()]

    print(f"\n  📊 Casos extremos:")
    print(f"    Peor decisión de cierre (mayor oportunidad perdida):")
    print(f"      - PnL en W=25: {peor_decision['PnL_fwd_pts_25']:+.2f} pts")
    print(f"      - PnL hubiera sido en W=50: {peor_decision['PnL_fwd_pts_50']:+.2f} pts")
    print(f"      - Oportunidad perdida: {peor_decision['diferencia_w50_w25']:+.2f} pts")

    print(f"\n    Mejor decisión de cierre (mayor pérdida evitada):")
    print(f"      - PnL en W=25: {mejor_decision['PnL_fwd_pts_25']:+.2f} pts")
    print(f"      - PnL hubiera sido en W=50: {mejor_decision['PnL_fwd_pts_50']:+.2f} pts")
    print(f"      - Pérdida evitada: {-mejor_decision['diferencia_w50_w25']:+.2f} pts")

if len(continuadas) > 0:
    print(f"\n\n📍 OPERACIONES QUE CONTINUARON HASTA W=50 ({len(continuadas)} operaciones):")
    print("-"*100)

    pnl_medio_continuadas = continuadas['PnL_fwd_pts_50'].mean()
    wr_continuadas = (continuadas['PnL_fwd_pts_50'] > 0).sum() / len(continuadas) * 100

    print(f"    PnL promedio en W=50:  {pnl_medio_continuadas:+.2f} pts")
    print(f"    Win rate:              {wr_continuadas:.1f}%")

# ============================================================================
# ANÁLISIS POR CATEGORÍA DE DETERIORO
# ============================================================================

print("\n" + "="*100)
print("4. MATRIZ DE DECISIÓN: IMPACTO POR NIVEL DE DETERIORO")
print("="*100)

# Crear categorías de deterioro más granulares
df_validas['categoria_deterioro'] = pd.cut(
    df_validas['delta_pnldv_25'],
    bins=[-np.inf, -100, -75, -50, -20, 0, 50, 100, np.inf],
    labels=['Muy fuerte (< -100)', 'Fuerte (-100 a -75)', 'Moderado (-75 a -50)',
            'Leve (-50 a -20)', 'Mínimo (-20 a 0)', 'Sin deterioro (0 a 50)',
            'Mejora moderada (50 a 100)', 'Mejora fuerte (> 100)']
)

analisis_categoria = df_validas.groupby('categoria_deterioro').agg({
    'PnL_fwd_pts_25': ['count', 'mean', 'median'],
    'PnL_fwd_pts_50': ['mean', 'median'],
}).round(2)

analisis_categoria.columns = ['N_ops', 'PnL_W25_mean', 'PnL_W25_median', 'PnL_W50_mean', 'PnL_W50_median']
analisis_categoria['Diferencia_mean'] = analisis_categoria['PnL_W50_mean'] - analisis_categoria['PnL_W25_mean']
analisis_categoria['Recomendacion'] = analisis_categoria['Diferencia_mean'].apply(
    lambda x: '✅ Cerrar en W=25' if x < -10 else ('⚠️ Evaluar caso a caso' if x < 10 else '❌ Dejar correr')
)

print("\nPnL promedio según nivel de deterioro del PnLDV en W=25:")
print(analisis_categoria.to_string())

# ============================================================================
# VISUALIZACIONES
# ============================================================================

print("\n" + "="*100)
print("GENERANDO VISUALIZACIONES...")
print("="*100)

# Figura 1: Comparación de estrategias por umbral
fig1 = plt.figure(figsize=(20, 12))

# Subplot 1: Mejora absoluta por umbral
plt.subplot(2, 3, 1)
colores = ['green' if x > 0 else 'red' for x in df_resultados['mejora_absoluta']]
bars = plt.bar(range(len(df_resultados)), df_resultados['mejora_absoluta'], color=colores, alpha=0.7)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Umbral de Deterioro', fontsize=11)
plt.ylabel('Mejora en PnL Total (pts)', fontsize=11)
plt.title('Mejora de PnL Total por Umbral de Cierre', fontsize=13, fontweight='bold')
plt.xticks(range(len(df_resultados)), [f"U{i+1}" for i in range(len(df_resultados))], rotation=0)
plt.grid(True, alpha=0.3, axis='y')

# Marcar el mejor
plt.bar(idx_mejor, df_resultados.loc[idx_mejor, 'mejora_absoluta'],
        color='gold', alpha=0.9, edgecolor='darkgreen', linewidth=3)

# Subplot 2: PnL promedio por umbral
plt.subplot(2, 3, 2)
x = range(len(df_resultados))
width = 0.35
plt.bar([i - width/2 for i in x], df_resultados['pnl_medio_activa'], width,
        label='Estrategia Activa', color='blue', alpha=0.7)
plt.bar([i + width/2 for i in x], df_resultados['pnl_medio_pasiva'], width,
        label='Estrategia Pasiva', color='orange', alpha=0.7)
plt.xlabel('Umbral de Deterioro', fontsize=11)
plt.ylabel('PnL Promedio (pts)', fontsize=11)
plt.title('PnL Promedio: Activa vs Pasiva', fontsize=13, fontweight='bold')
plt.xticks(range(len(df_resultados)), [f"U{i+1}" for i in range(len(df_resultados))])
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Subplot 3: Win Rate
plt.subplot(2, 3, 3)
plt.plot(range(len(df_resultados)), df_resultados['wr_activa'],
         marker='o', linewidth=2, markersize=8, label='Estrategia Activa', color='blue')
plt.plot(range(len(df_resultados)), df_resultados['wr_pasiva'],
         marker='s', linewidth=2, markersize=8, label='Estrategia Pasiva', color='orange')
plt.xlabel('Umbral de Deterioro', fontsize=11)
plt.ylabel('Win Rate (%)', fontsize=11)
plt.title('Win Rate: Activa vs Pasiva', fontsize=13, fontweight='bold')
plt.xticks(range(len(df_resultados)), [f"U{i+1}" for i in range(len(df_resultados))])
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 4: Porcentaje de operaciones cerradas
plt.subplot(2, 3, 4)
plt.bar(range(len(df_resultados)), df_resultados['pct_cerradas'], color='purple', alpha=0.7)
plt.xlabel('Umbral de Deterioro', fontsize=11)
plt.ylabel('% Operaciones Cerradas en W=25', fontsize=11)
plt.title('Tasa de Cierre Anticipado por Umbral', fontsize=13, fontweight='bold')
plt.xticks(range(len(df_resultados)), [f"U{i+1}" for i in range(len(df_resultados))])
plt.grid(True, alpha=0.3, axis='y')

# Subplot 5: Sharpe Ratio
plt.subplot(2, 3, 5)
x = range(len(df_resultados))
width = 0.35
plt.bar([i - width/2 for i in x], df_resultados['sharpe_activa'], width,
        label='Estrategia Activa', color='blue', alpha=0.7)
plt.bar([i + width/2 for i in x], df_resultados['sharpe_pasiva'], width,
        label='Estrategia Pasiva', color='orange', alpha=0.7)
plt.xlabel('Umbral de Deterioro', fontsize=11)
plt.ylabel('Sharpe Ratio', fontsize=11)
plt.title('Sharpe Ratio: Activa vs Pasiva', fontsize=13, fontweight='bold')
plt.xticks(range(len(df_resultados)), [f"U{i+1}" for i in range(len(df_resultados))])
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Subplot 6: Reducción de riesgo
plt.subplot(2, 3, 6)
colores_riesgo = ['green' if x > 0 else 'red' for x in df_resultados['reduccion_riesgo']]
plt.bar(range(len(df_resultados)), df_resultados['reduccion_riesgo'], color=colores_riesgo, alpha=0.7)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Umbral de Deterioro', fontsize=11)
plt.ylabel('Reducción de Riesgo (%)', fontsize=11)
plt.title('Reducción de Volatilidad vs Estrategia Pasiva', fontsize=13, fontweight='bold')
plt.xticks(range(len(df_resultados)), [f"U{i+1}" for i in range(len(df_resultados))])
plt.grid(True, alpha=0.3, axis='y')

# Leyenda de umbrales
leyenda_text = "UMBRALES:\n" + "\n".join([f"U{i+1}: {u['nombre']}" for i, u in enumerate(umbrales)])
plt.figtext(0.99, 0.01, leyenda_text, fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('comparacion_estrategias_umbrales.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 1 guardado: comparacion_estrategias_umbrales.png")

# Figura 2: Análisis del mejor umbral
fig2 = plt.figure(figsize=(20, 10))

# Recrear con mejor umbral
df_validas['cerrar_mejor'] = mejor_umbral_func(df_validas['delta_pnldv_25'])
cerradas_mejor = df_validas[df_validas['cerrar_mejor']].copy()
continuadas_mejor = df_validas[~df_validas['cerrar_mejor']].copy()

# Subplot 1: Distribución de PnL - Cerradas en W=25 vs lo que hubieran sido en W=50
plt.subplot(2, 3, 1)
if len(cerradas_mejor) > 0:
    plt.hist(cerradas_mejor['PnL_fwd_pts_25'], bins=30, alpha=0.6, label='PnL en W=25 (cierre)', color='blue')
    plt.hist(cerradas_mejor['PnL_fwd_pts_50'], bins=30, alpha=0.6, label='PnL que hubiera sido en W=50', color='red')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('PnL (pts)', fontsize=11)
    plt.ylabel('Frecuencia', fontsize=11)
    plt.title(f'Distribución PnL: Operaciones Cerradas en W=25\n({len(cerradas_mejor)} ops)',
              fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

# Subplot 2: Scatter - Deterioro vs Impacto de cerrar
plt.subplot(2, 3, 2)
if len(cerradas_mejor) > 0:
    cerradas_mejor['impacto_cierre'] = cerradas_mejor['PnL_fwd_pts_25'] - cerradas_mejor['PnL_fwd_pts_50']
    scatter = plt.scatter(cerradas_mejor['delta_pnldv_25'], cerradas_mejor['impacto_cierre'],
                         c=cerradas_mejor['impacto_cierre'], cmap='RdYlGn', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Impacto del cierre (pts)')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlabel('Deterioro PnLDV en W=25 (pts)', fontsize=11)
    plt.ylabel('Impacto del Cierre (PnL_W25 - PnL_W50)', fontsize=11)
    plt.title('Relación: Deterioro vs Beneficio del Cierre', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)

# Subplot 3: Box plot comparativo
plt.subplot(2, 3, 3)
data_box = [
    df_validas['PnL_fwd_pts_50'],  # Pasiva
    df_validas[df_validas['cerrar_mejor']]['PnL_fwd_pts_25'],  # Activa: cerradas en W=25
    df_validas[~df_validas['cerrar_mejor']]['PnL_fwd_pts_50']  # Activa: continuadas
]
labels_box = ['Pasiva\n(todas a W=50)', 'Activa\n(cerradas en W=25)', 'Activa\n(continuadas a W=50)']
bp = plt.boxplot(data_box, labels=labels_box, patch_artist=True)
colors = ['orange', 'lightblue', 'lightgreen']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.ylabel('PnL (pts)', fontsize=11)
plt.title('Distribución de PnL por Estrategia', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Subplot 4: Evolución temporal simulada
plt.subplot(2, 3, 4)
# PnL acumulado
pnl_acum_pasiva = df_validas['PnL_fwd_pts_50'].cumsum()
pnl_w25 = np.where(df_validas['cerrar_mejor'], df_validas['PnL_fwd_pts_25'], df_validas['PnL_fwd_pts_50'])
pnl_acum_activa = pd.Series(pnl_w25).cumsum()

plt.plot(pnl_acum_pasiva.values, label='Estrategia Pasiva', linewidth=2, color='orange', alpha=0.8)
plt.plot(pnl_acum_activa.values, label='Estrategia Activa', linewidth=2, color='blue', alpha=0.8)
plt.fill_between(range(len(pnl_acum_activa)), pnl_acum_pasiva.values, pnl_acum_activa.values,
                 where=(pnl_acum_activa.values >= pnl_acum_pasiva.values), alpha=0.3, color='green', label='Ventaja Activa')
plt.fill_between(range(len(pnl_acum_activa)), pnl_acum_pasiva.values, pnl_acum_activa.values,
                 where=(pnl_acum_activa.values < pnl_acum_pasiva.values), alpha=0.3, color='red', label='Ventaja Pasiva')
plt.xlabel('Número de Operación', fontsize=11)
plt.ylabel('PnL Acumulado (pts)', fontsize=11)
plt.title('PnL Acumulado: Activa vs Pasiva', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 5: Matriz de confusión de decisiones
plt.subplot(2, 3, 5)
if len(cerradas_mejor) > 0:
    # Clasificar decisiones
    cerradas_mejor['decision'] = cerradas_mejor['impacto_cierre'].apply(
        lambda x: 'Buena decisión\n(evitó pérdida)' if x > 10 else
                 ('Decisión neutra' if abs(x) <= 10 else 'Mala decisión\n(perdió ganancia)')
    )

    decision_counts = cerradas_mejor['decision'].value_counts()
    colors_pie = ['green', 'yellow', 'red']
    plt.pie(decision_counts.values, labels=decision_counts.index, autopct='%1.1f%%',
            colors=colors_pie[:len(decision_counts)], startangle=90)
    plt.title(f'Calidad de Decisiones de Cierre\n({len(cerradas_mejor)} ops cerradas)',
              fontsize=12, fontweight='bold')

# Subplot 6: Heatmap de decisión
plt.subplot(2, 3, 6)
# Crear matriz: deterioro vs resultado
deterioro_bins = pd.cut(df_validas['delta_pnldv_25'], bins=10)
resultado_bins = pd.cut(df_validas['PnL_fwd_pts_50'] - df_validas['PnL_fwd_pts_25'], bins=10)
pivot = pd.crosstab(deterioro_bins, resultado_bins)
sns.heatmap(pivot, cmap='RdYlGn', center=0, annot=False, cbar_kws={'label': 'Frecuencia'})
plt.xlabel('Resultado de esperar (PnL_W50 - PnL_W25)', fontsize=10)
plt.ylabel('Deterioro PnLDV en W=25', fontsize=10)
plt.title('Heatmap: Deterioro vs Resultado de Esperar', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)

plt.tight_layout()
plt.savefig('analisis_detallado_mejor_umbral.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 2 guardado: analisis_detallado_mejor_umbral.png")

# Figura 3: Análisis por categoría de deterioro
fig3 = plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
categorias = analisis_categoria.index.tolist()
x_pos = np.arange(len(categorias))
width = 0.35

plt.bar(x_pos - width/2, analisis_categoria['PnL_W25_mean'], width,
        label='PnL en W=25', color='blue', alpha=0.7)
plt.bar(x_pos + width/2, analisis_categoria['PnL_W50_mean'], width,
        label='PnL en W=50', color='orange', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Categoría de Deterioro', fontsize=11)
plt.ylabel('PnL Promedio (pts)', fontsize=11)
plt.title('PnL Promedio por Categoría de Deterioro', fontsize=13, fontweight='bold')
plt.xticks(x_pos, categorias, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(2, 2, 2)
diferencias = analisis_categoria['Diferencia_mean'].values
colors_diff = ['green' if d < -10 else ('yellow' if abs(d) < 10 else 'red') for d in diferencias]
bars = plt.bar(x_pos, diferencias, color=colors_diff, alpha=0.7)
plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
plt.axhline(y=-10, color='green', linestyle=':', linewidth=1, label='Umbral: cerrar recomendado')
plt.axhline(y=10, color='red', linestyle=':', linewidth=1, label='Umbral: dejar correr recomendado')
plt.xlabel('Categoría de Deterioro', fontsize=11)
plt.ylabel('Diferencia (PnL_W50 - PnL_W25)', fontsize=11)
plt.title('Impacto de Esperar hasta W=50 por Categoría', fontsize=13, fontweight='bold')
plt.xticks(x_pos, categorias, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(2, 2, 3)
n_ops = analisis_categoria['N_ops'].values
plt.bar(x_pos, n_ops, color='purple', alpha=0.7)
plt.xlabel('Categoría de Deterioro', fontsize=11)
plt.ylabel('Número de Operaciones', fontsize=11)
plt.title('Frecuencia por Categoría de Deterioro', fontsize=13, fontweight='bold')
plt.xticks(x_pos, categorias, rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

# Añadir tabla de recomendaciones
plt.subplot(2, 2, 4)
plt.axis('off')
tabla_data = []
for idx, row in analisis_categoria.iterrows():
    tabla_data.append([
        str(idx)[:20],
        f"{row['N_ops']:.0f}",
        f"{row['Diferencia_mean']:+.1f}",
        row['Recomendacion']
    ])

tabla = plt.table(cellText=tabla_data,
                 colLabels=['Categoría', 'N ops', 'Dif.', 'Recomendación'],
                 cellLoc='left', loc='center',
                 colWidths=[0.4, 0.15, 0.15, 0.3])
tabla.auto_set_font_size(False)
tabla.set_fontsize(9)
tabla.scale(1, 2)
plt.title('Matriz de Decisión por Categoría', fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('analisis_por_categoria_deterioro.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 3 guardado: analisis_por_categoria_deterioro.png")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*100)
print("RESUMEN EJECUTIVO Y CONCLUSIONES")
print("="*100)

print(f"\n🎯 PREGUNTA INICIAL:")
print(f"   ¿Hubiera mejorado el PnL general cerrar en W=25 cuando el PnLDV sufre deterioro?")

print(f"\n✅ RESPUESTA:")
if mejor_umbral['mejora_absoluta'] > 0:
    print(f"   SÍ. La estrategia de cierre anticipado hubiera MEJORADO el rendimiento.")
else:
    print(f"   NO. La estrategia de cierre anticipado hubiera EMPEORADO el rendimiento.")

print(f"\n📊 MEJOR ESTRATEGIA IDENTIFICADA:")
print(f"   {mejor_umbral['umbral']}")

print(f"\n💰 IMPACTO FINANCIERO:")
print(f"   PnL total adicional ganado:  {mejor_umbral['mejora_absoluta']:+.2f} pts ({mejor_umbral['mejora_porcentual']:+.2f}%)")
print(f"   PnL promedio por operación:  {mejor_umbral['mejora_media']:+.2f} pts")
print(f"   Total de operaciones:        {len(df_validas)}")

print(f"\n📈 MÉTRICAS DE RENDIMIENTO:")
print(f"   Win rate mejorado en:        {mejor_umbral['mejora_wr']:+.2f}%")
print(f"   Sharpe ratio mejorado en:    {mejor_umbral['sharpe_activa'] - mejor_umbral['sharpe_pasiva']:+.3f}")
print(f"   Riesgo reducido en:          {mejor_umbral['reduccion_riesgo']:+.2f}%")

print(f"\n⚙️ IMPLEMENTACIÓN:")
print(f"   Operaciones a cerrar en W=25: {mejor_umbral['n_cerradas']:.0f} ({mejor_umbral['pct_cerradas']:.1f}% del total)")
print(f"   Operaciones que continúan:    {len(df_validas) - mejor_umbral['n_cerradas']:.0f} ({100-mejor_umbral['pct_cerradas']:.1f}%)")

if len(cerradas_mejor) > 0:
    print(f"\n💡 EFECTIVIDAD DEL CIERRE:")
    print(f"   Operaciones que evitaron empeorar:  {(cerradas_mejor['diferencia_w50_w25'] < 0).sum()} ({(cerradas_mejor['diferencia_w50_w25'] < 0).sum()/len(cerradas_mejor)*100:.1f}%)")
    print(f"   Pérdida promedio evitada:           {-cerradas_mejor['diferencia_w50_w25'].mean():+.2f} pts")

print("\n" + "="*100)
print("Archivos generados:")
print("  1. comparacion_estrategias_umbrales.png")
print("  2. analisis_detallado_mejor_umbral.png")
print("  3. analisis_por_categoria_deterioro.png")
print("="*100)
