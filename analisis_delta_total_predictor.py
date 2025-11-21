"""
ANÁLISIS DE DELTA_TOTAL COMO PREDICTOR DE PnL FWD Y PnLDV FWD
============================================================

Pregunta: ¿Tiene delta_total (especialmente cuando cae de valor)
          correlación con PnL FWD y PnLDV FWD?
          ¿Es un buen predictor del comportamiento futuro?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (22, 14)

print("="*100)
print("ANÁLISIS DE DELTA_TOTAL COMO PREDICTOR")
print("="*100)
print(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Cargar datos
df = pd.read_csv('PNLDV.csv')
print(f"Dataset: {len(df)} operaciones\n")

# Filtrar operaciones válidas
df_analisis = df[
    df['delta_total'].notna() &
    df['PnL_fwd_pts_50'].notna()
].copy()

print(f"Operaciones con delta_total y PnL FWD 50 válidos: {len(df_analisis)}")
print(f"Excluidas: {len(df) - len(df_analisis)} operaciones\n")

# ============================================================================
# PASO 1: ANÁLISIS DESCRIPTIVO DE DELTA_TOTAL
# ============================================================================

print("="*100)
print("PASO 1: ANÁLISIS DESCRIPTIVO DE DELTA_TOTAL")
print("="*100)

delta_stats = df_analisis['delta_total'].describe()
print("\nEstadísticas de delta_total:")
print(delta_stats)

print("\nPercentiles adicionales:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    val = df_analisis['delta_total'].quantile(p/100)
    print(f"  P{p:>2}: {val:>8.5f}")

# Categorizar delta por su valor absoluto y signo
df_analisis['delta_abs'] = df_analisis['delta_total'].abs()
df_analisis['delta_signo'] = np.where(
    df_analisis['delta_total'] > 0.01, 'Positivo (alcista)',
    np.where(df_analisis['delta_total'] < -0.01, 'Negativo (bajista)', 'Neutral')
)

# Categorizar por magnitud
df_analisis['delta_magnitud'] = pd.cut(
    df_analisis['delta_abs'],
    bins=[0, 0.01, 0.02, 0.03, 0.05, 1.0],
    labels=['Muy Bajo (<0.01)', 'Bajo (0.01-0.02)', 'Medio (0.02-0.03)',
            'Alto (0.03-0.05)', 'Muy Alto (>0.05)']
)

print("\nDistribución por signo de delta:")
print(df_analisis['delta_signo'].value_counts())

print("\nDistribución por magnitud de delta (absoluto):")
print(df_analisis['delta_magnitud'].value_counts().sort_index())

# ============================================================================
# PASO 2: CORRELACIÓN DELTA_TOTAL CON PnL FWD
# ============================================================================

print("\n" + "="*100)
print("PASO 2: CORRELACIÓN DELTA_TOTAL CON PnL FWD")
print("="*100)

ventanas = ['01', '05', '25', '50']
correlaciones_pnl = {}

print("\nCorrelaciones de Pearson: delta_total vs PnL FWD:")
print("-"*100)
print(f"{'Ventana':<10} {'Correlación':<15} {'P-value':<15} {'N válidos':<15} {'Interpretación'}")
print("-"*100)

for v in ventanas:
    pnl_col = f'PnL_fwd_pts_{v}'

    if pnl_col in df_analisis.columns:
        mask = df_analisis[pnl_col].notna() & df_analisis['delta_total'].notna()
        data = df_analisis[mask]

        if len(data) > 10:
            corr, pval = stats.pearsonr(data['delta_total'], data[pnl_col])

            # Interpretación
            if abs(corr) > 0.3:
                interp = "MODERADA-FUERTE"
            elif abs(corr) > 0.1:
                interp = "DÉBIL"
            else:
                interp = "MUY DÉBIL"

            signo = "POSITIVA" if corr > 0 else "NEGATIVA"

            correlaciones_pnl[v] = {
                'corr': corr,
                'pval': pval,
                'n': len(data)
            }

            print(f"FWD_{v:<6} {corr:>8.4f}      {pval:>12.6f}    {len(data):>8}      {interp} {signo}")

# Análisis por quintiles de delta
print("\n" + "="*100)
print("PnL FWD PROMEDIO POR QUINTIL DE DELTA_TOTAL")
print("="*100)

for v in ventanas:
    pnl_col = f'PnL_fwd_pts_{v}'

    if pnl_col in df_analisis.columns:
        print(f"\n--- FWD_{v} ---")

        # Crear quintiles
        mask = df_analisis[pnl_col].notna() & df_analisis['delta_total'].notna()
        data = df_analisis[mask].copy()

        try:
            data['delta_quintil'] = pd.qcut(data['delta_total'], q=5, labels=False, duplicates='drop')

            analisis = data.groupby('delta_quintil').agg({
                'delta_total': ['min', 'max', 'mean', 'count'],
                pnl_col: ['mean', 'median', 'std']
            }).round(4)

            print("\nPnL por quintil de delta_total:")
            print(analisis)

            # Tasa de deterioro por quintil
            data['deterioro'] = (data[pnl_col] < -100).astype(int)
            tasa_deterioro = data.groupby('delta_quintil')['deterioro'].mean() * 100

            print("\nTasa de deterioro grave (<-100) por quintil:")
            for q in range(5):
                if q in tasa_deterioro.index:
                    print(f"  Q{q+1}: {tasa_deterioro[q]:.1f}%")
        except Exception as e:
            print(f"Error al crear quintiles: {e}")

# ============================================================================
# PASO 3: CORRELACIÓN DELTA_TOTAL CON PnLDV FWD
# ============================================================================

print("\n" + "="*100)
print("PASO 3: CORRELACIÓN DELTA_TOTAL CON PnLDV FWD")
print("="*100)

correlaciones_pnldv = {}

print("\nCorrelaciones de Pearson: delta_total vs PnLDV FWD:")
print("-"*100)
print(f"{'Variable':<20} {'Correlación':<15} {'P-value':<15} {'N válidos':<15} {'Interpretación'}")
print("-"*100)

# PnLDV inicial
if 'PnLDV' in df_analisis.columns:
    mask = df_analisis['PnLDV'].notna() & df_analisis['delta_total'].notna()
    data = df_analisis[mask]

    if len(data) > 10:
        corr, pval = stats.pearsonr(data['delta_total'], data['PnLDV'])

        if abs(corr) > 0.3:
            interp = "MODERADA-FUERTE"
        elif abs(corr) > 0.1:
            interp = "DÉBIL"
        else:
            interp = "MUY DÉBIL"

        signo = "POSITIVA" if corr > 0 else "NEGATIVA"

        correlaciones_pnldv['T0'] = {'corr': corr, 'pval': pval, 'n': len(data)}

        print(f"PnLDV (T+0)          {corr:>8.4f}      {pval:>12.6f}    {len(data):>8}      {interp} {signo}")

# PnLDV FWD
for v in ventanas:
    pnldv_col = f'PnLDV_fwd_{v}'

    if pnldv_col in df_analisis.columns:
        mask = df_analisis[pnldv_col].notna() & df_analisis['delta_total'].notna()
        data = df_analisis[mask]

        if len(data) > 10:
            corr, pval = stats.pearsonr(data['delta_total'], data[pnldv_col])

            if abs(corr) > 0.3:
                interp = "MODERADA-FUERTE"
            elif abs(corr) > 0.1:
                interp = "DÉBIL"
            else:
                interp = "MUY DÉBIL"

            signo = "POSITIVA" if corr > 0 else "NEGATIVA"

            correlaciones_pnldv[v] = {'corr': corr, 'pval': pval, 'n': len(data)}

            print(f"PnLDV_fwd_{v:<10} {corr:>8.4f}      {pval:>12.6f}    {len(data):>8}      {interp} {signo}")

# ============================================================================
# PASO 4: ANÁLISIS DE "CAÍDAS" DE DELTA
# ============================================================================

print("\n" + "="*100)
print("PASO 4: ANÁLISIS DE DELTAS EXTREMOS")
print("="*100)

# Definir deltas "extremos" o "caídos"
# Delta muy negativo = muy bajista (caído)
# Delta muy positivo = muy alcista

percentil_10 = df_analisis['delta_total'].quantile(0.10)
percentil_90 = df_analisis['delta_total'].quantile(0.90)

df_analisis['delta_categoria'] = pd.cut(
    df_analisis['delta_total'],
    bins=[-np.inf, percentil_10, percentil_90, np.inf],
    labels=['Muy Negativo (P0-P10)', 'Neutral (P10-P90)', 'Muy Positivo (P90-P100)']
)

print(f"\nUmbrales de delta:")
print(f"  Muy Negativo: delta_total ≤ {percentil_10:.5f}")
print(f"  Neutral: {percentil_10:.5f} < delta_total < {percentil_90:.5f}")
print(f"  Muy Positivo: delta_total ≥ {percentil_90:.5f}")

print("\nDistribución:")
print(df_analisis['delta_categoria'].value_counts())

# Análisis por categoría
print("\n" + "="*100)
print("PnL FWD 50 PROMEDIO POR CATEGORÍA DE DELTA")
print("="*100)

analisis_categoria = df_analisis.groupby('delta_categoria').agg({
    'PnL_fwd_pts_50': ['count', 'mean', 'median', 'std'],
    'delta_total': ['mean', 'min', 'max']
}).round(2)

print("\n", analisis_categoria)

# Tasa de deterioro por categoría
print("\nTasa de deterioro grave (<-100) por categoría de delta:")
df_analisis['deterioro_grave'] = (df_analisis['PnL_fwd_pts_50'] < -100).astype(int)

tasa_deterioro_cat = df_analisis.groupby('delta_categoria')['deterioro_grave'].agg(['sum', 'count', 'mean'])
tasa_deterioro_cat['tasa_pct'] = tasa_deterioro_cat['mean'] * 100

print(tasa_deterioro_cat)

# Win rate por categoría
print("\nWin Rate (PnL > 0) por categoría de delta:")
df_analisis['win'] = (df_analisis['PnL_fwd_pts_50'] > 0).astype(int)

win_rate_cat = df_analisis.groupby('delta_categoria')['win'].agg(['sum', 'count', 'mean'])
win_rate_cat['wr_pct'] = win_rate_cat['mean'] * 100

print(win_rate_cat)

# ============================================================================
# PASO 5: ANÁLISIS DE DELTA NEUTRAL vs DIRECCIONAL
# ============================================================================

print("\n" + "="*100)
print("PASO 5: DELTA NEUTRAL vs DIRECCIONAL")
print("="*100)

# Definir delta "neutral" como muy cercano a 0
umbral_neutral = 0.02  # |delta| < 0.02 se considera neutral

df_analisis['es_neutral'] = df_analisis['delta_abs'] < umbral_neutral
df_analisis['es_direccional'] = df_analisis['delta_abs'] >= umbral_neutral

print(f"\nOperaciones con delta neutral (|delta| < {umbral_neutral}):")
n_neutral = df_analisis['es_neutral'].sum()
print(f"  {n_neutral} ops ({n_neutral/len(df_analisis)*100:.1f}%)")

print(f"\nOperaciones con delta direccional (|delta| ≥ {umbral_neutral}):")
n_direccional = df_analisis['es_direccional'].sum()
print(f"  {n_direccional} ops ({n_direccional/len(df_analisis)*100:.1f}%)")

# Comparar rendimiento
print("\nComparación de rendimiento:")
print("-"*100)

comparacion = df_analisis.groupby('es_neutral').agg({
    'PnL_fwd_pts_50': ['mean', 'median', 'std'],
    'deterioro_grave': 'mean',
    'win': 'mean'
}).round(2)

comparacion.index = ['Direccional', 'Neutral']
print(comparacion)

# Test estadístico
neutral_pnl = df_analisis[df_analisis['es_neutral']]['PnL_fwd_pts_50'].dropna()
direccional_pnl = df_analisis[df_analisis['es_direccional']]['PnL_fwd_pts_50'].dropna()

if len(neutral_pnl) > 0 and len(direccional_pnl) > 0:
    t_stat, t_pval = stats.ttest_ind(neutral_pnl, direccional_pnl)
    print(f"\nTest t de diferencia de medias:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {t_pval:.6f}")

    if t_pval < 0.05:
        print(f"  → Diferencia ESTADÍSTICAMENTE SIGNIFICATIVA")
    else:
        print(f"  → Diferencia NO significativa")

# ============================================================================
# PASO 6: DELTA COMO PREDICTOR DE DETERIORO
# ============================================================================

print("\n" + "="*100)
print("PASO 6: DELTA COMO PREDICTOR DE DETERIORO")
print("="*100)

from sklearn.metrics import roc_auc_score

# Calcular AUC-ROC de delta_total como predictor
try:
    auc = roc_auc_score(df_analisis['deterioro_grave'], df_analisis['delta_abs'])
    auc = max(auc, 1 - auc)  # Tomar el correcto según dirección

    print(f"\nAUC-ROC de |delta_total| como predictor de deterioro grave:")
    print(f"  AUC = {auc:.3f}")

    if auc > 0.7:
        print(f"  → Predictor BUENO")
    elif auc > 0.6:
        print(f"  → Predictor ACEPTABLE")
    else:
        print(f"  → Predictor POBRE")
except Exception as e:
    print(f"Error calculando AUC: {e}")

# Comparar con otros predictores conocidos
print("\nComparación con otros predictores (AUC-ROC):")
print("-"*100)

predictores_comparar = {
    'delta_abs': df_analisis['delta_abs'],
    'delta_total': df_analisis['delta_total'],
}

# Agregar PnLDV si existe
if 'PnLDV' in df_analisis.columns:
    predictores_comparar['PnLDV'] = df_analisis['PnLDV']

# Agregar IV si existe
for iv_col in ['iv_k3', 'iv_k2', 'iv_k1']:
    if iv_col in df_analisis.columns:
        predictores_comparar[iv_col] = df_analisis[iv_col]

for nombre, serie in predictores_comparar.items():
    try:
        mask = serie.notna() & df_analisis['deterioro_grave'].notna()
        if mask.sum() > 10:
            auc = roc_auc_score(
                df_analisis[mask]['deterioro_grave'],
                serie[mask]
            )
            auc = max(auc, 1 - auc)
            print(f"  {nombre:<20} AUC = {auc:.3f}")
    except:
        pass

# ============================================================================
# PASO 7: ANÁLISIS POR SIGNO DE MOVIMIENTO DEL MERCADO
# ============================================================================

print("\n" + "="*100)
print("PASO 7: DELTA vs MOVIMIENTO DEL MERCADO")
print("="*100)

# Calcular cambio de SPX
if 'SPX_chg_pct_50' in df_analisis.columns:
    print("\nRelación entre delta_total y cambio del SPX:")

    df_analisis['spx_movimiento'] = pd.cut(
        df_analisis['SPX_chg_pct_50'],
        bins=[-np.inf, -5, 5, np.inf],
        labels=['Bajista (<-5%)', 'Lateral (-5% a +5%)', 'Alcista (>+5%)']
    )

    print("\nPnL FWD 50 promedio por combinación delta/movimiento mercado:")
    pivot = df_analisis.pivot_table(
        values='PnL_fwd_pts_50',
        index='delta_categoria',
        columns='spx_movimiento',
        aggfunc='mean'
    ).round(2)

    print(pivot)

# ============================================================================
# VISUALIZACIONES
# ============================================================================

print("\n" + "="*100)
print("GENERANDO VISUALIZACIONES...")
print("="*100)

fig = plt.figure(figsize=(22, 16))

# Subplot 1: Distribución de delta_total
plt.subplot(3, 3, 1)
plt.hist(df_analisis['delta_total'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Delta = 0 (neutral)')
plt.axvline(x=percentil_10, color='orange', linestyle=':', linewidth=2, label='P10')
plt.axvline(x=percentil_90, color='orange', linestyle=':', linewidth=2, label='P90')
plt.xlabel('delta_total', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.title('Distribución de delta_total', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Subplot 2: Scatter delta vs PnL FWD 50
plt.subplot(3, 3, 2)
plt.hexbin(df_analisis['delta_total'], df_analisis['PnL_fwd_pts_50'],
           gridsize=30, cmap='YlOrRd', alpha=0.7)
plt.colorbar(label='Densidad')
z = np.polyfit(df_analisis['delta_total'].dropna(),
               df_analisis['PnL_fwd_pts_50'].dropna(), 1)
p = np.poly1d(z)
x_line = np.linspace(df_analisis['delta_total'].min(), df_analisis['delta_total'].max(), 100)
plt.plot(x_line, p(x_line), "b--", linewidth=2, label='Regresión')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
plt.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
plt.xlabel('delta_total', fontsize=12)
plt.ylabel('PnL FWD 50 (pts)', fontsize=12)
plt.title(f'delta_total vs PnL FWD 50\n(Corr={correlaciones_pnl.get("50", {}).get("corr", 0):.3f})',
          fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 3: Box plot PnL por categoría de delta
plt.subplot(3, 3, 3)
data_box = []
labels_box = []
for cat in ['Muy Negativo (P0-P10)', 'Neutral (P10-P90)', 'Muy Positivo (P90-P100)']:
    data_cat = df_analisis[df_analisis['delta_categoria'] == cat]['PnL_fwd_pts_50'].dropna()
    if len(data_cat) > 0:
        data_box.append(data_cat)
        labels_box.append(cat.split(' (')[0])

bp = plt.boxplot(data_box, labels=labels_box, patch_artist=True)
colors = ['red', 'yellow', 'green']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
plt.ylabel('PnL FWD 50 (pts)', fontsize=12)
plt.title('PnL FWD 50 por Categoría de Delta', fontsize=13, fontweight='bold')
plt.xticks(rotation=15)
plt.grid(True, alpha=0.3, axis='y')

# Subplot 4: Correlación por ventana
plt.subplot(3, 3, 4)
ventanas_plot = []
corr_pnl = []
corr_pnldv = []

for v in ventanas:
    if v in correlaciones_pnl:
        ventanas_plot.append(f'FWD_{v}')
        corr_pnl.append(correlaciones_pnl[v]['corr'])
        corr_pnldv.append(correlaciones_pnldv.get(v, {}).get('corr', 0))

x = np.arange(len(ventanas_plot))
width = 0.35

bars1 = plt.bar(x - width/2, corr_pnl, width, label='vs PnL FWD', color='blue', alpha=0.7)
bars2 = plt.bar(x + width/2, corr_pnldv, width, label='vs PnLDV FWD', color='red', alpha=0.7)

plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
plt.xlabel('Ventana', fontsize=12)
plt.ylabel('Correlación', fontsize=12)
plt.title('Correlación delta_total con PnL y PnLDV', fontsize=13, fontweight='bold')
plt.xticks(x, ventanas_plot)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Subplot 5: Tasa de deterioro por categoría
plt.subplot(3, 3, 5)
categorias = tasa_deterioro_cat.index.tolist()
tasas = tasa_deterioro_cat['tasa_pct'].values

colors_bar = ['red', 'yellow', 'green']
plt.bar(range(len(categorias)), tasas, color=colors_bar, alpha=0.7, edgecolor='black')
plt.axhline(y=df_analisis['deterioro_grave'].mean()*100,
            color='black', linestyle='--', linewidth=2, label='Tasa base')
plt.xlabel('Categoría de Delta', fontsize=12)
plt.ylabel('Tasa de Deterioro Grave (%)', fontsize=12)
plt.title('Tasa de Deterioro por Categoría de Delta', fontsize=13, fontweight='bold')
plt.xticks(range(len(categorias)), [c.split(' (')[0] for c in categorias], rotation=15)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Subplot 6: Win rate por categoría
plt.subplot(3, 3, 6)
wr = win_rate_cat['wr_pct'].values

plt.bar(range(len(categorias)), wr, color=colors_bar, alpha=0.7, edgecolor='black')
plt.axhline(y=df_analisis['win'].mean()*100,
            color='black', linestyle='--', linewidth=2, label='Win rate base')
plt.xlabel('Categoría de Delta', fontsize=12)
plt.ylabel('Win Rate (%)', fontsize=12)
plt.title('Win Rate por Categoría de Delta', fontsize=13, fontweight='bold')
plt.xticks(range(len(categorias)), [c.split(' (')[0] for c in categorias], rotation=15)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Subplot 7: Delta absoluto vs PnL
plt.subplot(3, 3, 7)
plt.hexbin(df_analisis['delta_abs'], df_analisis['PnL_fwd_pts_50'],
           gridsize=30, cmap='RdYlGn', alpha=0.7)
plt.colorbar(label='Densidad')
plt.xlabel('|delta_total| (magnitud)', fontsize=12)
plt.ylabel('PnL FWD 50 (pts)', fontsize=12)
plt.title('Magnitud de Delta vs PnL FWD 50', fontsize=13, fontweight='bold')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.grid(True, alpha=0.3)

# Subplot 8: Comparación neutral vs direccional
plt.subplot(3, 3, 8)
tipos = ['Neutral\n|δ|<0.02', 'Direccional\n|δ|≥0.02']
pnl_neutral = df_analisis[df_analisis['es_neutral']]['PnL_fwd_pts_50'].mean()
pnl_direccional = df_analisis[df_analisis['es_direccional']]['PnL_fwd_pts_50'].mean()

bars = plt.bar(tipos, [pnl_neutral, pnl_direccional],
               color=['gray', 'blue'], alpha=0.7, edgecolor='black')
plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
plt.ylabel('PnL FWD 50 Promedio (pts)', fontsize=12)
plt.title('Neutral vs Direccional', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Añadir valores
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=11)

# Subplot 9: PnL por quintil de delta
plt.subplot(3, 3, 9)
try:
    data_q = df_analisis.copy()
    data_q['delta_quintil'] = pd.qcut(data_q['delta_total'], q=5, labels=False, duplicates='drop')
    pnl_por_quintil = data_q.groupby('delta_quintil')['PnL_fwd_pts_50'].mean()

    colors_q = ['darkred', 'orange', 'yellow', 'lightgreen', 'darkgreen']
    plt.bar(range(len(pnl_por_quintil)), pnl_por_quintil.values,
            color=colors_q[:len(pnl_por_quintil)], alpha=0.7, edgecolor='black')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
    plt.xlabel('Quintil de delta_total', fontsize=12)
    plt.ylabel('PnL FWD 50 Promedio (pts)', fontsize=12)
    plt.title('PnL Promedio por Quintil de Delta', fontsize=13, fontweight='bold')
    plt.xticks(range(len(pnl_por_quintil)), [f'Q{i+1}\n(más neg)' if i==0 else f'Q{i+1}' if i<4 else f'Q{i+1}\n(más pos)'
                                              for i in range(len(pnl_por_quintil))])
    plt.grid(True, alpha=0.3, axis='y')
except Exception as e:
    print(f"Error en subplot 9: {e}")

plt.tight_layout()
plt.savefig('analisis_delta_total_predictor.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: analisis_delta_total_predictor.png")

# ============================================================================
# RESUMEN EJECUTIVO
# ============================================================================

print("\n\n" + "="*100)
print("RESUMEN EJECUTIVO Y CONCLUSIONES")
print("="*100)

print("\n🎯 PREGUNTA: ¿Tiene delta_total relación como predictor de PnL y PnLDV FWD?")
print("="*100)

print("\n1. CORRELACIÓN CON PnL FWD:")
for v in ventanas:
    if v in correlaciones_pnl:
        corr = correlaciones_pnl[v]['corr']
        print(f"   FWD_{v}: r = {corr:+.3f} {'(SIGNIFICATIVA)' if abs(corr) > 0.1 else '(débil)'}")

print("\n2. CORRELACIÓN CON PnLDV FWD:")
for v in ventanas:
    if v in correlaciones_pnldv:
        corr = correlaciones_pnldv[v]['corr']
        print(f"   FWD_{v}: r = {corr:+.3f} {'(SIGNIFICATIVA)' if abs(corr) > 0.1 else '(débil)'}")

print(f"\n3. PODER PREDICTIVO:")
try:
    auc_delta = roc_auc_score(df_analisis['deterioro_grave'], df_analisis['delta_abs'])
    auc_delta = max(auc_delta, 1 - auc_delta)
    print(f"   AUC-ROC como predictor de deterioro: {auc_delta:.3f}")

    if auc_delta < 0.55:
        conclusion = "POBRE - No es un buen predictor"
    elif auc_delta < 0.65:
        conclusion = "DÉBIL - Poder predictivo limitado"
    elif auc_delta < 0.75:
        conclusion = "ACEPTABLE - Cierto poder predictivo"
    else:
        conclusion = "BUENO - Buen poder predictivo"

    print(f"   Conclusión: {conclusion}")
except:
    pass

print(f"\n4. RENDIMIENTO POR CATEGORÍA DE DELTA:")
print(f"   Delta muy negativo (P0-P10): {df_analisis[df_analisis['delta_categoria']=='Muy Negativo (P0-P10)']['PnL_fwd_pts_50'].mean():.2f} pts")
print(f"   Delta neutral (P10-P90):     {df_analisis[df_analisis['delta_categoria']=='Neutral (P10-P90)']['PnL_fwd_pts_50'].mean():.2f} pts")
print(f"   Delta muy positivo (P90-P100): {df_analisis[df_analisis['delta_categoria']=='Muy Positivo (P90-P100)']['PnL_fwd_pts_50'].mean():.2f} pts")

print("\n" + "="*100)
print("ANÁLISIS COMPLETADO")
print("="*100)
