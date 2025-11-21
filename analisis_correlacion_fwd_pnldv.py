"""
Análisis Exhaustivo de Correlación entre FWD PTS y FWD PNLDV
============================================================

Este script analiza:
1. Correlaciones entre PnL FWD PTS y PnLDV FWD para todas las ventanas
2. Cómo afecta al PNL FWD la subida o bajada del PNLDV FWD
3. Si una caída de PNLDV FWD está relacionada con los PNL FWD
4. Si mantener estable el PNLDV FWD inicial respecto al PNLDV T+0 está ligado con mejor performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

# Cargar datos
print("="*80)
print("ANÁLISIS EXHAUSTIVO DE CORRELACIÓN: FWD PTS vs FWD PNLDV")
print("="*80)
print(f"\nFecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

df = pd.read_csv('PNLDV.csv')
print(f"Dataset cargado: {len(df)} operaciones\n")

# ============================================================================
# 1. ANÁLISIS DE CORRELACIONES GENERALES
# ============================================================================
print("\n" + "="*80)
print("1. CORRELACIONES ENTRE PNL FWD PTS Y PNLDV FWD")
print("="*80)

ventanas = ['01', '05', '25', '50']
correlaciones = {}

print("\nCorrelaciones de Pearson por ventana temporal:")
print("-" * 80)
print(f"{'Ventana':<10} {'Correlación':<15} {'P-value':<15} {'N válidos':<15} {'Interpretación'}")
print("-" * 80)

for v in ventanas:
    pnl_col = f'PnL_fwd_pts_{v}'
    pnldv_col = f'PnLDV_fwd_{v}'

    # Filtrar datos válidos
    mask = df[pnl_col].notna() & df[pnldv_col].notna()
    pnl_data = df.loc[mask, pnl_col]
    pnldv_data = df.loc[mask, pnldv_col]

    if len(pnl_data) > 0:
        corr, pval = stats.pearsonr(pnl_data, pnldv_data)
        correlaciones[v] = {'corr': corr, 'pval': pval, 'n': len(pnl_data)}

        # Interpretación
        if abs(corr) > 0.7:
            interp = "FUERTE"
        elif abs(corr) > 0.4:
            interp = "MODERADA"
        elif abs(corr) > 0.2:
            interp = "DÉBIL"
        else:
            interp = "MUY DÉBIL"

        signo = "POSITIVA" if corr > 0 else "NEGATIVA"

        print(f"FWD_{v:<6} {corr:>8.4f}      {pval:>12.6f}    {len(pnl_data):>8}      {interp} {signo}")

# ============================================================================
# 2. ANÁLISIS DE CAMBIOS EN PNLDV FWD Y SU EFECTO EN PNL FWD
# ============================================================================
print("\n" + "="*80)
print("2. IMPACTO DE CAMBIOS EN PNLDV FWD SOBRE PNL FWD")
print("="*80)

for v in ventanas:
    pnl_col = f'PnL_fwd_pts_{v}'
    pnldv_col = f'PnLDV_fwd_{v}'

    print(f"\n--- Ventana FWD_{v} ---")

    # Calcular cambio del PNLDV FWD respecto al PNLDV T+0
    df[f'delta_pnldv_{v}'] = df[pnldv_col] - df['PnLDV']
    df[f'pct_change_pnldv_{v}'] = ((df[pnldv_col] - df['PnLDV']) / df['PnLDV'].abs()) * 100

    # Clasificar operaciones según cambio de PNLDV
    df[f'categoria_pnldv_{v}'] = pd.cut(
        df[f'delta_pnldv_{v}'],
        bins=[-np.inf, -100, -50, 50, 100, np.inf],
        labels=['Caída Fuerte (< -100)', 'Caída Moderada (-100 a -50)',
                'Estable (-50 a +50)', 'Subida Moderada (+50 a +100)',
                'Subida Fuerte (> +100)']
    )

    # Análisis por categoría
    resumen = df.groupby(f'categoria_pnldv_{v}').agg({
        pnl_col: ['mean', 'median', 'std', 'count'],
        pnldv_col: ['mean', 'median'],
        f'delta_pnldv_{v}': ['mean', 'median']
    }).round(2)

    print("\nPNL FWD PTS según cambio en PNLDV FWD:")
    print(resumen)

    # Tasa de ganancia por categoría
    print(f"\nTasa de operaciones ganadoras (PnL > 0) por categoría:")
    win_rates = df.groupby(f'categoria_pnldv_{v}').apply(
        lambda x: (x[pnl_col] > 0).sum() / len(x) * 100 if len(x) > 0 else 0
    ).round(2)
    print(win_rates)

# ============================================================================
# 3. ANÁLISIS ESPECÍFICO: CAÍDAS DE PNLDV FWD
# ============================================================================
print("\n" + "="*80)
print("3. ANÁLISIS ESPECÍFICO: CAÍDAS DE PNLDV FWD")
print("="*80)

print("\n¿Están las caídas de PNLDV FWD relacionadas con peores PNL FWD?")
print("-" * 80)

caidas_analysis = []

for v in ventanas:
    pnl_col = f'PnL_fwd_pts_{v}'
    pnldv_col = f'PnLDV_fwd_{v}'
    delta_col = f'delta_pnldv_{v}'

    # Operaciones con caída de PNLDV FWD (delta negativo)
    caidas = df[df[delta_col] < 0].copy()
    no_caidas = df[df[delta_col] >= 0].copy()

    if len(caidas) > 0 and len(no_caidas) > 0:
        pnl_caidas = caidas[pnl_col].mean()
        pnl_no_caidas = no_caidas[pnl_col].mean()

        # Test t para comparar medias
        t_stat, t_pval = stats.ttest_ind(
            caidas[pnl_col].dropna(),
            no_caidas[pnl_col].dropna()
        )

        print(f"\nVentana FWD_{v}:")
        print(f"  Operaciones con caída de PNLDV FWD: {len(caidas)} ({len(caidas)/len(df)*100:.1f}%)")
        print(f"  PNL promedio con caída: {pnl_caidas:.2f} pts")
        print(f"  PNL promedio sin caída: {pnl_no_caidas:.2f} pts")
        print(f"  Diferencia: {pnl_caidas - pnl_no_caidas:.2f} pts")
        print(f"  Test t p-value: {t_pval:.6f} {'***SIGNIFICATIVO' if t_pval < 0.05 else '(no significativo)'}")

        caidas_analysis.append({
            'ventana': v,
            'n_caidas': len(caidas),
            'pnl_caidas': pnl_caidas,
            'pnl_no_caidas': pnl_no_caidas,
            'diferencia': pnl_caidas - pnl_no_caidas,
            'pval': t_pval
        })

# ============================================================================
# 4. ANÁLISIS DE ESTABILIDAD: PNLDV FWD vs PNLDV T+0
# ============================================================================
print("\n" + "="*80)
print("4. ESTABILIDAD DE PNLDV FWD RESPECTO A PNLDV T+0")
print("="*80)

print("\n¿Mantener estable el PNLDV FWD inicial respecto al PNLDV T+0 mejora el performance?")
print("-" * 80)

estabilidad_results = []

for v in ventanas:
    pnl_col = f'PnL_fwd_pts_{v}'
    pnldv_col = f'PnLDV_fwd_{v}'
    delta_col = f'delta_pnldv_{v}'
    pct_col = f'pct_change_pnldv_{v}'

    # Definir "estable" como cambio menor al 10%
    df[f'estable_{v}'] = df[pct_col].abs() < 10

    estables = df[df[f'estable_{v}'] == True].copy()
    inestables = df[df[f'estable_{v}'] == False].copy()

    if len(estables) > 0 and len(inestables) > 0:
        pnl_estables = estables[pnl_col].mean()
        pnl_inestables = inestables[pnl_col].mean()

        win_rate_estables = (estables[pnl_col] > 0).sum() / len(estables) * 100
        win_rate_inestables = (inestables[pnl_col] > 0).sum() / len(inestables) * 100

        # Test de significancia
        t_stat, t_pval = stats.ttest_ind(
            estables[pnl_col].dropna(),
            inestables[pnl_col].dropna()
        )

        print(f"\nVentana FWD_{v}:")
        print(f"  Operaciones ESTABLES (cambio PNLDV < 10%): {len(estables)} ({len(estables)/len(df)*100:.1f}%)")
        print(f"    - PNL promedio: {pnl_estables:.2f} pts")
        print(f"    - Win rate: {win_rate_estables:.1f}%")
        print(f"  Operaciones INESTABLES (cambio PNLDV >= 10%): {len(inestables)} ({len(inestables)/len(df)*100:.1f}%)")
        print(f"    - PNL promedio: {pnl_inestables:.2f} pts")
        print(f"    - Win rate: {win_rate_inestables:.1f}%")
        print(f"  DIFERENCIA en PNL: {pnl_estables - pnl_inestables:.2f} pts")
        print(f"  DIFERENCIA en Win Rate: {win_rate_estables - win_rate_inestables:.1f}%")
        print(f"  Test t p-value: {t_pval:.6f} {'***SIGNIFICATIVO' if t_pval < 0.05 else '(no significativo)'}")

        estabilidad_results.append({
            'ventana': v,
            'n_estables': len(estables),
            'pnl_estables': pnl_estables,
            'wr_estables': win_rate_estables,
            'pnl_inestables': pnl_inestables,
            'wr_inestables': win_rate_inestables,
            'dif_pnl': pnl_estables - pnl_inestables,
            'dif_wr': win_rate_estables - win_rate_inestables,
            'pval': t_pval
        })

# ============================================================================
# 5. ANÁLISIS ADICIONAL: EVOLUCIÓN TEMPORAL DE PNLDV FWD
# ============================================================================
print("\n" + "="*80)
print("5. EVOLUCIÓN TEMPORAL DE PNLDV FWD")
print("="*80)

print("\n¿Cómo evoluciona el PNLDV desde T+0 hasta las distintas ventanas FWD?")
print("-" * 80)

evolucion_cols = ['PnLDV'] + [f'PnLDV_fwd_{v}' for v in ventanas]
evolucion_stats = df[evolucion_cols].describe().round(2)

print("\nEstadísticas descriptivas de PNLDV a través del tiempo:")
print(evolucion_stats)

# Calcular deterioro/mejora promedio
print("\nCambio promedio de PNLDV desde T+0:")
for v in ventanas:
    cambio_medio = df[f'delta_pnldv_{v}'].mean()
    pct_cambio = df[f'pct_change_pnldv_{v}'].mean()
    print(f"  FWD_{v}: {cambio_medio:+.2f} pts ({pct_cambio:+.2f}%)")

# ============================================================================
# 6. ANÁLISIS DE REGRESIÓN
# ============================================================================
print("\n" + "="*80)
print("6. ANÁLISIS DE REGRESIÓN LINEAL")
print("="*80)

print("\nRegresión: PnL FWD ~ PNLDV FWD + Delta PNLDV")
print("-" * 80)

for v in ventanas:
    pnl_col = f'PnL_fwd_pts_{v}'
    pnldv_col = f'PnLDV_fwd_{v}'
    delta_col = f'delta_pnldv_{v}'

    # Preparar datos para regresión
    mask = df[pnl_col].notna() & df[pnldv_col].notna() & df[delta_col].notna()
    X = df.loc[mask, [pnldv_col, delta_col]].values
    y = df.loc[mask, pnl_col].values

    if len(X) > 0:
        from sklearn.linear_model import LinearRegression

        reg = LinearRegression()
        reg.fit(X, y)

        r2 = reg.score(X, y)

        print(f"\nVentana FWD_{v}:")
        print(f"  R² = {r2:.4f}")
        print(f"  Coef. PNLDV FWD: {reg.coef_[0]:.4f}")
        print(f"  Coef. Delta PNLDV: {reg.coef_[1]:.4f}")
        print(f"  Intercepto: {reg.intercept_:.2f}")

        # Interpretación
        if reg.coef_[1] > 0:
            print(f"  → Un aumento en PNLDV FWD está asociado con MEJOR PnL FWD")
        else:
            print(f"  → Un aumento en PNLDV FWD está asociado con PEOR PnL FWD")

# ============================================================================
# GENERACIÓN DE VISUALIZACIONES
# ============================================================================
print("\n" + "="*80)
print("GENERANDO VISUALIZACIONES...")
print("="*80)

# Figura 1: Correlaciones
fig1 = plt.figure(figsize=(20, 12))

for idx, v in enumerate(ventanas, 1):
    pnl_col = f'PnL_fwd_pts_{v}'
    pnldv_col = f'PnLDV_fwd_{v}'

    plt.subplot(2, 2, idx)

    mask = df[pnl_col].notna() & df[pnldv_col].notna()
    x = df.loc[mask, pnldv_col]
    y = df.loc[mask, pnl_col]

    # Scatter plot con densidad
    plt.hexbin(x, y, gridsize=30, cmap='YlOrRd', alpha=0.7)
    plt.colorbar(label='Densidad')

    # Línea de regresión
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    plt.plot(x_line, p(x_line), "b--", linewidth=2, label=f'Regresión lineal')

    corr = correlaciones[v]['corr']
    plt.title(f'FWD_{v}: Correlación = {corr:.3f}', fontsize=14, fontweight='bold')
    plt.xlabel(f'PnLDV FWD_{v}', fontsize=12)
    plt.ylabel(f'PnL FWD PTS_{v}', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.savefig('correlacion_fwd_pnldv_scatter.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 1 guardado: correlacion_fwd_pnldv_scatter.png")

# Figura 2: Impacto de cambios en PNLDV
fig2 = plt.figure(figsize=(20, 12))

for idx, v in enumerate(ventanas, 1):
    plt.subplot(2, 2, idx)

    pnl_col = f'PnL_fwd_pts_{v}'
    categoria_col = f'categoria_pnldv_{v}'

    data_box = []
    labels_box = []

    for cat in df[categoria_col].dropna().unique():
        data_cat = df[df[categoria_col] == cat][pnl_col].dropna()
        if len(data_cat) > 0:
            data_box.append(data_cat)
            labels_box.append(str(cat))

    bp = plt.boxplot(data_box, labels=labels_box, patch_artist=True)

    # Colorear las cajas
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.title(f'FWD_{v}: PnL según cambio en PNLDV', fontsize=14, fontweight='bold')
    plt.ylabel('PnL FWD PTS', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('impacto_cambios_pnldv.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 2 guardado: impacto_cambios_pnldv.png")

# Figura 3: Estabilidad vs Performance
fig3 = plt.figure(figsize=(20, 10))

# Subplot 1: PNL promedio
plt.subplot(1, 2, 1)
x_pos = np.arange(len(ventanas))
width = 0.35

pnl_estables = [estabilidad_results[i]['pnl_estables'] for i in range(len(ventanas))]
pnl_inestables = [estabilidad_results[i]['pnl_inestables'] for i in range(len(ventanas))]

bars1 = plt.bar(x_pos - width/2, pnl_estables, width, label='PNLDV Estable', color='green', alpha=0.7)
bars2 = plt.bar(x_pos + width/2, pnl_inestables, width, label='PNLDV Inestable', color='red', alpha=0.7)

plt.xlabel('Ventana FWD', fontsize=12)
plt.ylabel('PNL Promedio (pts)', fontsize=12)
plt.title('PNL Promedio: Estabilidad de PNLDV FWD', fontsize=14, fontweight='bold')
plt.xticks(x_pos, [f'FWD_{v}' for v in ventanas])
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

# Subplot 2: Win Rate
plt.subplot(1, 2, 2)
wr_estables = [estabilidad_results[i]['wr_estables'] for i in range(len(ventanas))]
wr_inestables = [estabilidad_results[i]['wr_inestables'] for i in range(len(ventanas))]

bars1 = plt.bar(x_pos - width/2, wr_estables, width, label='PNLDV Estable', color='green', alpha=0.7)
bars2 = plt.bar(x_pos + width/2, wr_inestables, width, label='PNLDV Inestable', color='red', alpha=0.7)

plt.xlabel('Ventana FWD', fontsize=12)
plt.ylabel('Win Rate (%)', fontsize=12)
plt.title('Win Rate: Estabilidad de PNLDV FWD', fontsize=14, fontweight='bold')
plt.xticks(x_pos, [f'FWD_{v}' for v in ventanas])
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.axhline(y=50, color='black', linestyle='--', linewidth=1, label='50% break-even')

plt.tight_layout()
plt.savefig('estabilidad_pnldv_performance.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 3 guardado: estabilidad_pnldv_performance.png")

# Figura 4: Evolución temporal de PNLDV
fig4 = plt.figure(figsize=(16, 10))

plt.subplot(2, 1, 1)
# Evolución promedio
ventanas_labels = ['T+0'] + [f'FWD_{v}' for v in ventanas]
pnldv_means = [df['PnLDV'].mean()] + [df[f'PnLDV_fwd_{v}'].mean() for v in ventanas]

plt.plot(ventanas_labels, pnldv_means, marker='o', linewidth=2, markersize=10, color='blue')
plt.fill_between(range(len(ventanas_labels)), pnldv_means, alpha=0.3, color='blue')
plt.xlabel('Ventana Temporal', fontsize=12)
plt.ylabel('PNLDV Promedio', fontsize=12)
plt.title('Evolución Promedio de PNLDV a través del tiempo', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
# Cambio desde T+0
cambios = [0] + [df[f'delta_pnldv_{v}'].mean() for v in ventanas]
colores = ['green' if c >= 0 else 'red' for c in cambios]

plt.bar(ventanas_labels, cambios, color=colores, alpha=0.7)
plt.xlabel('Ventana Temporal', fontsize=12)
plt.ylabel('Cambio desde T+0 (pts)', fontsize=12)
plt.title('Cambio Promedio de PNLDV desde T+0', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig('evolucion_temporal_pnldv.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 4 guardado: evolucion_temporal_pnldv.png")

# Figura 5: Heatmap de correlaciones
fig5 = plt.figure(figsize=(14, 10))

# Crear matriz de correlación
cols_analisis = []
for v in ventanas:
    cols_analisis.extend([f'PnL_fwd_pts_{v}', f'PnLDV_fwd_{v}'])

corr_matrix = df[cols_analisis].corr()

# Renombrar para mejor visualización
rename_dict = {}
for v in ventanas:
    rename_dict[f'PnL_fwd_pts_{v}'] = f'PNL_{v}'
    rename_dict[f'PnLDV_fwd_{v}'] = f'PNLDV_{v}'

corr_matrix = corr_matrix.rename(columns=rename_dict, index=rename_dict)

# Heatmap
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlación: PNL FWD vs PNLDV FWD', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('heatmap_correlaciones_fwd.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 5 guardado: heatmap_correlaciones_fwd.png")

# ============================================================================
# RESUMEN EJECUTIVO
# ============================================================================
print("\n" + "="*80)
print("RESUMEN EJECUTIVO Y CONCLUSIONES")
print("="*80)

print("\n1. CORRELACIONES PRINCIPALES:")
for v in ventanas:
    corr = correlaciones[v]['corr']
    pval = correlaciones[v]['pval']
    print(f"   FWD_{v}: r = {corr:+.3f} (p < {'0.001' if pval < 0.001 else f'{pval:.3f}'})")

print("\n2. IMPACTO DE CAÍDAS DE PNLDV FWD:")
for item in caidas_analysis:
    print(f"   FWD_{item['ventana']}: {item['diferencia']:+.2f} pts de diferencia " +
          f"({'SIGNIFICATIVO' if item['pval'] < 0.05 else 'no significativo'})")

print("\n3. EFECTO DE ESTABILIDAD DE PNLDV FWD:")
for item in estabilidad_results:
    print(f"   FWD_{item['ventana']}: {item['dif_pnl']:+.2f} pts mejor con PNLDV estable " +
          f"({'SIGNIFICATIVO' if item['pval'] < 0.05 else 'no significativo'})")

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO EXITOSAMENTE")
print("="*80)
print("\nArchivos generados:")
print("  1. correlacion_fwd_pnldv_scatter.png")
print("  2. impacto_cambios_pnldv.png")
print("  3. estabilidad_pnldv_performance.png")
print("  4. evolucion_temporal_pnldv.png")
print("  5. heatmap_correlaciones_fwd.png")
print("\n" + "="*80)
