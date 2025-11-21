"""
ANÁLISIS DE PERSISTENCIA DE DETERIORO: FWD 5 → FWD 25 → FWD 50
=============================================================

Pregunta: Si un trade sufre deterioro en FWD 5, ¿también sufre deterioro en FWD 25?
         Si un trade sufre deterioro en FWD 25, ¿también sufre deterioro en FWD 50?

Este análisis examina la persistencia del deterioro a través del tiempo.
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
plt.rcParams['figure.figsize'] = (20, 14)

print("="*100)
print("ANÁLISIS DE PERSISTENCIA DE DETERIORO: FWD 5 → FWD 25 → FWD 50")
print("="*100)
print(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Cargar datos
df = pd.read_csv('PNLDV.csv')
print(f"Dataset original: {len(df)} operaciones\n")

# Filtrar operaciones válidas (con datos en todas las ventanas)
df_analisis = df[
    df['PnL_fwd_pts_05'].notna() &
    df['PnL_fwd_pts_25'].notna() &
    df['PnL_fwd_pts_50'].notna()
].copy()

print(f"Operaciones con datos válidos en FWD 5, 25 y 50: {len(df_analisis)}")
print(f"Excluidas: {len(df) - len(df_analisis)} operaciones\n")

# ============================================================================
# DEFINIR NIVELES DE DETERIORO
# ============================================================================

print("="*100)
print("PASO 1: DEFINICIÓN DE DETERIORO")
print("="*100)

# Definir múltiples umbrales de deterioro
umbrales_deterioro = {
    'grave': -100,      # Deterioro grave
    'moderado': -50,    # Deterioro moderado
    'leve': 0,          # Cualquier pérdida
}

print("\nUmbrales de deterioro definidos:")
for nombre, valor in umbrales_deterioro.items():
    print(f"  {nombre.capitalize():<12} PnL < {valor} pts")

# Crear variables de deterioro para cada ventana
for ventana in ['05', '25', '50']:
    col_pnl = f'PnL_fwd_pts_{ventana}'

    # Deterioro grave
    df_analisis[f'deterioro_grave_{ventana}'] = (df_analisis[col_pnl] < umbrales_deterioro['grave']).astype(int)

    # Deterioro moderado
    df_analisis[f'deterioro_moderado_{ventana}'] = (df_analisis[col_pnl] < umbrales_deterioro['moderado']).astype(int)

    # Deterioro leve (cualquier pérdida)
    df_analisis[f'deterioro_leve_{ventana}'] = (df_analisis[col_pnl] < umbrales_deterioro['leve']).astype(int)

# Mostrar frecuencias
print("\nFrecuencia de deterioro por ventana:")
print("-"*100)
print(f"{'Ventana':<12} {'Grave (<-100)':<20} {'Moderado (<-50)':<20} {'Leve (<0)':<20}")
print("-"*100)

for ventana in ['05', '25', '50']:
    grave = df_analisis[f'deterioro_grave_{ventana}'].sum()
    moderado = df_analisis[f'deterioro_moderado_{ventana}'].sum()
    leve = df_analisis[f'deterioro_leve_{ventana}'].sum()

    print(f"FWD_{ventana:<6} {grave:>6} ({grave/len(df_analisis)*100:>5.1f}%)    "
          f"{moderado:>6} ({moderado/len(df_analisis)*100:>5.1f}%)    "
          f"{leve:>6} ({leve/len(df_analisis)*100:>5.1f}%)")

# ============================================================================
# ANÁLISIS 1: PERSISTENCIA FWD 5 → FWD 25
# ============================================================================

print("\n" + "="*100)
print("ANÁLISIS 1: PERSISTENCIA DE DETERIORO FWD 5 → FWD 25")
print("="*100)

def analizar_persistencia(df, ventana_inicial, ventana_final, tipo_deterioro='grave'):
    """Analiza cuántos deterioros en ventana inicial persisten en ventana final"""

    col_inicial = f'deterioro_{tipo_deterioro}_{ventana_inicial}'
    col_final = f'deterioro_{tipo_deterioro}_{ventana_final}'

    # Operaciones con deterioro en ventana inicial
    deterioro_inicial = df[df[col_inicial] == 1].copy()
    n_deterioro_inicial = len(deterioro_inicial)

    if n_deterioro_inicial == 0:
        return None

    # De estas, cuántas siguen con deterioro en ventana final
    persisten = deterioro_inicial[deterioro_inicial[col_final] == 1]
    n_persisten = len(persisten)

    # De estas, cuántas se recuperan
    recuperan = deterioro_inicial[deterioro_inicial[col_final] == 0]
    n_recuperan = len(recuperan)

    # Tasa de persistencia
    tasa_persistencia = n_persisten / n_deterioro_inicial * 100 if n_deterioro_inicial > 0 else 0
    tasa_recuperacion = n_recuperan / n_deterioro_inicial * 100 if n_deterioro_inicial > 0 else 0

    # Operaciones SIN deterioro inicial
    sin_deterioro_inicial = df[df[col_inicial] == 0].copy()
    n_sin_deterioro_inicial = len(sin_deterioro_inicial)

    # De estas, cuántas desarrollan deterioro
    desarrollan = sin_deterioro_inicial[sin_deterioro_inicial[col_final] == 1]
    n_desarrollan = len(desarrollan)

    tasa_desarrollo = n_desarrollan / n_sin_deterioro_inicial * 100 if n_sin_deterioro_inicial > 0 else 0

    # PnL promedio de los que persisten vs los que recuperan
    pnl_col_final = f'PnL_fwd_pts_{ventana_final}'

    if n_persisten > 0:
        pnl_persisten = persisten[pnl_col_final].mean()
    else:
        pnl_persisten = np.nan

    if n_recuperan > 0:
        pnl_recuperan = recuperan[pnl_col_final].mean()
    else:
        pnl_recuperan = np.nan

    return {
        'n_deterioro_inicial': n_deterioro_inicial,
        'n_persisten': n_persisten,
        'n_recuperan': n_recuperan,
        'tasa_persistencia': tasa_persistencia,
        'tasa_recuperacion': tasa_recuperacion,
        'n_sin_deterioro_inicial': n_sin_deterioro_inicial,
        'n_desarrollan': n_desarrollan,
        'tasa_desarrollo': tasa_desarrollo,
        'pnl_persisten': pnl_persisten,
        'pnl_recuperan': pnl_recuperan
    }

# Análisis para cada nivel de deterioro
tipos_deterioro = ['grave', 'moderado', 'leve']

for tipo in tipos_deterioro:
    print(f"\n{'='*100}")
    print(f"DETERIORO {tipo.upper()} (PnL < {umbrales_deterioro[tipo]} pts)")
    print("="*100)

    resultado = analizar_persistencia(df_analisis, '05', '25', tipo)

    if resultado:
        print(f"\n📊 OPERACIONES CON DETERIORO {tipo.upper()} EN FWD 5: {resultado['n_deterioro_inicial']}")
        print("-"*100)
        print(f"  ✅ Se RECUPERAN en FWD 25:     {resultado['n_recuperan']:>6} ({resultado['tasa_recuperacion']:>5.1f}%)")
        print(f"  ❌ PERSISTEN en FWD 25:        {resultado['n_persisten']:>6} ({resultado['tasa_persistencia']:>5.1f}%)")

        if not np.isnan(resultado['pnl_recuperan']):
            print(f"\n  PnL promedio en FWD 25:")
            print(f"    Los que se recuperan:  {resultado['pnl_recuperan']:>8.2f} pts")
        if not np.isnan(resultado['pnl_persisten']):
            print(f"    Los que persisten:     {resultado['pnl_persisten']:>8.2f} pts")

        print(f"\n📊 OPERACIONES SIN DETERIORO {tipo.upper()} EN FWD 5: {resultado['n_sin_deterioro_inicial']}")
        print("-"*100)
        print(f"  ⚠️ DESARROLLAN deterioro en FWD 25: {resultado['n_desarrollan']:>6} ({resultado['tasa_desarrollo']:>5.1f}%)")

        print(f"\n🎯 CONCLUSIÓN:")
        print(f"  Si un trade tiene deterioro {tipo} en FWD 5,")
        print(f"  hay {resultado['tasa_persistencia']:.1f}% de probabilidad de que PERSISTA en FWD 25")

# ============================================================================
# ANÁLISIS 2: PERSISTENCIA FWD 25 → FWD 50
# ============================================================================

print("\n\n" + "="*100)
print("ANÁLISIS 2: PERSISTENCIA DE DETERIORO FWD 25 → FWD 50")
print("="*100)

for tipo in tipos_deterioro:
    print(f"\n{'='*100}")
    print(f"DETERIORO {tipo.upper()} (PnL < {umbrales_deterioro[tipo]} pts)")
    print("="*100)

    resultado = analizar_persistencia(df_analisis, '25', '50', tipo)

    if resultado:
        print(f"\n📊 OPERACIONES CON DETERIORO {tipo.upper()} EN FWD 25: {resultado['n_deterioro_inicial']}")
        print("-"*100)
        print(f"  ✅ Se RECUPERAN en FWD 50:     {resultado['n_recuperan']:>6} ({resultado['tasa_recuperacion']:>5.1f}%)")
        print(f"  ❌ PERSISTEN en FWD 50:        {resultado['n_persisten']:>6} ({resultado['tasa_persistencia']:>5.1f}%)")

        if not np.isnan(resultado['pnl_recuperan']):
            print(f"\n  PnL promedio en FWD 50:")
            print(f"    Los que se recuperan:  {resultado['pnl_recuperan']:>8.2f} pts")
        if not np.isnan(resultado['pnl_persisten']):
            print(f"    Los que persisten:     {resultado['pnl_persisten']:>8.2f} pts")

        print(f"\n📊 OPERACIONES SIN DETERIORO {tipo.upper()} EN FWD 25: {resultado['n_sin_deterioro_inicial']}")
        print("-"*100)
        print(f"  ⚠️ DESARROLLAN deterioro en FWD 50: {resultado['n_desarrollan']:>6} ({resultado['tasa_desarrollo']:>5.1f}%)")

        print(f"\n🎯 CONCLUSIÓN:")
        print(f"  Si un trade tiene deterioro {tipo} en FWD 25,")
        print(f"  hay {resultado['tasa_persistencia']:.1f}% de probabilidad de que PERSISTA en FWD 50")

# ============================================================================
# MATRIZ DE TRANSICIÓN
# ============================================================================

print("\n\n" + "="*100)
print("MATRICES DE TRANSICIÓN DE ESTADOS")
print("="*100)

def crear_matriz_transicion(df, ventana_inicial, ventana_final, tipo_deterioro='grave'):
    """Crea una matriz de transición de estados"""

    col_inicial = f'deterioro_{tipo_deterioro}_{ventana_inicial}'
    col_final = f'deterioro_{tipo_deterioro}_{ventana_final}'

    # Crear tabla de contingencia
    matriz = pd.crosstab(
        df[col_inicial],
        df[col_final],
        rownames=[f'FWD_{ventana_inicial}'],
        colnames=[f'FWD_{ventana_final}'],
        margins=True
    )

    # Calcular probabilidades condicionales (por fila)
    matriz_prob = pd.crosstab(
        df[col_inicial],
        df[col_final],
        rownames=[f'FWD_{ventana_inicial}'],
        colnames=[f'FWD_{ventana_final}'],
        normalize='index'
    ) * 100

    return matriz, matriz_prob

# Matriz FWD 5 → FWD 25 (Deterioro Grave)
print("\n1. DETERIORO GRAVE (<-100 pts): FWD 5 → FWD 25")
print("-"*100)
matriz_5_25, matriz_5_25_prob = crear_matriz_transicion(df_analisis, '05', '25', 'grave')

print("\nConteo de operaciones:")
print(matriz_5_25)
print("\nProbabilidades condicionales (%):")
print(matriz_5_25_prob.round(1))

# Matriz FWD 25 → FWD 50 (Deterioro Grave)
print("\n\n2. DETERIORO GRAVE (<-100 pts): FWD 25 → FWD 50")
print("-"*100)
matriz_25_50, matriz_25_50_prob = crear_matriz_transicion(df_analisis, '25', '50', 'grave')

print("\nConteo de operaciones:")
print(matriz_25_50)
print("\nProbabilidades condicionales (%):")
print(matriz_25_50_prob.round(1))

# Matriz FWD 5 → FWD 25 (Deterioro Leve)
print("\n\n3. DETERIORO LEVE (<0 pts): FWD 5 → FWD 25")
print("-"*100)
matriz_5_25_leve, matriz_5_25_leve_prob = crear_matriz_transicion(df_analisis, '05', '25', 'leve')

print("\nConteo de operaciones:")
print(matriz_5_25_leve)
print("\nProbabilidades condicionales (%):")
print(matriz_5_25_leve_prob.round(1))

# Matriz FWD 25 → FWD 50 (Deterioro Leve)
print("\n\n4. DETERIORO LEVE (<0 pts): FWD 25 → FWD 50")
print("-"*100)
matriz_25_50_leve, matriz_25_50_leve_prob = crear_matriz_transicion(df_analisis, '25', '50', 'leve')

print("\nConteo de operaciones:")
print(matriz_25_50_leve)
print("\nProbabilidades condicionales (%):")
print(matriz_25_50_leve_prob.round(1))

# ============================================================================
# ANÁLISIS DE TRAYECTORIAS COMPLETAS
# ============================================================================

print("\n\n" + "="*100)
print("ANÁLISIS DE TRAYECTORIAS COMPLETAS: FWD 5 → FWD 25 → FWD 50")
print("="*100)

# Crear variable de trayectoria para deterioro grave
df_analisis['trayectoria_grave'] = (
    df_analisis['deterioro_grave_05'].astype(str) + ' → ' +
    df_analisis['deterioro_grave_25'].astype(str) + ' → ' +
    df_analisis['deterioro_grave_50'].astype(str)
)

# Mapear a nombres descriptivos
trayectoria_map = {
    '0 → 0 → 0': 'Siempre OK',
    '0 → 0 → 1': 'Deteriora solo en FWD50',
    '0 → 1 → 0': 'Deteriora en FWD25, recupera',
    '0 → 1 → 1': 'Deteriora desde FWD25',
    '1 → 0 → 0': 'Deteriora en FWD5, recupera',
    '1 → 0 → 1': 'Recupera en FWD25, deteriora en FWD50',
    '1 → 1 → 0': 'Deterioro FWD5 y 25, recupera',
    '1 → 1 → 1': 'Deterioro persistente'
}

df_analisis['trayectoria_nombre'] = df_analisis['trayectoria_grave'].map(trayectoria_map)

print("\nDETERIORO GRAVE (<-100 pts) - Distribución de trayectorias:")
print("-"*100)

trayectorias = df_analisis['trayectoria_nombre'].value_counts().sort_values(ascending=False)

for trayectoria, count in trayectorias.items():
    pct = count / len(df_analisis) * 100
    pnl_medio = df_analisis[df_analisis['trayectoria_nombre'] == trayectoria]['PnL_fwd_pts_50'].mean()
    print(f"{trayectoria:<40} {count:>6} ops ({pct:>5.1f}%)  |  PnL FWD50 promedio: {pnl_medio:>8.2f} pts")

# Análisis de recuperación vs persistencia
print("\n\n🔍 ANÁLISIS DE RECUPERACIÓN vs PERSISTENCIA:")
print("="*100)

# Operaciones que empiezan mal en FWD5
empiezan_mal = df_analisis[df_analisis['deterioro_grave_05'] == 1].copy()

if len(empiezan_mal) > 0:
    # Recuperan completamente
    recuperan_completo = empiezan_mal[empiezan_mal['deterioro_grave_50'] == 0]

    # Persisten hasta el final
    persisten_completo = empiezan_mal[empiezan_mal['deterioro_grave_50'] == 1]

    print(f"\nDe {len(empiezan_mal)} operaciones con deterioro grave en FWD 5:")
    print(f"  ✅ {len(recuperan_completo)} ({len(recuperan_completo)/len(empiezan_mal)*100:.1f}%) se recuperan para FWD 50")
    print(f"  ❌ {len(persisten_completo)} ({len(persisten_completo)/len(empiezan_mal)*100:.1f}%) persisten con deterioro hasta FWD 50")

    if len(recuperan_completo) > 0:
        print(f"\n  PnL FWD 50 de los que se recuperan:")
        print(f"    Promedio: {recuperan_completo['PnL_fwd_pts_50'].mean():>8.2f} pts")
        print(f"    Mediana:  {recuperan_completo['PnL_fwd_pts_50'].median():>8.2f} pts")

    if len(persisten_completo) > 0:
        print(f"\n  PnL FWD 50 de los que persisten con deterioro:")
        print(f"    Promedio: {persisten_completo['PnL_fwd_pts_50'].mean():>8.2f} pts")
        print(f"    Mediana:  {persisten_completo['PnL_fwd_pts_50'].median():>8.2f} pts")

# ============================================================================
# VISUALIZACIONES
# ============================================================================

print("\n\n" + "="*100)
print("GENERANDO VISUALIZACIONES...")
print("="*100)

# Figura 1: Diagramas de Sankey (flujo de estados)
fig1 = plt.figure(figsize=(22, 14))

# Subplot 1: Tasas de persistencia para deterioro grave
plt.subplot(2, 3, 1)
resultado_5_25_grave = analizar_persistencia(df_analisis, '05', '25', 'grave')
resultado_25_50_grave = analizar_persistencia(df_analisis, '25', '50', 'grave')

categorias = ['FWD 5→25', 'FWD 25→50']
tasas_persistencia = [resultado_5_25_grave['tasa_persistencia'], resultado_25_50_grave['tasa_persistencia']]
tasas_recuperacion = [resultado_5_25_grave['tasa_recuperacion'], resultado_25_50_grave['tasa_recuperacion']]

x = np.arange(len(categorias))
width = 0.35

bars1 = plt.bar(x - width/2, tasas_persistencia, width, label='Persiste', color='red', alpha=0.7)
bars2 = plt.bar(x + width/2, tasas_recuperacion, width, label='Recupera', color='green', alpha=0.7)

plt.xlabel('Transición', fontsize=12)
plt.ylabel('Porcentaje (%)', fontsize=12)
plt.title('Deterioro GRAVE: Persistencia vs Recuperación', fontsize=13, fontweight='bold')
plt.xticks(x, categorias)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Añadir valores en las barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

# Subplot 2: Tasas de persistencia para deterioro leve
plt.subplot(2, 3, 2)
resultado_5_25_leve = analizar_persistencia(df_analisis, '05', '25', 'leve')
resultado_25_50_leve = analizar_persistencia(df_analisis, '25', '50', 'leve')

tasas_persistencia_leve = [resultado_5_25_leve['tasa_persistencia'], resultado_25_50_leve['tasa_persistencia']]
tasas_recuperacion_leve = [resultado_5_25_leve['tasa_recuperacion'], resultado_25_50_leve['tasa_recuperacion']]

bars1 = plt.bar(x - width/2, tasas_persistencia_leve, width, label='Persiste', color='orange', alpha=0.7)
bars2 = plt.bar(x + width/2, tasas_recuperacion_leve, width, label='Recupera', color='lightgreen', alpha=0.7)

plt.xlabel('Transición', fontsize=12)
plt.ylabel('Porcentaje (%)', fontsize=12)
plt.title('Deterioro LEVE: Persistencia vs Recuperación', fontsize=13, fontweight='bold')
plt.xticks(x, categorias)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

# Subplot 3: Matriz de transición FWD 5 → 25 (grave)
plt.subplot(2, 3, 3)
sns.heatmap(matriz_5_25_prob, annot=True, fmt='.1f', cmap='RdYlGn_r', center=50,
            cbar_kws={'label': 'Probabilidad (%)'}, linewidths=2, linecolor='white')
plt.title('Deterioro GRAVE: FWD 5 → FWD 25\n(Probabilidades condicionales %)', fontsize=13, fontweight='bold')
plt.ylabel('Estado en FWD 5', fontsize=11)
plt.xlabel('Estado en FWD 25', fontsize=11)

# Subplot 4: Matriz de transición FWD 25 → 50 (grave)
plt.subplot(2, 3, 4)
sns.heatmap(matriz_25_50_prob, annot=True, fmt='.1f', cmap='RdYlGn_r', center=50,
            cbar_kws={'label': 'Probabilidad (%)'}, linewidths=2, linecolor='white')
plt.title('Deterioro GRAVE: FWD 25 → FWD 50\n(Probabilidades condicionales %)', fontsize=13, fontweight='bold')
plt.ylabel('Estado en FWD 25', fontsize=11)
plt.xlabel('Estado en FWD 50', fontsize=11)

# Subplot 5: Distribución de trayectorias
plt.subplot(2, 3, 5)
trayectorias_top = trayectorias.head(8)
colors_traj = ['green' if 'Siempre OK' in t or 'recupera' in t else 'red' for t in trayectorias_top.index]

plt.barh(range(len(trayectorias_top)), trayectorias_top.values, color=colors_traj, alpha=0.7)
plt.yticks(range(len(trayectorias_top)), trayectorias_top.index, fontsize=10)
plt.xlabel('Número de Operaciones', fontsize=11)
plt.title('Trayectorias Completas\n(FWD 5 → 25 → 50, Deterioro Grave)', fontsize=13, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')

# Subplot 6: PnL promedio por trayectoria
plt.subplot(2, 3, 6)
pnl_por_trayectoria = df_analisis.groupby('trayectoria_nombre')['PnL_fwd_pts_50'].mean().sort_values()
colors_pnl = ['red' if v < 0 else 'green' for v in pnl_por_trayectoria.values]

plt.barh(range(len(pnl_por_trayectoria)), pnl_por_trayectoria.values, color=colors_pnl, alpha=0.7)
plt.yticks(range(len(pnl_por_trayectoria)), pnl_por_trayectoria.index, fontsize=10)
plt.xlabel('PnL FWD 50 Promedio (pts)', fontsize=11)
plt.title('PnL FWD 50 Promedio por Trayectoria', fontsize=13, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=2)
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('persistencia_deterioro_analisis.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: persistencia_deterioro_analisis.png")

# ============================================================================
# RESUMEN EJECUTIVO
# ============================================================================

print("\n\n" + "="*100)
print("RESUMEN EJECUTIVO")
print("="*100)

print("\n🎯 PREGUNTA 1: ¿Cuántos trades con deterioro en FWD 5 persisten en FWD 25?")
print("-"*100)

for tipo in ['grave', 'leve']:
    resultado = analizar_persistencia(df_analisis, '05', '25', tipo)
    if resultado:
        print(f"\nDeterioro {tipo.upper()} (< {umbrales_deterioro[tipo]} pts):")
        print(f"  De {resultado['n_deterioro_inicial']} operaciones con deterioro en FWD 5:")
        print(f"    → {resultado['tasa_persistencia']:.1f}% ({resultado['n_persisten']} ops) PERSISTEN en FWD 25")
        print(f"    → {resultado['tasa_recuperacion']:.1f}% ({resultado['n_recuperan']} ops) SE RECUPERAN")

print("\n\n🎯 PREGUNTA 2: ¿Cuántos trades con deterioro en FWD 25 persisten en FWD 50?")
print("-"*100)

for tipo in ['grave', 'leve']:
    resultado = analizar_persistencia(df_analisis, '25', '50', tipo)
    if resultado:
        print(f"\nDeterioro {tipo.upper()} (< {umbrales_deterioro[tipo]} pts):")
        print(f"  De {resultado['n_deterioro_inicial']} operaciones con deterioro en FWD 25:")
        print(f"    → {resultado['tasa_persistencia']:.1f}% ({resultado['n_persisten']} ops) PERSISTEN en FWD 50")
        print(f"    → {resultado['tasa_recuperacion']:.1f}% ({resultado['n_recuperan']} ops) SE RECUPERAN")

print("\n\n💡 CONCLUSIONES CLAVE:")
print("="*100)

# Calcular conclusiones clave
resultado_grave_5_25 = analizar_persistencia(df_analisis, '05', '25', 'grave')
resultado_grave_25_50 = analizar_persistencia(df_analisis, '25', '50', 'grave')

print(f"\n1. DETERIORO GRAVE:")
print(f"   • Si un trade tiene deterioro grave en FWD 5:")
print(f"     → {resultado_grave_5_25['tasa_persistencia']:.1f}% de probabilidad de persistir en FWD 25")
print(f"   • Si un trade tiene deterioro grave en FWD 25:")
print(f"     → {resultado_grave_25_50['tasa_persistencia']:.1f}% de probabilidad de persistir en FWD 50")

print(f"\n2. SEÑAL TEMPRANA:")
if resultado_grave_5_25['tasa_persistencia'] > 70:
    print(f"   • El deterioro grave en FWD 5 es una SEÑAL FUERTE de problemas persistentes")
    print(f"   • {resultado_grave_5_25['tasa_persistencia']:.1f}% de persistencia indica alta probabilidad de continuación")
elif resultado_grave_5_25['tasa_recuperacion'] > 50:
    print(f"   • El deterioro grave en FWD 5 NO es señal definitiva - muchos se recuperan")
    print(f"   • {resultado_grave_5_25['tasa_recuperacion']:.1f}% de recuperación indica alta resiliencia")

print(f"\n3. PUNTO DE NO RETORNO:")
if resultado_grave_25_50['tasa_persistencia'] > 70:
    print(f"   • FWD 25 es punto crítico - deterioro grave aquí rara vez se recupera")
    print(f"   • {resultado_grave_25_50['tasa_persistencia']:.1f}% de persistencia hasta FWD 50")
else:
    print(f"   • Incluso en FWD 25, hay posibilidad de recuperación")
    print(f"   • {resultado_grave_25_50['tasa_recuperacion']:.1f}% de las operaciones se recuperan")

print("\n" + "="*100)
print("ANÁLISIS COMPLETADO")
print("="*100)
