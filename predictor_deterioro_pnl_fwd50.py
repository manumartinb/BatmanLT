"""
ANÁLISIS PREDICTIVO: Identificación de Drivers de Deterioro Grave PnL FWD 50
==============================================================================

Objetivo: Encontrar señales tempranas (T+0 o FWD intermedio) que predigan
          con alta confiabilidad un deterioro grave del PnL en W=50

Enfoque: Análisis iterativo y exhaustivo usando:
         1. Análisis univariado (variable por variable)
         2. Análisis multivariado (combinaciones)
         3. Machine Learning (Random Forest, XGBoost)
         4. Reglas interpretables
         5. Sistema de scoring predictivo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuración
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (20, 12)

print("="*100)
print("ANÁLISIS PREDICTIVO: DRIVERS DE DETERIORO GRAVE PnL FWD 50")
print("="*100)
print(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Cargar datos
df = pd.read_csv('PNLDV.csv')
print(f"Dataset original: {len(df)} operaciones")

# Filtrar operaciones válidas (con datos en W=50)
df_analisis = df[df['PnL_fwd_pts_50'].notna()].copy()
print(f"Operaciones con PnL FWD 50 válido: {len(df_analisis)}")
print(f"Excluidas: {len(df) - len(df_analisis)} operaciones\n")

# ============================================================================
# PASO 1: DEFINIR "DETERIORO GRAVE"
# ============================================================================

print("="*100)
print("PASO 1: DEFINICIÓN DE DETERIORO GRAVE")
print("="*100)

# Analizar distribución de PnL FWD 50
pnl_50 = df_analisis['PnL_fwd_pts_50']

print("\nDistribución de PnL FWD 50:")
print(f"  Media:      {pnl_50.mean():>8.2f} pts")
print(f"  Mediana:    {pnl_50.median():>8.2f} pts")
print(f"  Desv.Std:   {pnl_50.std():>8.2f} pts")
print(f"  Mínimo:     {pnl_50.min():>8.2f} pts")
print(f"  Máximo:     {pnl_50.max():>8.2f} pts")

print("\nPercentiles:")
for p in [5, 10, 25, 50, 75, 90, 95]:
    val = pnl_50.quantile(p/100)
    print(f"  P{p:>2}: {val:>8.2f} pts")

# Definir múltiples umbrales de deterioro
umbrales_deterioro = {
    'Muy Grave (< -150)': -150,
    'Grave (< -100)': -100,
    'Moderado (< -50)': -50,
    'Leve (< 0)': 0
}

print("\nDefiniciones de deterioro y frecuencias:")
print("-"*80)
for nombre, umbral in umbrales_deterioro.items():
    n_deterioro = (pnl_50 < umbral).sum()
    pct = n_deterioro / len(pnl_50) * 100
    print(f"  {nombre:<25} {n_deterioro:>5} ops ({pct:>5.1f}%)")

# Usar deterioro GRAVE (< -100) como objetivo principal
UMBRAL_DETERIORO = -100
df_analisis['deterioro_grave'] = (df_analisis['PnL_fwd_pts_50'] < UMBRAL_DETERIORO).astype(int)

n_deterioro = df_analisis['deterioro_grave'].sum()
n_ok = len(df_analisis) - n_deterioro

print(f"\n{'='*80}")
print(f"OBJETIVO: Predecir deterioro GRAVE (PnL FWD 50 < {UMBRAL_DETERIORO} pts)")
print(f"  Operaciones con deterioro grave: {n_deterioro} ({n_deterioro/len(df_analisis)*100:.1f}%)")
print(f"  Operaciones OK:                  {n_ok} ({n_ok/len(df_analisis)*100:.1f}%)")
print(f"{'='*80}\n")

# ============================================================================
# PASO 2: IDENTIFICAR VARIABLES CANDIDATAS
# ============================================================================

print("="*100)
print("PASO 2: EXPLORACIÓN DE VARIABLES CANDIDATAS")
print("="*100)

# Categorías de variables
variables_t0 = {
    'Estructura': ['k1', 'k2', 'k3', 'DTE1/DTE2', 'net_credit', 'net_credit_diff'],
    'Precios': ['price_mid_short1', 'price_mid_long2', 'price_mid_short3'],
    'Volatilidad': ['iv_k1', 'iv_k2', 'iv_k3'],
    'Greeks': ['delta_total', 'theta_total', 'delta_k1', 'delta_k2', 'delta_k3'],
    'Calidad': ['BQI_ABS', 'BQI_V2_ABS', 'BQR_1000', 'Asym', 'FF_ATM', 'FF_BAT'],
    'Riesgo': ['Death valley', 'PnLDV', 'EarL', 'EarR', 'EarScore'],
    'Mercado': ['SPX']
}

variables_fwd = {
    'PnLDV_FWD': ['PnLDV_fwd_01', 'PnLDV_fwd_05', 'PnLDV_fwd_25'],
    'PnL_FWD': ['PnL_fwd_pts_01', 'PnL_fwd_pts_05', 'PnL_fwd_pts_25'],
    'Cambios_PnLDV': ['delta_pnldv_25'] if 'delta_pnldv_25' in df_analisis.columns else []
}

# Agregar variables derivadas
df_analisis['spread_width'] = df_analisis['k3'] - df_analisis['k1']
df_analisis['wing_ratio'] = (df_analisis['k2'] - df_analisis['k1']) / (df_analisis['k3'] - df_analisis['k2'])
df_analisis['iv_spread'] = df_analisis['iv_k3'] - df_analisis['iv_k1']
df_analisis['delta_net'] = abs(df_analisis['delta_total'])

# Calcular cambios de PnLDV si no existen
if 'delta_pnldv_25' not in df_analisis.columns:
    df_analisis['delta_pnldv_25'] = df_analisis['PnLDV_fwd_25'] - df_analisis['PnLDV']

print("\nVariables disponibles por categoría:")
for categoria, vars_list in variables_t0.items():
    disponibles = [v for v in vars_list if v in df_analisis.columns]
    print(f"  {categoria:<15} {len(disponibles)} variables: {', '.join(disponibles[:3])}...")

# ============================================================================
# PASO 3: ANÁLISIS UNIVARIADO - ENCONTRAR MEJORES PREDICTORES INDIVIDUALES
# ============================================================================

print("\n" + "="*100)
print("PASO 3: ANÁLISIS UNIVARIADO - Poder Predictivo Individual")
print("="*100)

# Función para analizar poder predictivo de una variable
def analizar_predictor(df, variable, target='deterioro_grave', n_bins=5):
    """Analiza el poder predictivo de una variable individual"""

    data = df[[variable, target]].dropna()
    if len(data) == 0:
        return None

    # Crear quintiles
    try:
        data['quintil'] = pd.qcut(data[variable], q=n_bins, labels=False, duplicates='drop')
    except:
        return None

    # Calcular tasa de deterioro por quintil
    analisis = data.groupby('quintil').agg({
        target: ['sum', 'count', 'mean'],
        variable: ['min', 'max', 'mean']
    }).round(3)

    # Calcular métricas de poder predictivo
    tasa_base = data[target].mean()
    tasas_quintiles = data.groupby('quintil')[target].mean()

    # Rango de tasas (diferencia entre quintil con mayor y menor tasa)
    rango = tasas_quintiles.max() - tasas_quintiles.min()

    # Correlación
    corr = data[variable].corr(data[target])

    # AUC-ROC (si variable es continua)
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(data[target], data[variable])
        auc = max(auc, 1 - auc)  # Tomar el mayor (dirección correcta)
    except:
        auc = 0.5

    return {
        'variable': variable,
        'correlacion': corr,
        'auc': auc,
        'rango_tasas': rango,
        'tasa_base': tasa_base,
        'mejor_quintil': tasas_quintiles.idxmax(),
        'peor_quintil': tasas_quintiles.idxmin(),
        'tasa_mejor': tasas_quintiles.max(),
        'tasa_peor': tasas_quintiles.min(),
        'n_validos': len(data),
        'analisis': analisis
    }

# Analizar todas las variables
print("\nAnalizando poder predictivo de cada variable...")

resultados_univariados = []

# Variables T+0
for categoria, vars_list in variables_t0.items():
    for var in vars_list:
        if var in df_analisis.columns:
            resultado = analizar_predictor(df_analisis, var)
            if resultado:
                resultado['categoria'] = categoria
                resultado['momento'] = 'T+0'
                resultados_univariados.append(resultado)

# Variables FWD
for categoria, vars_list in variables_fwd.items():
    for var in vars_list:
        if var in df_analisis.columns:
            resultado = analizar_predictor(df_analisis, var)
            if resultado:
                resultado['categoria'] = categoria
                resultado['momento'] = 'FWD'
                resultados_univariados.append(resultado)

# Variables derivadas
for var in ['spread_width', 'wing_ratio', 'iv_spread', 'delta_net']:
    if var in df_analisis.columns:
        resultado = analizar_predictor(df_analisis, var)
        if resultado:
            resultado['categoria'] = 'Derivadas'
            resultado['momento'] = 'T+0'
            resultados_univariados.append(resultado)

df_predictores = pd.DataFrame(resultados_univariados)

# Ordenar por poder predictivo
df_predictores['score_predictivo'] = (
    abs(df_predictores['correlacion']) * 0.3 +
    (df_predictores['auc'] - 0.5) * 2 * 0.3 +
    df_predictores['rango_tasas'] * 0.4
)

df_predictores = df_predictores.sort_values('score_predictivo', ascending=False)

print("\n" + "="*100)
print("TOP 20 MEJORES PREDICTORES INDIVIDUALES")
print("="*100)
print(f"{'#':<3} {'Variable':<30} {'Momento':<8} {'Corr':<8} {'AUC':<7} {'Rango':<8} {'Score':<8}")
print("-"*100)

for idx, row in df_predictores.head(20).iterrows():
    print(f"{idx+1:<3} {row['variable']:<30} {row['momento']:<8} "
          f"{row['correlacion']:>7.3f} {row['auc']:>6.3f} {row['rango_tasas']:>7.3f} "
          f"{row['score_predictivo']:>7.3f}")

# Análisis detallado de los TOP 5 predictores
print("\n" + "="*100)
print("ANÁLISIS DETALLADO: TOP 5 PREDICTORES")
print("="*100)

for idx, row in df_predictores.head(5).iterrows():
    print(f"\n{idx+1}. {row['variable']} ({row['momento']})")
    print("-"*80)
    print(f"   Correlación con deterioro: {row['correlacion']:+.3f}")
    print(f"   AUC-ROC: {row['auc']:.3f}")
    print(f"   Tasa base de deterioro: {row['tasa_base']*100:.1f}%")
    print(f"   Tasa en quintil PEOR: {row['tasa_peor']*100:.1f}%")
    print(f"   Tasa en quintil MEJOR: {row['tasa_mejor']*100:.1f}%")
    print(f"   Rango (diferencia): {row['rango_tasas']*100:.1f}%")

    # Mostrar umbrales de quintiles
    data_var = df_analisis[[row['variable'], 'deterioro_grave']].dropna()
    quintiles = pd.qcut(data_var[row['variable']], q=5, duplicates='drop')

    print(f"\n   Tasa de deterioro por quintil:")
    for q in range(5):
        mask = pd.qcut(data_var[row['variable']], q=5, labels=False, duplicates='drop') == q
        if mask.sum() > 0:
            tasa = data_var[mask]['deterioro_grave'].mean()
            val_min = data_var[mask][row['variable']].min()
            val_max = data_var[mask][row['variable']].max()
            n = mask.sum()
            print(f"     Q{q+1} [{val_min:>8.2f} a {val_max:>8.2f}]: {tasa*100:>5.1f}% ({n} ops)")

# ============================================================================
# PASO 4: ENCONTRAR UMBRALES CRÍTICOS
# ============================================================================

print("\n" + "="*100)
print("PASO 4: UMBRALES CRÍTICOS - Reglas Simples de Alta Precisión")
print("="*100)

# Para cada top predictor, encontrar umbrales que maximicen precisión
def encontrar_umbral_optimo(df, variable, target='deterioro_grave', direccion='menor'):
    """Encuentra el umbral que maximiza precision manteniendo recall razonable"""

    data = df[[variable, target]].dropna()
    valores = sorted(data[variable].unique())

    mejores_umbrales = []

    for percentil in [5, 10, 15, 20, 25]:
        if direccion == 'menor':
            umbral = data[variable].quantile(percentil/100)
            mascara = data[variable] <= umbral
        else:
            umbral = data[variable].quantile(1 - percentil/100)
            mascara = data[variable] >= umbral

        if mascara.sum() > 10:  # Mínimo 10 operaciones
            precision = data[mascara][target].mean()
            recall = data[mascara][target].sum() / data[target].sum()
            n_ops = mascara.sum()

            mejores_umbrales.append({
                'umbral': umbral,
                'precision': precision,
                'recall': recall,
                'n_ops': n_ops,
                'percentil': percentil
            })

    return pd.DataFrame(mejores_umbrales)

print("\nREGLAS SIMPLES DE ALTA PRECISIÓN:")
print("="*100)

reglas_simples = []

for idx, row in df_predictores.head(10).iterrows():
    variable = row['variable']

    # Determinar dirección (mayor o menor valor = mayor riesgo)
    corr = row['correlacion']
    direccion = 'mayor' if corr > 0 else 'menor'

    umbrales = encontrar_umbral_optimo(df_analisis, variable, direccion=direccion)

    if len(umbrales) > 0:
        # Seleccionar umbral con mejor balance precision-recall
        umbrales['score'] = umbrales['precision'] * 0.7 + umbrales['recall'] * 0.3
        mejor = umbrales.sort_values('score', ascending=False).iloc[0]

        if mejor['precision'] > 0.25:  # Solo si precision > 25% (vs ~13% base)
            reglas_simples.append({
                'variable': variable,
                'umbral': mejor['umbral'],
                'direccion': direccion,
                'precision': mejor['precision'],
                'recall': mejor['recall'],
                'n_ops': mejor['n_ops'],
                'percentil': mejor['percentil'],
                'lift': mejor['precision'] / row['tasa_base']
            })

df_reglas = pd.DataFrame(reglas_simples).sort_values('precision', ascending=False)

print(f"\n{'Variable':<30} {'Condición':<15} {'Umbral':<12} {'Precisión':<12} {'Recall':<10} {'Lift':<8} {'N ops'}")
print("-"*100)

for _, regla in df_reglas.head(15).iterrows():
    operador = '<=' if regla['direccion'] == 'menor' else '>='
    condicion = f"{operador} {regla['umbral']:.2f}"
    print(f"{regla['variable']:<30} {condicion:<15} (P{regla['percentil']:<2})      "
          f"{regla['precision']*100:>6.1f}%      {regla['recall']*100:>6.1f}%    "
          f"{regla['lift']:>6.2f}x  {regla['n_ops']:.0f}")

print(f"\nNota: Lift = cuántas veces más probable es deterioro vs tasa base ({n_deterioro/len(df_analisis)*100:.1f}%)")

# ============================================================================
# PASO 5: COMBINACIONES DE VARIABLES (Reglas AND)
# ============================================================================

print("\n" + "="*100)
print("PASO 5: REGLAS COMBINADAS (AND) - Mayor Precisión")
print("="*100)

print("\nBuscando combinaciones de 2 variables que maximicen precisión...")

reglas_combinadas = []

# Probar combinaciones de los top predictores
top_vars = df_predictores.head(15)['variable'].tolist()

from itertools import combinations

for var1, var2 in combinations(top_vars, 2):
    # Para cada variable, usar su mejor umbral
    regla1 = df_reglas[df_reglas['variable'] == var1]
    regla2 = df_reglas[df_reglas['variable'] == var2]

    if len(regla1) > 0 and len(regla2) > 0:
        regla1 = regla1.iloc[0]
        regla2 = regla2.iloc[0]

        # Aplicar ambas condiciones
        if regla1['direccion'] == 'menor':
            cond1 = df_analisis[var1] <= regla1['umbral']
        else:
            cond1 = df_analisis[var1] >= regla1['umbral']

        if regla2['direccion'] == 'menor':
            cond2 = df_analisis[var2] <= regla2['umbral']
        else:
            cond2 = df_analisis[var2] >= regla2['umbral']

        mascara_combinada = cond1 & cond2
        n_ops = mascara_combinada.sum()

        if n_ops >= 10:  # Mínimo 10 operaciones
            precision = df_analisis[mascara_combinada]['deterioro_grave'].mean()
            recall = df_analisis[mascara_combinada]['deterioro_grave'].sum() / n_deterioro

            if precision > 0.30:  # Solo si precision > 30%
                reglas_combinadas.append({
                    'var1': var1,
                    'var2': var2,
                    'umbral1': regla1['umbral'],
                    'dir1': regla1['direccion'],
                    'umbral2': regla2['umbral'],
                    'dir2': regla2['direccion'],
                    'precision': precision,
                    'recall': recall,
                    'n_ops': n_ops,
                    'lift': precision / (n_deterioro/len(df_analisis))
                })

df_reglas_combinadas = pd.DataFrame(reglas_combinadas).sort_values('precision', ascending=False)

print(f"\nTOP 15 REGLAS COMBINADAS (ordenadas por precisión):")
print("="*100)

for idx, regla in df_reglas_combinadas.head(15).iterrows():
    op1 = '<=' if regla['dir1'] == 'menor' else '>='
    op2 = '<=' if regla['dir2'] == 'menor' else '>='

    print(f"\n{idx+1}. SI {regla['var1']} {op1} {regla['umbral1']:.2f}")
    print(f"   Y  {regla['var2']} {op2} {regla['umbral2']:.2f}")
    print(f"   → Precisión: {regla['precision']*100:.1f}% | Recall: {regla['recall']*100:.1f}% | "
          f"Lift: {regla['lift']:.2f}x | N: {regla['n_ops']:.0f} ops")

# Continúa en la siguiente parte...
print("\n[Continuando con Machine Learning y Sistema de Scoring...]")
