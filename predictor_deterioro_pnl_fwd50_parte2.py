"""
ANÁLISIS PREDICTIVO PARTE 2: Machine Learning y Sistema de Scoring
===================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Continuar con los datos del script anterior
# (Este script asume que se ejecutó el anterior primero)

# ============================================================================
# PASO 6: MACHINE LEARNING - IMPORTANCIA DE VARIABLES
# ============================================================================

print("\n" + "="*100)
print("PASO 6: MACHINE LEARNING - Feature Importance")
print("="*100)

# Preparar datos para ML
# Seleccionar variables relevantes (solo T+0 y FWD hasta W=25)
variables_ml = []

# Variables T+0
for categoria, vars_list in variables_t0.items():
    for var in vars_list:
        if var in df_analisis.columns and df_analisis[var].notna().sum() > 1000:
            variables_ml.append(var)

# Variables FWD hasta W=25 (no incluir W=50 para no hacer trampa)
for var in ['PnLDV_fwd_01', 'PnLDV_fwd_05', 'PnLDV_fwd_25',
            'PnL_fwd_pts_01', 'PnL_fwd_pts_05', 'PnL_fwd_pts_25']:
    if var in df_analisis.columns and df_analisis[var].notna().sum() > 1000:
        variables_ml.append(var)

# Variables derivadas
for var in ['spread_width', 'wing_ratio', 'iv_spread', 'delta_net']:
    if var in df_analisis.columns:
        variables_ml.append(var)

# Agregar cambio de PnLDV en W=25
if 'delta_pnldv_25' not in df_analisis.columns:
    df_analisis['delta_pnldv_25'] = df_analisis['PnLDV_fwd_25'] - df_analisis['PnLDV']
variables_ml.append('delta_pnldv_25')

print(f"\nVariables para Machine Learning: {len(variables_ml)}")
print(f"Variables: {', '.join(variables_ml[:10])}...")

# Crear dataset para ML
X = df_analisis[variables_ml].copy()
y = df_analisis['deterioro_grave'].copy()

# Eliminar filas con NaN
mask_validos = X.notna().all(axis=1) & y.notna()
X_clean = X[mask_validos]
y_clean = y[mask_validos]

print(f"\nDatos válidos para ML: {len(X_clean)} operaciones")
print(f"  Deterioro grave: {y_clean.sum()} ({y_clean.mean()*100:.1f}%)")
print(f"  OK: {(~y_clean.astype(bool)).sum()} ({(~y_clean.astype(bool)).mean()*100:.1f}%)")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.3, random_state=42, stratify=y_clean
)

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ============================================================================
# Modelo 1: Random Forest
# ============================================================================

print("\n" + "-"*100)
print("MODELO 1: RANDOM FOREST")
print("-"*100)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=50,
    min_samples_leaf=20,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)

# Predicciones
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Métricas
auc_rf = roc_auc_score(y_test, y_prob_rf)
print(f"\nAUC-ROC en Test: {auc_rf:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['OK', 'Deterioro']))

# Feature Importance
feature_importance_rf = pd.DataFrame({
    'variable': X_clean.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTOP 20 VARIABLES MÁS IMPORTANTES (Random Forest):")
print("-"*80)
print(f"{'#':<4} {'Variable':<35} {'Importancia':<15}")
print("-"*80)

for idx, row in feature_importance_rf.head(20).iterrows():
    print(f"{idx+1:<4} {row['variable']:<35} {row['importance']:.4f}")

# ============================================================================
# Modelo 2: Gradient Boosting
# ============================================================================

print("\n" + "-"*100)
print("MODELO 2: GRADIENT BOOSTING")
print("-"*100)

gb_model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    min_samples_split=50,
    min_samples_leaf=20,
    random_state=42
)

gb_model.fit(X_train, y_train)

# Predicciones
y_pred_gb = gb_model.predict(X_test)
y_prob_gb = gb_model.predict_proba(X_test)[:, 1]

# Métricas
auc_gb = roc_auc_score(y_test, y_prob_gb)
print(f"\nAUC-ROC en Test: {auc_gb:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_gb, target_names=['OK', 'Deterioro']))

# Feature Importance
feature_importance_gb = pd.DataFrame({
    'variable': X_clean.columns,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTOP 20 VARIABLES MÁS IMPORTANTES (Gradient Boosting):")
print("-"*80)
print(f"{'#':<4} {'Variable':<35} {'Importancia':<15}")
print("-"*80)

for idx, row in feature_importance_gb.head(20).iterrows():
    print(f"{idx+1:<4} {row['variable']:<35} {row['importance']:.4f}")

# ============================================================================
# Modelo 3: Árbol de Decisión Simple (interpretable)
# ============================================================================

print("\n" + "-"*100)
print("MODELO 3: ÁRBOL DE DECISIÓN SIMPLE (interpretable)")
print("-"*100)

# Usar solo top variables para árbol simple
top_vars_arbol = feature_importance_rf.head(10)['variable'].tolist()

X_arbol = X_clean[top_vars_arbol]
X_train_arbol, X_test_arbol, y_train_arbol, y_test_arbol = train_test_split(
    X_arbol, y_clean, test_size=0.3, random_state=42, stratify=y_clean
)

tree_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=100,
    min_samples_leaf=50,
    random_state=42,
    class_weight='balanced'
)

tree_model.fit(X_train_arbol, y_train_arbol)

# Predicciones
y_pred_tree = tree_model.predict(X_test_arbol)
y_prob_tree = tree_model.predict_proba(X_test_arbol)[:, 1]

# Métricas
auc_tree = roc_auc_score(y_test_arbol, y_prob_tree)
print(f"\nAUC-ROC en Test: {auc_tree:.3f}")

print("\nClassification Report:")
print(classification_report(y_test_arbol, y_pred_tree, target_names=['OK', 'Deterioro']))

# ============================================================================
# PASO 7: SISTEMA DE SCORING PREDICTIVO
# ============================================================================

print("\n" + "="*100)
print("PASO 7: SISTEMA DE SCORING PREDICTIVO")
print("="*100)

# Crear scoring basado en los mejores predictores y reglas encontradas
# Scoring de 0 a 100: mayor score = mayor riesgo de deterioro

def calcular_risk_score(row, reglas_df, feature_importance_df):
    """Calcula un risk score de 0 a 100"""

    score = 0
    detalles = []

    # Componente 1: Top 5 predictores (40 puntos)
    top5_vars = feature_importance_df.head(5)['variable'].tolist()

    for var in top5_vars:
        if var in row.index and pd.notna(row[var]):
            # Normalizar valor de la variable
            val = row[var]

            # Buscar en reglas simples si existe umbral
            regla = reglas_df[reglas_df['variable'] == var]

            if len(regla) > 0:
                regla = regla.iloc[0]
                umbral = regla['umbral']
                direccion = regla['direccion']

                if direccion == 'menor' and val <= umbral:
                    puntos = 8 * regla['precision'] / 0.5  # Escalar por precisión
                    score += puntos
                    detalles.append(f"{var} <= {umbral:.2f}: +{puntos:.1f} pts")
                elif direccion == 'mayor' and val >= umbral:
                    puntos = 8 * regla['precision'] / 0.5
                    score += puntos
                    detalles.append(f"{var} >= {umbral:.2f}: +{puntos:.1f} pts")

    # Componente 2: Reglas combinadas (30 puntos)
    # [Implementar evaluación de reglas combinadas]

    # Componente 3: Probabilidad ML (30 puntos)
    # [Agregar predicción de ML si disponible]

    return min(score, 100), detalles  # Cap a 100

# Calcular scores para todo el dataset
print("\nCalculando Risk Scores...")

scores = []
for idx, row in df_analisis.iterrows():
    score, _ = calcular_risk_score(row, df_reglas, feature_importance_rf)
    scores.append(score)

df_analisis['risk_score'] = scores

# Analizar distribución de scores
print("\nDistribución de Risk Scores:")
print(df_analisis['risk_score'].describe())

# Analizar efectividad del scoring
print("\n" + "="*100)
print("EFECTIVIDAD DEL SISTEMA DE SCORING")
print("="*100)

# Crear deciles de risk score
df_analisis['score_decil'] = pd.qcut(df_analisis['risk_score'], q=10, labels=False, duplicates='drop')

analisis_scoring = df_analisis.groupby('score_decil').agg({
    'deterioro_grave': ['sum', 'count', 'mean'],
    'risk_score': ['min', 'max', 'mean'],
    'PnL_fwd_pts_50': 'mean'
}).round(3)

print("\nTasa de deterioro por decil de Risk Score:")
print(analisis_scoring)

# ============================================================================
# VISUALIZACIONES
# ============================================================================

print("\n" + "="*100)
print("GENERANDO VISUALIZACIONES...")
print("="*100)

# Figura 1: Feature Importance
fig1 = plt.figure(figsize=(20, 12))

plt.subplot(2, 2, 1)
top15_rf = feature_importance_rf.head(15)
plt.barh(range(len(top15_rf)), top15_rf['importance'], color='steelblue', alpha=0.7)
plt.yticks(range(len(top15_rf)), top15_rf['variable'])
plt.xlabel('Importancia', fontsize=12)
plt.title('Top 15 Variables - Random Forest', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')

plt.subplot(2, 2, 2)
top15_gb = feature_importance_gb.head(15)
plt.barh(range(len(top15_gb)), top15_gb['importance'], color='darkgreen', alpha=0.7)
plt.yticks(range(len(top15_gb)), top15_gb['variable'])
plt.xlabel('Importancia', fontsize=12)
plt.title('Top 15 Variables - Gradient Boosting', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')

plt.subplot(2, 2, 3)
# ROC Curves
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_prob_gb)
fpr_tree, tpr_tree, _ = roc_curve(y_test_arbol, y_prob_tree)

plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={auc_rf:.3f})', linewidth=2)
plt.plot(fpr_gb, tpr_gb, label=f'Gradient Boosting (AUC={auc_gb:.3f})', linewidth=2)
plt.plot(fpr_tree, tpr_tree, label=f'Decision Tree (AUC={auc_tree:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Comparación de Modelos', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
# Risk Score vs Tasa de Deterioro
if 'score_decil' in df_analisis.columns:
    decil_analisis = df_analisis.groupby('score_decil')['deterioro_grave'].agg(['mean', 'count'])
    plt.bar(decil_analisis.index, decil_analisis['mean'] * 100, color='red', alpha=0.7)
    plt.axhline(y=df_analisis['deterioro_grave'].mean()*100, color='black',
                linestyle='--', linewidth=2, label='Tasa Base')
    plt.xlabel('Decil de Risk Score', fontsize=12)
    plt.ylabel('Tasa de Deterioro (%)', fontsize=12)
    plt.title('Tasa de Deterioro por Decil de Risk Score', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('ml_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 1 guardado: ml_feature_importance.png")

# Figura 2: Árbol de Decisión
fig2 = plt.figure(figsize=(24, 12))
plot_tree(tree_model, feature_names=top_vars_arbol, class_names=['OK', 'Deterioro'],
          filled=True, rounded=True, fontsize=10)
plt.title('Árbol de Decisión Interpretable (max_depth=5)', fontsize=16, fontweight='bold', pad=20)
plt.savefig('arbol_decision_interpretable.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 2 guardado: arbol_decision_interpretable.png")

# Figura 3: Análisis detallado de top predictores
fig3 = plt.figure(figsize=(20, 16))

top5_predictores = df_predictores.head(5)

for idx, (_, predictor) in enumerate(top5_predictores.iterrows(), 1):
    plt.subplot(3, 2, idx)

    var = predictor['variable']
    data_plot = df_analisis[[var, 'deterioro_grave']].dropna()

    # Crear bins
    try:
        data_plot['bin'] = pd.qcut(data_plot[var], q=10, duplicates='drop')
        tasa_por_bin = data_plot.groupby('bin')['deterioro_grave'].mean() * 100

        x_pos = range(len(tasa_por_bin))
        plt.bar(x_pos, tasa_por_bin.values, color='red', alpha=0.7)
        plt.axhline(y=df_analisis['deterioro_grave'].mean()*100,
                   color='black', linestyle='--', linewidth=2, label='Tasa Base')
        plt.xlabel(f'{var} (deciles)', fontsize=11)
        plt.ylabel('Tasa de Deterioro (%)', fontsize=11)
        plt.title(f'{idx}. {var}\n(Corr={predictor["correlacion"]:.3f}, AUC={predictor["auc"]:.3f})',
                 fontsize=12, fontweight='bold')
        plt.xticks(x_pos, [f'D{i+1}' for i in range(len(tasa_por_bin))], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
    except:
        pass

plt.tight_layout()
plt.savefig('top_predictores_detalle.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 3 guardado: top_predictores_detalle.png")

print("\n" + "="*100)
print("ANÁLISIS PARTE 2 COMPLETADO")
print("="*100)
