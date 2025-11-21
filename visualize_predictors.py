#!/usr/bin/env python3
"""
Visualizaciones para el análisis de predictores
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

print("Generando visualizaciones...")

# Cargar resultados
results_t0 = pd.read_csv('/home/user/BatmanLT/predictors_analysis_t0.csv')
results_t25 = pd.read_csv('/home/user/BatmanLT/predictors_analysis_t25.csv')
importance_t0 = pd.read_csv('/home/user/BatmanLT/feature_importance_t0.csv')
importance_t25 = pd.read_csv('/home/user/BatmanLT/feature_importance_t25.csv')

# Cargar datos originales
df = pd.read_csv('/home/user/BatmanLT/PNLDV.csv')
df_valid = df[df['PnL_fwd_pts_50'].notna()].copy()
percentile_20 = df_valid['PnL_fwd_pts_50'].quantile(0.20)
df_valid['PnL_desastroso'] = (df_valid['PnL_fwd_pts_50'] < percentile_20).astype(int)

# Recrear variables derivadas (necesarias para visualización)
df_valid['PnLDV_per_credit'] = df_valid['PnLDV'] / (df_valid['net_credit'].abs() + 1)
df_valid['danger_score'] = (
    (df_valid['PnLDV'] < df_valid['PnLDV'].quantile(0.25)).astype(int) * 3 +
    (df_valid['BQI_ABS'] > df_valid['BQI_ABS'].quantile(0.75)).astype(int) * 2 +
    (df_valid['Death valley'] > df_valid['Death valley'].quantile(0.75)).astype(int) * 2 +
    (df_valid['delta_total'].abs() > df_valid['delta_total'].abs().quantile(0.75)).astype(int)
)
df_valid['PnL_deterioration_25'] = df_valid['PnL_fwd_pts_25'] / (df_valid['net_credit'] + 0.001)
df_valid['PnLDV_deterioration_25'] = df_valid['PnLDV_fwd_25'] / (df_valid['PnLDV'].abs() + 1)

# ============================================================================
# FIGURA 1: Overview de Predictores
# ============================================================================
fig1 = plt.figure(figsize=(20, 14))
gs1 = fig1.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1.1 Top 15 predictores por AUC (T+0)
ax1 = fig1.add_subplot(gs1[0, 0])
top15_t0 = results_t0.head(15).sort_values('auc')
ax1.barh(range(len(top15_t0)), top15_t0['auc'], color='steelblue', alpha=0.8)
ax1.set_yticks(range(len(top15_t0)))
ax1.set_yticklabels(top15_t0['variable'], fontsize=8)
ax1.set_xlabel('AUC Score')
ax1.set_title('Top 15 Predictores por AUC\n(T+0)', fontweight='bold', fontsize=11)
ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=1)
ax1.grid(axis='x', alpha=0.3)

# 1.2 Top 15 predictores por correlación absoluta (T+0)
ax2 = fig1.add_subplot(gs1[0, 1])
top15_corr = results_t0.nlargest(15, 'correlation', keep='first')
colors = ['red' if x < 0 else 'green' for x in top15_corr['correlation']]
ax2.barh(range(len(top15_corr)), top15_corr['correlation'].abs(), color=colors, alpha=0.7)
ax2.set_yticks(range(len(top15_corr)))
ax2.set_yticklabels(top15_corr['variable'], fontsize=8)
ax2.set_xlabel('|Correlation|')
ax2.set_title('Top 15 por Correlación Absoluta\n(T+0)', fontweight='bold', fontsize=11)
ax2.grid(axis='x', alpha=0.3)

# 1.3 Predictive Score combinado (T+0)
ax3 = fig1.add_subplot(gs1[0, 2])
top15_score = results_t0.head(15).sort_values('predictive_score')
ax3.barh(range(len(top15_score)), top15_score['predictive_score'], color='purple', alpha=0.8)
ax3.set_yticks(range(len(top15_score)))
ax3.set_yticklabels(top15_score['variable'], fontsize=8)
ax3.set_xlabel('Predictive Score')
ax3.set_title('Top 15 por Score Combinado\n(T+0)', fontweight='bold', fontsize=11)
ax3.grid(axis='x', alpha=0.3)

# 1.4 Feature Importance Random Forest (T+0)
ax4 = fig1.add_subplot(gs1[1, 0])
top15_imp_t0 = importance_t0.head(15).sort_values('importance')
ax4.barh(range(len(top15_imp_t0)), top15_imp_t0['importance'], color='orange', alpha=0.8)
ax4.set_yticks(range(len(top15_imp_t0)))
ax4.set_yticklabels(top15_imp_t0['variable'], fontsize=8)
ax4.set_xlabel('Feature Importance')
ax4.set_title('Top 15 Feature Importance\nRandom Forest (T+0)', fontweight='bold', fontsize=11)
ax4.grid(axis='x', alpha=0.3)

# 1.5 Feature Importance Random Forest (T+25)
ax5 = fig1.add_subplot(gs1[1, 1])
top15_imp_t25 = importance_t25.head(15).sort_values('importance')
ax5.barh(range(len(top15_imp_t25)), top15_imp_t25['importance'], color='darkgreen', alpha=0.8)
ax5.set_yticks(range(len(top15_imp_t25)))
ax5.set_yticklabels(top15_imp_t25['variable'], fontsize=8)
ax5.set_xlabel('Feature Importance')
ax5.set_title('Top 15 Feature Importance\nRandom Forest (T+0+T+25)', fontweight='bold', fontsize=11)
ax5.grid(axis='x', alpha=0.3)

# 1.6 Comparación T+0 vs T+0+T+25
ax6 = fig1.add_subplot(gs1[1, 2])
# Tomar top 10 de cada uno
top10_vars_t0 = set(results_t0.head(10)['variable'])
top10_vars_t25 = set(results_t25.head(10)['variable'])

only_t0 = len(top10_vars_t0 - top10_vars_t25)
only_t25 = len(top10_vars_t25 - top10_vars_t0)
both = len(top10_vars_t0 & top10_vars_t25)

categories = ['Solo en\nTop T+0', 'En ambos\nTops', 'Solo en\nTop T+25']
values = [only_t0, both, only_t25]
colors_venn = ['steelblue', 'purple', 'darkgreen']

ax6.bar(categories, values, color=colors_venn, alpha=0.7, edgecolor='black')
ax6.set_ylabel('Número de variables')
ax6.set_title('Solapamiento Top 10\nT+0 vs T+0+T+25', fontweight='bold', fontsize=11)
ax6.grid(axis='y', alpha=0.3)
for i, v in enumerate(values):
    ax6.text(i, v + 0.1, str(v), ha='center', fontweight='bold')

# 1.7, 1.8, 1.9: Análisis detallado de top 3 predictores
top_3_vars = importance_t25.head(3)['variable'].tolist()

for idx, var in enumerate(top_3_vars):
    ax = fig1.add_subplot(gs1[2, idx])

    if var not in df_valid.columns:
        continue

    data = df_valid[[var, 'PnL_desastroso', 'PnL_fwd_pts_50']].dropna()

    if len(data) < 100:
        continue

    # Crear quintiles
    try:
        data['quintil'] = pd.qcut(data[var], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
    except:
        continue

    # Tasa de desastre por quintil
    disaster_rate = data.groupby('quintil')['PnL_desastroso'].mean() * 100

    ax.bar(range(len(disaster_rate)), disaster_rate.values, alpha=0.7, edgecolor='black',
           color=['darkred' if x > 30 else 'orange' if x > 20 else 'green' for x in disaster_rate.values])
    ax.set_xticks(range(len(disaster_rate)))
    ax.set_xticklabels(disaster_rate.index)
    ax.set_xlabel('Quintil')
    ax.set_ylabel('Tasa de Desastre (%)')
    ax.set_title(f'#{idx+1}: {var}\nTasa de desastre por quintil', fontweight='bold', fontsize=9)
    ax.axhline(y=20, color='red', linestyle='--', linewidth=1, label='Baseline (20%)')
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    # Añadir valores
    for i, v in enumerate(disaster_rate.values):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=8)

plt.suptitle('ANÁLISIS DE PREDICTORES DE PnL DESASTROSO EN VENTANA 50',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('/home/user/BatmanLT/predictors_overview.png', dpi=150, bbox_inches='tight')
print("Guardado: predictors_overview.png")

# ============================================================================
# FIGURA 2: Análisis Detallado de Top Predictores
# ============================================================================
fig2 = plt.figure(figsize=(20, 12))
gs2 = fig2.add_gridspec(2, 3, hspace=0.25, wspace=0.3)

top_6_vars = importance_t25.head(6)['variable'].tolist()

for idx, var in enumerate(top_6_vars):
    row = idx // 3
    col = idx % 3
    ax = fig2.add_subplot(gs2[row, col])

    if var not in df_valid.columns:
        continue

    data = df_valid[[var, 'PnL_desastroso', 'PnL_fwd_pts_50']].dropna()

    if len(data) < 100:
        continue

    # Scatter plot con densidad
    disaster = data[data['PnL_desastroso'] == 1]
    normal = data[data['PnL_desastroso'] == 0]

    ax.scatter(normal[var], normal['PnL_fwd_pts_50'], alpha=0.2, s=10,
               color='blue', label=f'Normal (n={len(normal)})')
    ax.scatter(disaster[var], disaster['PnL_fwd_pts_50'], alpha=0.4, s=15,
               color='red', label=f'Desastroso (n={len(disaster)})')

    ax.axhline(y=percentile_20, color='red', linestyle='--', linewidth=2,
               label=f'Umbral desastroso ({percentile_20:.1f})')

    ax.set_xlabel(var, fontsize=9)
    ax.set_ylabel('PnL FWD 50 (pts)', fontsize=9)
    ax.set_title(f'#{idx+1}: {var}', fontweight='bold', fontsize=10)
    ax.legend(fontsize=7, loc='best')
    ax.grid(alpha=0.3)

plt.suptitle('ANÁLISIS DETALLADO: Top 6 Predictores',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('/home/user/BatmanLT/predictors_detailed.png', dpi=150, bbox_inches='tight')
print("Guardado: predictors_detailed.png")

# ============================================================================
# FIGURA 3: Comparación de Categorías de Predictores
# ============================================================================
fig3, axes = plt.subplots(2, 2, figsize=(16, 12))

# Categorizar variables
def categorize_variable(var_name):
    if var_name in ['PnLDV_fwd_25', 'PnL_fwd_pts_25', 'PnL_deterioration_25', 'PnLDV_deterioration_25', 'SPX_chg_pct_25']:
        return 'T+25'
    elif 'per_credit' in var_name or 'ratio' in var_name or '_x_' in var_name or 'deterioration' in var_name or 'danger_score' in var_name:
        return 'Derivada'
    elif var_name.startswith('iv_') or 'iv' in var_name:
        return 'IV/Volatilidad'
    elif var_name in ['delta_total', 'theta_total'] or any(x in var_name for x in ['delta', 'theta']):
        return 'Greeks'
    elif var_name in ['PnLDV', 'BQI_ABS', 'Death valley', 'EarL', 'EarR', 'EarScore']:
        return 'Estructura'
    else:
        return 'Otras'

results_t25['category'] = results_t25['variable'].apply(categorize_variable)

# 3.1 Distribución por categoría
ax1 = axes[0, 0]
category_counts = results_t25['category'].value_counts()
colors_cat = plt.cm.Set3(range(len(category_counts)))
ax1.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
        colors=colors_cat, startangle=90)
ax1.set_title('Distribución de Variables\npor Categoría', fontweight='bold')

# 3.2 AUC promedio por categoría
ax2 = axes[0, 1]
category_auc = results_t25.groupby('category')['auc'].mean().sort_values()
ax2.barh(range(len(category_auc)), category_auc.values, color='steelblue', alpha=0.7)
ax2.set_yticks(range(len(category_auc)))
ax2.set_yticklabels(category_auc.index)
ax2.set_xlabel('AUC Promedio')
ax2.set_title('Poder Predictivo Promedio\npor Categoría', fontweight='bold')
ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=1)
ax2.grid(axis='x', alpha=0.3)

# 3.3 Top variables por categoría
ax3 = axes[1, 0]
top_by_category = results_t25.groupby('category').apply(
    lambda x: x.nlargest(1, 'predictive_score')
).reset_index(drop=True)
top_by_category = top_by_category.sort_values('predictive_score')

ax3.barh(range(len(top_by_category)), top_by_category['predictive_score'],
         color='purple', alpha=0.7)
ax3.set_yticks(range(len(top_by_category)))
ax3.set_yticklabels([f"{row['category']}\n({row['variable'][:20]}...)"
                      for _, row in top_by_category.iterrows()], fontsize=8)
ax3.set_xlabel('Predictive Score')
ax3.set_title('Mejor Variable de Cada Categoría', fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# 3.4 Matriz de correlación entre top predictores
ax4 = axes[1, 1]
top_5_for_corr = importance_t25.head(5)['variable'].tolist()
top_5_for_corr = [v for v in top_5_for_corr if v in df_valid.columns]

if len(top_5_for_corr) >= 3:
    corr_data = df_valid[top_5_for_corr].corr()
    im = ax4.imshow(corr_data, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax4.set_xticks(range(len(top_5_for_corr)))
    ax4.set_yticks(range(len(top_5_for_corr)))
    ax4.set_xticklabels([v[:15] for v in top_5_for_corr], rotation=45, ha='right', fontsize=8)
    ax4.set_yticklabels([v[:15] for v in top_5_for_corr], fontsize=8)
    ax4.set_title('Correlación entre\nTop 5 Predictores', fontweight='bold')

    # Añadir valores
    for i in range(len(top_5_for_corr)):
        for j in range(len(top_5_for_corr)):
            text = ax4.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)

    plt.colorbar(im, ax=ax4)

plt.suptitle('ANÁLISIS POR CATEGORÍAS DE PREDICTORES',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

plt.savefig('/home/user/BatmanLT/predictors_categories.png', dpi=150, bbox_inches='tight')
print("Guardado: predictors_categories.png")

print("\n" + "="*80)
print("VISUALIZACIONES COMPLETADAS")
print("="*80)
print("Archivos generados:")
print("- predictors_overview.png")
print("- predictors_detailed.png")
print("- predictors_categories.png")
