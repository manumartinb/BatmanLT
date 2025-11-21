#!/usr/bin/env python3
"""
Análisis de correlación entre PnL FWD de diferentes configuraciones DTE1/DTE2
que comienzan en la misma fecha.

Este script responde a la pregunta: ¿Los trades con diferentes configuraciones
DTE1/DTE2 que parten del mismo día se mueven correlacionados en términos de PnL FWD,
o la diferencia de DTE les da independencia?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data(filepath):
    """Carga los datos del CSV"""
    print("Cargando datos...")
    df = pd.read_csv(filepath)
    df['dia'] = pd.to_datetime(df['dia'])
    print(f"Total de trades: {len(df)}")
    print(f"Rango de fechas: {df['dia'].min()} a {df['dia'].max()}")
    print(f"Configuraciones DTE1/DTE2 únicas: {df['DTE1/DTE2'].nunique()}")
    return df

def analyze_overlap_dates(df):
    """Analiza cuántas fechas tienen múltiples configuraciones DTE"""
    overlap_analysis = df.groupby('dia').agg({
        'DTE1/DTE2': ['count', 'nunique'],
        'PnL_fwd_pts_50': 'count'
    }).reset_index()
    overlap_analysis.columns = ['dia', 'num_trades', 'num_configuraciones', 'pnl_count']

    # Filtrar fechas con múltiples configuraciones
    multi_config_dates = overlap_analysis[overlap_analysis['num_configuraciones'] > 1]

    print(f"\n=== ANÁLISIS DE OVERLAP ===")
    print(f"Fechas únicas totales: {len(overlap_analysis)}")
    print(f"Fechas con múltiples configuraciones DTE: {len(multi_config_dates)}")
    print(f"Promedio de configuraciones por fecha: {overlap_analysis['num_configuraciones'].mean():.2f}")
    print(f"Máximo de configuraciones en una fecha: {overlap_analysis['num_configuraciones'].max()}")

    return multi_config_dates, overlap_analysis

def create_correlation_matrix(df, pnl_column='PnL_fwd_pts_50', top_n=100):
    """
    Crea una matriz de correlación entre diferentes configuraciones DTE
    basándose en trades que comienzan en fechas similares
    """
    print(f"\n=== ANÁLISIS DE CORRELACIÓN ({pnl_column}) ===")

    # Seleccionar solo las top N configuraciones más comunes
    top_configs = df['DTE1/DTE2'].value_counts().head(top_n).index.tolist()
    print(f"Analizando las {len(top_configs)} configuraciones más comunes")

    df_top = df[df['DTE1/DTE2'].isin(top_configs)].copy()

    # Filtrar solo fechas con múltiples configuraciones
    dates_with_multiple = df_top.groupby('dia')['DTE1/DTE2'].nunique()
    dates_with_multiple = dates_with_multiple[dates_with_multiple > 1].index

    df_multi = df_top[df_top['dia'].isin(dates_with_multiple)].copy()
    print(f"Trades en fechas con múltiples configuraciones: {len(df_multi)}")

    # Pivot: filas = fechas, columnas = configuraciones DTE, valores = PnL
    pivot_df = df_multi.pivot_table(
        index='dia',
        columns='DTE1/DTE2',
        values=pnl_column,
        aggfunc='first'  # Si hay duplicados, tomar el primero
    )

    # Eliminar columnas con demasiados NaN
    threshold = 0.1  # Al menos 10% de datos (más permisivo)
    pivot_df = pivot_df.dropna(thresh=int(len(pivot_df) * threshold), axis=1)

    print(f"Configuraciones DTE con suficientes datos: {len(pivot_df.columns)}")
    print(f"Fechas con datos: {len(pivot_df)}")

    # Calcular correlación
    corr_matrix = pivot_df.corr(method='pearson')

    return corr_matrix, pivot_df

def analyze_specific_examples(df, config1, config2, pnl_column='PnL_fwd_pts_50'):
    """
    Analiza la correlación entre dos configuraciones específicas
    """
    print(f"\n=== ANÁLISIS ESPECÍFICO: {config1} vs {config2} ===")

    # Filtrar trades con estas configuraciones
    df1 = df[df['DTE1/DTE2'] == config1][['dia', pnl_column]].rename(columns={pnl_column: 'pnl1'})
    df2 = df[df['DTE1/DTE2'] == config2][['dia', pnl_column]].rename(columns={pnl_column: 'pnl2'})

    # Merge por fecha para encontrar trades en el mismo día
    merged = pd.merge(df1, df2, on='dia', how='inner')

    if len(merged) < 3:
        print(f"Insuficientes datos coincidentes ({len(merged)} puntos)")
        return None

    print(f"Fechas coincidentes: {len(merged)}")

    # Calcular correlaciones
    pearson_corr, pearson_p = pearsonr(merged['pnl1'], merged['pnl2'])
    spearman_corr, spearman_p = spearmanr(merged['pnl1'], merged['pnl2'])

    print(f"Correlación de Pearson: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
    print(f"Correlación de Spearman: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")

    # Estadísticas descriptivas
    print(f"\nEstadísticas {config1}:")
    print(f"  Media: {merged['pnl1'].mean():.2f}")
    print(f"  Std: {merged['pnl1'].std():.2f}")
    print(f"\nEstadísticas {config2}:")
    print(f"  Media: {merged['pnl2'].mean():.2f}")
    print(f"  Std: {merged['pnl2'].std():.2f}")

    return merged, pearson_corr, spearman_corr

def plot_correlation_heatmap(corr_matrix, output_file='correlacion_dte_heatmap.png'):
    """Visualiza la matriz de correlación como heatmap"""
    fig, ax = plt.subplots(figsize=(16, 14))

    # Ordenar por valor de DTE1 (primer número)
    sorted_cols = sorted(corr_matrix.columns,
                        key=lambda x: tuple(map(int, str(x).split('/'))))
    corr_matrix_sorted = corr_matrix.loc[sorted_cols, sorted_cols]

    sns.heatmap(corr_matrix_sorted,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn',
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Correlación de Pearson'},
                ax=ax)

    ax.set_title('Correlación de PnL FWD entre Configuraciones DTE1/DTE2\n(Trades que comienzan en las mismas fechas)',
                 fontsize=14, pad=20)
    ax.set_xlabel('Configuración DTE1/DTE2', fontsize=12)
    ax.set_ylabel('Configuración DTE1/DTE2', fontsize=12)

    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nHeatmap guardado en: {output_file}")
    plt.close()

def plot_scatter_examples(df, examples, pnl_column='PnL_fwd_pts_50'):
    """Crea scatter plots para ejemplos específicos"""
    n_examples = len(examples)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (config1, config2) in enumerate(examples):
        if idx >= 4:
            break

        result = analyze_specific_examples(df, config1, config2, pnl_column)

        if result is None:
            axes[idx].text(0.5, 0.5, f'Datos insuficientes\n{config1} vs {config2}',
                          ha='center', va='center', fontsize=12)
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
            continue

        merged, pearson_corr, spearman_corr = result

        # Scatter plot
        axes[idx].scatter(merged['pnl1'], merged['pnl2'], alpha=0.6, s=50)

        # Línea de tendencia
        z = np.polyfit(merged['pnl1'], merged['pnl2'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(merged['pnl1'].min(), merged['pnl1'].max(), 100)
        axes[idx].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

        # Línea de correlación perfecta (referencia)
        axes[idx].plot([merged['pnl1'].min(), merged['pnl1'].max()],
                       [merged['pnl1'].min(), merged['pnl1'].max()],
                       'k:', alpha=0.3, linewidth=1, label='Correlación perfecta')

        axes[idx].set_xlabel(f'PnL FWD {config1}', fontsize=10)
        axes[idx].set_ylabel(f'PnL FWD {config2}', fontsize=10)
        axes[idx].set_title(f'{config1} vs {config2}\nPearson: {pearson_corr:.3f} | Spearman: {spearman_corr:.3f}\n(n={len(merged)} fechas coincidentes)',
                           fontsize=10)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(fontsize=8)

    plt.suptitle('Correlación de PnL FWD entre Configuraciones DTE (Mismo Día de Inicio)',
                 fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig('correlacion_dte_scatter.png', dpi=300, bbox_inches='tight')
    print("Scatter plots guardados en: correlacion_dte_scatter.png")
    plt.close()

def analyze_by_dte_distance(df, pnl_column='PnL_fwd_pts_50', top_n=50):
    """
    Analiza cómo la distancia entre DTEs afecta la correlación
    """
    print("\n=== ANÁLISIS POR DISTANCIA DTE ===")

    # Obtener solo las configuraciones más comunes para hacer el análisis manejable
    configs = df['DTE1/DTE2'].value_counts().head(top_n).index.tolist()
    print(f"Analizando las {len(configs)} configuraciones más comunes")

    # Parsear DTE1 y DTE2
    dte_dict = {}
    for config in configs:
        try:
            dte1, dte2 = map(int, config.split('/'))
            dte_dict[config] = (dte1, dte2)
        except:
            continue

    correlations_by_distance = []

    # Calcular correlación para cada par de configuraciones
    for i, config1 in enumerate(configs):
        if config1 not in dte_dict:
            continue
        for config2 in configs[i+1:]:
            if config2 not in dte_dict:
                continue

            # Calcular distancia entre configuraciones
            dte1_a, dte2_a = dte_dict[config1]
            dte1_b, dte2_b = dte_dict[config2]

            dist_dte1 = abs(dte1_a - dte1_b)
            dist_dte2 = abs(dte2_a - dte2_b)
            dist_avg = (dist_dte1 + dist_dte2) / 2

            # Encontrar fechas coincidentes
            df1 = df[df['DTE1/DTE2'] == config1][['dia', pnl_column]].rename(columns={pnl_column: 'pnl1'})
            df2 = df[df['DTE1/DTE2'] == config2][['dia', pnl_column]].rename(columns={pnl_column: 'pnl2'})
            merged = pd.merge(df1, df2, on='dia', how='inner')

            if len(merged) >= 5:  # Mínimo 5 puntos
                corr, _ = pearsonr(merged['pnl1'], merged['pnl2'])
                correlations_by_distance.append({
                    'config1': config1,
                    'config2': config2,
                    'dist_dte1': dist_dte1,
                    'dist_dte2': dist_dte2,
                    'dist_avg': dist_avg,
                    'correlation': corr,
                    'n_points': len(merged)
                })

    corr_df = pd.DataFrame(correlations_by_distance)

    if len(corr_df) > 0:
        # Análisis por bins de distancia
        corr_df['dist_bin'] = pd.cut(corr_df['dist_avg'], bins=5, labels=['Muy cercano', 'Cercano', 'Medio', 'Lejano', 'Muy lejano'])

        print("\nCorrelación promedio por distancia DTE:")
        print(corr_df.groupby('dist_bin')['correlation'].agg(['mean', 'std', 'count']))

        # Gráfico
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Scatter: Distancia vs Correlación
        scatter = ax1.scatter(corr_df['dist_avg'], corr_df['correlation'],
                            c=corr_df['n_points'], cmap='viridis',
                            s=50, alpha=0.6)
        ax1.set_xlabel('Distancia Promedio DTE', fontsize=12)
        ax1.set_ylabel('Correlación de Pearson', fontsize=12)
        ax1.set_title('Correlación vs Distancia entre Configuraciones DTE', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.colorbar(scatter, ax=ax1, label='Número de fechas coincidentes')

        # Boxplot por bins
        corr_df.boxplot(column='correlation', by='dist_bin', ax=ax2)
        ax2.set_xlabel('Distancia DTE', fontsize=12)
        ax2.set_ylabel('Correlación de Pearson', fontsize=12)
        ax2.set_title('Distribución de Correlación por Distancia DTE', fontsize=14)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.suptitle('')

        plt.tight_layout()
        plt.savefig('correlacion_vs_distancia_dte.png', dpi=300, bbox_inches='tight')
        print("\nGráfico de distancia guardado en: correlacion_vs_distancia_dte.png")
        plt.close()

        return corr_df

    return None

def generate_summary_report(df, corr_matrix, corr_df, multi_config_dates):
    """Genera un informe resumen con las conclusiones"""

    report = []
    report.append("=" * 80)
    report.append("ANÁLISIS DE CORRELACIÓN PnL FWD ENTRE CONFIGURACIONES DTE1/DTE2")
    report.append("=" * 80)
    report.append("")
    report.append("PREGUNTA: ¿Los trades con diferentes configuraciones DTE1/DTE2 que comienzan")
    report.append("en la misma fecha se mueven correlacionados en términos de PnL FWD,")
    report.append("o la diferencia de DTE les da independencia?")
    report.append("")
    report.append("-" * 80)
    report.append("1. DATOS GENERALES")
    report.append("-" * 80)
    report.append(f"Total de trades analizados: {len(df):,}")
    report.append(f"Configuraciones DTE1/DTE2 únicas: {df['DTE1/DTE2'].nunique()}")
    report.append(f"Fechas únicas: {df['dia'].nunique():,}")
    report.append(f"Fechas con múltiples configuraciones: {len(multi_config_dates):,}")
    report.append(f"Porcentaje de fechas con múltiples configs: {len(multi_config_dates)/df['dia'].nunique()*100:.1f}%")
    report.append("")

    report.append("-" * 80)
    report.append("2. ANÁLISIS DE CORRELACIÓN")
    report.append("-" * 80)

    # Estadísticas de la matriz de correlación
    # Obtener solo el triángulo superior (sin diagonal)
    if len(corr_matrix) > 0:
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        correlations = upper_triangle.values.flatten()
        correlations = correlations[~np.isnan(correlations)]

        if len(correlations) > 0:
            report.append(f"Número de pares de configuraciones analizadas: {len(correlations)}")
            report.append(f"Correlación promedio: {np.mean(correlations):.4f}")
            report.append(f"Correlación mediana: {np.median(correlations):.4f}")
            report.append(f"Desviación estándar: {np.std(correlations):.4f}")
            report.append(f"Rango: [{np.min(correlations):.4f}, {np.max(correlations):.4f}]")
            report.append("")

            # Distribución de correlaciones
            report.append("Distribución de correlaciones:")
            report.append(f"  Muy fuerte (|r| > 0.7): {np.sum(np.abs(correlations) > 0.7)} ({np.sum(np.abs(correlations) > 0.7)/len(correlations)*100:.1f}%)")
            report.append(f"  Fuerte (0.5 < |r| <= 0.7): {np.sum((np.abs(correlations) > 0.5) & (np.abs(correlations) <= 0.7))} ({np.sum((np.abs(correlations) > 0.5) & (np.abs(correlations) <= 0.7))/len(correlations)*100:.1f}%)")
            report.append(f"  Moderada (0.3 < |r| <= 0.5): {np.sum((np.abs(correlations) > 0.3) & (np.abs(correlations) <= 0.5))} ({np.sum((np.abs(correlations) > 0.3) & (np.abs(correlations) <= 0.5))/len(correlations)*100:.1f}%)")
            report.append(f"  Débil (0.1 < |r| <= 0.3): {np.sum((np.abs(correlations) > 0.1) & (np.abs(correlations) <= 0.3))} ({np.sum((np.abs(correlations) > 0.1) & (np.abs(correlations) <= 0.3))/len(correlations)*100:.1f}%)")
            report.append(f"  Muy débil (|r| <= 0.1): {np.sum(np.abs(correlations) <= 0.1)} ({np.sum(np.abs(correlations) <= 0.1)/len(correlations)*100:.1f}%)")
            report.append("")
        else:
            report.append("No se encontraron suficientes datos en la matriz de correlación.")
            report.append("(Muchas configuraciones DTE son únicas y no se repiten en las mismas fechas)")
            report.append("")
            # Usar datos del análisis por distancia
            if corr_df is not None and len(corr_df) > 0:
                correlations_alt = corr_df['correlation'].values
                avg_corr = np.mean(np.abs(correlations_alt))
            else:
                avg_corr = 0
    else:
        report.append("No se encontraron suficientes datos en la matriz de correlación.")
        report.append("(Muchas configuraciones DTE son únicas y no se repiten en las mismas fechas)")
        report.append("")
        # Usar datos del análisis por distancia
        if corr_df is not None and len(corr_df) > 0:
            correlations_alt = corr_df['correlation'].values
            avg_corr = np.mean(np.abs(correlations_alt))
        else:
            avg_corr = 0

    if corr_df is not None and len(corr_df) > 0:
        report.append("-" * 80)
        report.append("3. ANÁLISIS POR DISTANCIA DTE")
        report.append("-" * 80)

        # Correlación por distancia
        corr_by_dist = corr_df.groupby('dist_bin')['correlation'].agg(['mean', 'std', 'count'])
        report.append("Correlación promedio por distancia entre configuraciones:")
        for idx, row in corr_by_dist.iterrows():
            report.append(f"  {idx}: {row['mean']:.4f} ± {row['std']:.4f} (n={int(row['count'])})")
        report.append("")

    report.append("-" * 80)
    report.append("4. CONCLUSIONES")
    report.append("-" * 80)
    report.append("")

    # Conclusiones basadas en el análisis
    # Si ya no existe 'correlations' de la matriz, usar el análisis por distancia
    if 'correlations' in locals() and len(correlations) > 0:
        avg_corr = np.mean(np.abs(correlations))
    elif corr_df is not None and len(corr_df) > 0:
        avg_corr = np.mean(np.abs(corr_df['correlation'].values))
    else:
        avg_corr = 0

    if avg_corr > 0.7:
        conclusion = "ALTA CORRELACIÓN"
        detail = ("Los trades con diferentes configuraciones DTE1/DTE2 que comienzan en la misma fecha "
                 "muestran una ALTA correlación en sus PnL FWD. Esto significa que las diferentes "
                 "configuraciones NO tienen independencia significativa - tienden a moverse en la misma "
                 "dirección independientemente de la distancia temporal de las opciones.")
    elif avg_corr > 0.4:
        conclusion = "CORRELACIÓN MODERADA"
        detail = ("Los trades con diferentes configuraciones DTE1/DTE2 que comienzan en la misma fecha "
                 "muestran una correlación MODERADA en sus PnL FWD. Esto sugiere que aunque hay cierta "
                 "tendencia a moverse juntos, existe un grado significativo de independencia entre "
                 "las diferentes configuraciones.")
    else:
        conclusion = "CORRELACIÓN DÉBIL/INDEPENDENCIA"
        detail = ("Los trades con diferentes configuraciones DTE1/DTE2 que comienzan en la misma fecha "
                 "muestran correlación DÉBIL en sus PnL FWD. Esto indica que las diferentes configuraciones "
                 "tienen un alto grado de INDEPENDENCIA - los movimientos de PnL de una configuración "
                 "no predicen bien los movimientos de otra configuración.")

    report.append(f"CONCLUSIÓN GENERAL: {conclusion}")
    report.append("")
    report.append(detail)
    report.append("")

    # Ejemplo específico mencionado por el usuario
    report.append("EJEMPLO ESPECÍFICO (mencionado en la pregunta):")
    report.append("  Para responder si un trade 301/574 se mueve en la misma dirección que uno 740/1030")
    report.append("  cuando ambos parten del mismo día:")
    report.append("")

    # Buscar estos configs específicos
    if '301/574' in corr_matrix.index and '740/1030' in corr_matrix.columns:
        specific_corr = corr_matrix.loc['301/574', '740/1030']
        report.append(f"  Correlación 301/574 vs 740/1030: {specific_corr:.4f}")
        if abs(specific_corr) > 0.5:
            report.append("  → SÍ, tienden a moverse en la misma dirección (correlación significativa)")
        else:
            report.append("  → NO necesariamente, tienen bastante independencia")
    else:
        report.append("  (No se encontraron suficientes datos coincidentes para estas configuraciones específicas)")

    report.append("")
    report.append("-" * 80)
    report.append("INTERPRETACIÓN:")
    report.append("-" * 80)
    report.append("")
    report.append("Si la correlación es ALTA (>0.7):")
    report.append("  → Las diferentes configuraciones DTE NO ofrecen diversificación real")
    report.append("  → Todos los trades responden similarmente a las mismas condiciones de mercado")
    report.append("  → La diferencia de DTE NO les da independencia")
    report.append("")
    report.append("Si la correlación es BAJA (<0.3):")
    report.append("  → Las diferentes configuraciones DTE SÍ ofrecen diversificación")
    report.append("  → Cada configuración responde de manera más independiente")
    report.append("  → La diferencia de DTE SÍ les da independencia significativa")
    report.append("")
    report.append("=" * 80)

    report_text = "\n".join(report)

    # Guardar reporte
    with open('informe_correlacion_dte.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\nInforme guardado en: informe_correlacion_dte.txt")

    return report_text

def main():
    """Función principal"""
    print("=" * 80)
    print("ANÁLISIS DE CORRELACIÓN PnL FWD ENTRE CONFIGURACIONES DTE1/DTE2")
    print("=" * 80)

    # 1. Cargar datos
    df = load_data('/home/user/BatmanLT/VIX_combined_mediana.csv')

    # 2. Analizar overlap de fechas
    multi_config_dates, overlap_analysis = analyze_overlap_dates(df)

    # 3. Crear matriz de correlación (usando PnL_fwd_pts_50 como ejemplo principal)
    corr_matrix, pivot_df = create_correlation_matrix(df, 'PnL_fwd_pts_50')

    # 4. Visualizar matriz de correlación
    plot_correlation_heatmap(corr_matrix)

    # 5. Analizar ejemplos específicos
    # Encontrar algunas configuraciones comunes para el scatter plot
    top_configs = df['DTE1/DTE2'].value_counts().head(10).index.tolist()
    examples = [
        (top_configs[0], top_configs[1]),
        (top_configs[0], top_configs[2]),
        (top_configs[1], top_configs[3]),
        (top_configs[2], top_configs[4])
    ]

    # Intentar incluir el ejemplo específico del usuario si existe
    if '301/574' in top_configs and '740/1030' in top_configs:
        examples[0] = ('301/574', '740/1030')

    plot_scatter_examples(df, examples)

    # 6. Analizar por distancia DTE
    corr_df = analyze_by_dte_distance(df)

    # 7. Generar informe resumen
    generate_summary_report(df, corr_matrix, corr_df, multi_config_dates)

    print("\n" + "=" * 80)
    print("ANÁLISIS COMPLETADO")
    print("=" * 80)
    print("\nArchivos generados:")
    print("  1. correlacion_dte_heatmap.png - Matriz de correlación entre configuraciones")
    print("  2. correlacion_dte_scatter.png - Ejemplos específicos de correlación")
    print("  3. correlacion_vs_distancia_dte.png - Análisis de correlación vs distancia DTE")
    print("  4. informe_correlacion_dte.txt - Informe completo con conclusiones")
    print("\n")

if __name__ == "__main__":
    main()
