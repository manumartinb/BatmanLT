#!/usr/bin/env python3
"""
Análisis de Ventana Temporal para Independencia de Trades

Este script responde a la pregunta: ¿Cuántos días deben pasar entre iniciar
un trade T1+0 y otro trade T2+0 para que sus resultados NO estén correlacionados?

El objetivo es determinar la "ventana de independencia temporal" para
estrategias de diversificación.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from datetime import timedelta
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
    df = df.sort_values('dia').reset_index(drop=True)
    print(f"Total de trades: {len(df)}")
    print(f"Rango de fechas: {df['dia'].min()} a {df['dia'].max()}")
    return df

def calculate_temporal_correlations(df, pnl_column='PnL_fwd_pts_50',
                                   max_days=180, sample_size=1000):
    """
    Calcula la correlación entre trades según el gap temporal entre sus fechas de inicio

    Args:
        df: DataFrame con los datos
        pnl_column: Columna de PnL a analizar
        max_days: Máximo gap temporal a analizar (días)
        sample_size: Número de pares a muestrear por bin (para eficiencia)
    """
    print(f"\n=== ANÁLISIS DE CORRELACIÓN TEMPORAL ===")
    print(f"Analizando gaps de 0 a {max_days} días")

    # Preparar datos
    df_clean = df[['dia', 'DTE1/DTE2', pnl_column]].dropna()

    results = []

    # Crear bins de días
    day_bins = [0, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 150, 180]

    print("\nCalculando correlaciones por gap temporal...")
    print("(Esto puede tomar varios minutos)")

    for i in range(len(day_bins) - 1):
        min_days = day_bins[i]
        max_days_bin = day_bins[i + 1]

        print(f"\nProcesando gap: {min_days} a {max_days_bin} días...")

        correlations_in_bin = []
        pairs_analyzed = 0

        # Muestreo de fechas para eficiencia
        sample_dates = df_clean['dia'].drop_duplicates().sample(
            n=min(200, len(df_clean['dia'].unique())),
            random_state=42
        )

        for date1 in sample_dates:
            # Encontrar fechas en el rango objetivo
            min_date2 = date1 + timedelta(days=min_days)
            max_date2 = date1 + timedelta(days=max_days_bin)

            # Trades en la fecha1
            trades1 = df_clean[df_clean['dia'] == date1]

            # Trades en el rango de fechas objetivo
            trades2 = df_clean[
                (df_clean['dia'] >= min_date2) &
                (df_clean['dia'] < max_date2)
            ]

            if len(trades1) == 0 or len(trades2) == 0:
                continue

            # Muestrear configuraciones para comparar
            configs_to_compare = min(5, len(trades1))
            sample_configs = trades1['DTE1/DTE2'].sample(
                n=configs_to_compare,
                random_state=42
            ).tolist() if len(trades1) >= configs_to_compare else trades1['DTE1/DTE2'].tolist()

            for config in sample_configs:
                # PnL de la configuración en date1
                pnl1 = trades1[trades1['DTE1/DTE2'] == config][pnl_column].values

                if len(pnl1) == 0:
                    continue

                # PnL de la misma configuración en fechas cercanas
                pnl2_same_config = trades2[trades2['DTE1/DTE2'] == config][pnl_column].values

                if len(pnl2_same_config) > 0:
                    # Calcular correlación con la misma configuración
                    for p2 in pnl2_same_config[:5]:  # Máximo 5 comparaciones
                        if not np.isnan(pnl1[0]) and not np.isnan(p2):
                            # Aquí usamos la diferencia de días real
                            actual_date2 = trades2[trades2[pnl_column] == p2]['dia'].values[0]
                            if isinstance(actual_date2, np.datetime64):
                                actual_date2 = pd.Timestamp(actual_date2)
                            days_diff = (actual_date2 - date1).days

                            correlations_in_bin.append({
                                'days_gap': days_diff,
                                'pnl1': pnl1[0],
                                'pnl2': p2,
                                'config1': config,
                                'config2': config,
                                'same_config': True
                            })
                            pairs_analyzed += 1

                # También comparar con otras configuraciones
                other_configs = trades2[trades2['DTE1/DTE2'] != config]['DTE1/DTE2'].unique()
                if len(other_configs) > 0:
                    sample_other = np.random.choice(other_configs, size=min(3, len(other_configs)), replace=False)
                    for other_config in sample_other:
                        pnl2_other = trades2[trades2['DTE1/DTE2'] == other_config][pnl_column].values
                        if len(pnl2_other) > 0:
                            for p2 in pnl2_other[:3]:
                                if not np.isnan(pnl1[0]) and not np.isnan(p2):
                                    actual_date2 = trades2[trades2[pnl_column] == p2]['dia'].values[0]
                                    if isinstance(actual_date2, np.datetime64):
                                        actual_date2 = pd.Timestamp(actual_date2)
                                    days_diff = (actual_date2 - date1).days

                                    correlations_in_bin.append({
                                        'days_gap': days_diff,
                                        'pnl1': pnl1[0],
                                        'pnl2': p2,
                                        'config1': config,
                                        'config2': other_config,
                                        'same_config': False
                                    })
                                    pairs_analyzed += 1

        if len(correlations_in_bin) > 0:
            bin_df = pd.DataFrame(correlations_in_bin)

            # Calcular correlación para este bin
            if len(bin_df) >= 10:  # Mínimo 10 pares
                corr, p_value = pearsonr(bin_df['pnl1'], bin_df['pnl2'])

                results.append({
                    'days_gap_min': min_days,
                    'days_gap_max': max_days_bin,
                    'days_gap_mid': (min_days + max_days_bin) / 2,
                    'correlation': corr,
                    'p_value': p_value,
                    'n_pairs': len(bin_df),
                    'same_config_pct': (bin_df['same_config'].sum() / len(bin_df)) * 100
                })

                print(f"  Gap {min_days}-{max_days_bin} días: r={corr:.4f} (n={len(bin_df)} pares)")

    results_df = pd.DataFrame(results)
    return results_df

def calculate_rolling_correlation_by_day(df, pnl_column='PnL_fwd_pts_50',
                                         max_days=90, top_configs=30):
    """
    Calcula correlación día por día usando configuraciones comunes
    Método más directo y eficiente
    """
    print(f"\n=== ANÁLISIS DÍA POR DÍA (Método alternativo) ===")

    # Usar solo las configuraciones más comunes
    top_config_list = df['DTE1/DTE2'].value_counts().head(top_configs).index.tolist()
    df_top = df[df['DTE1/DTE2'].isin(top_config_list)].copy()

    print(f"Usando {len(top_config_list)} configuraciones más comunes")
    print(f"Trades disponibles: {len(df_top)}")

    results = []

    # Para cada gap de días
    for gap in range(0, max_days + 1):
        if gap % 10 == 0:
            print(f"Procesando gap de {gap} días...")

        correlations = []

        # Muestrear fechas para eficiencia
        sample_dates = df_top['dia'].drop_duplicates().sample(
            n=min(100, len(df_top['dia'].unique())),
            random_state=42
        ).tolist()

        for date1 in sample_dates:
            date2 = date1 + timedelta(days=gap)

            # Trades en date1 y date2
            trades_d1 = df_top[df_top['dia'] == date1]
            trades_d2 = df_top[df_top['dia'] == date2]

            if len(trades_d1) == 0 or len(trades_d2) == 0:
                continue

            # Encontrar configuraciones comunes
            common_configs = set(trades_d1['DTE1/DTE2']) & set(trades_d2['DTE1/DTE2'])

            if len(common_configs) >= 3:  # Al menos 3 configuraciones en común
                for config in common_configs:
                    pnl1 = trades_d1[trades_d1['DTE1/DTE2'] == config][pnl_column].values
                    pnl2 = trades_d2[trades_d2['DTE1/DTE2'] == config][pnl_column].values

                    if len(pnl1) > 0 and len(pnl2) > 0:
                        correlations.append({
                            'pnl1': pnl1[0],
                            'pnl2': pnl2[0]
                        })

        if len(correlations) >= 10:  # Mínimo 10 pares
            corr_df = pd.DataFrame(correlations)
            corr, p_value = pearsonr(corr_df['pnl1'], corr_df['pnl2'])

            results.append({
                'days_gap': gap,
                'correlation': corr,
                'p_value': p_value,
                'n_pairs': len(correlations)
            })

    results_df = pd.DataFrame(results)
    return results_df

def find_independence_threshold(results_df, threshold=0.3):
    """
    Encuentra el número de días donde la correlación cae por debajo del umbral
    """
    print(f"\n=== BÚSQUEDA DE UMBRAL DE INDEPENDENCIA ===")
    print(f"Umbral de correlación: {threshold}")

    # Encontrar el primer punto donde la correlación cae por debajo del umbral
    below_threshold = results_df[abs(results_df['correlation']) < threshold]

    if len(below_threshold) > 0:
        first_independent = below_threshold.iloc[0]
        days_to_independence = first_independent['days_gap']

        print(f"\n✓ INDEPENDENCIA ALCANZADA:")
        print(f"  A partir de ~{days_to_independence:.0f} días")
        print(f"  Correlación en ese punto: {first_independent['correlation']:.4f}")
        print(f"  Basado en {first_independent['n_pairs']:.0f} pares de trades")

        return days_to_independence, first_independent['correlation']
    else:
        print(f"\n✗ NO SE ALCANZÓ INDEPENDENCIA en el rango analizado")
        print(f"  Correlación mínima observada: {results_df['correlation'].min():.4f}")
        print(f"  En gap de: {results_df.loc[results_df['correlation'].idxmin(), 'days_gap']:.0f} días")

        return None, results_df['correlation'].min()

def plot_correlation_decay(results_df, output_file='correlacion_decay_temporal.png'):
    """
    Visualiza el decay de la correlación con el tiempo
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Gráfico 1: Correlación vs Días (línea)
    ax1 = axes[0, 0]
    ax1.plot(results_df['days_gap'], results_df['correlation'],
             'o-', linewidth=2, markersize=6, color='steelblue')
    ax1.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Alta correlación (0.7)')
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderada (0.5)')
    ax1.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='Independencia (0.3)')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    ax1.set_xlabel('Días entre inicio de trades', fontsize=12)
    ax1.set_ylabel('Correlación de Pearson', fontsize=12)
    ax1.set_title('Decay de Correlación entre Trades por Gap Temporal', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim(-0.1, 1.05)

    # Gráfico 2: Correlación absoluta vs Días
    ax2 = axes[0, 1]
    ax2.plot(results_df['days_gap'], abs(results_df['correlation']),
             'o-', linewidth=2, markersize=6, color='crimson')
    ax2.axhline(y=0.3, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Umbral independencia')
    ax2.fill_between(results_df['days_gap'], 0, 0.3, alpha=0.2, color='green', label='Zona de independencia')
    ax2.set_xlabel('Días entre inicio de trades', fontsize=12)
    ax2.set_ylabel('Correlación Absoluta |r|', fontsize=12)
    ax2.set_title('Correlación Absoluta vs Gap Temporal', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_ylim(-0.05, 1.05)

    # Gráfico 3: Scatter con tendencia
    ax3 = axes[1, 0]
    scatter = ax3.scatter(results_df['days_gap'], results_df['correlation'],
                         c=results_df['n_pairs'], cmap='viridis',
                         s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

    # Línea de tendencia (exponencial decay fit)
    try:
        from scipy.optimize import curve_fit

        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c

        x_data = results_df['days_gap'].values
        y_data = results_df['correlation'].values

        # Fit
        popt, _ = curve_fit(exp_decay, x_data, y_data,
                           p0=[1, 0.01, 0.2], maxfev=10000)

        x_fit = np.linspace(x_data.min(), x_data.max(), 100)
        y_fit = exp_decay(x_fit, *popt)
        ax3.plot(x_fit, y_fit, 'r--', linewidth=2, alpha=0.8,
                label=f'Ajuste exponencial: {popt[0]:.2f}·e^(-{popt[1]:.4f}·x) + {popt[2]:.2f}')
    except:
        pass

    ax3.axhline(y=0.3, color='green', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Días entre inicio de trades', fontsize=12)
    ax3.set_ylabel('Correlación de Pearson', fontsize=12)
    ax3.set_title('Decay de Correlación con Ajuste Exponencial', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)
    plt.colorbar(scatter, ax=ax3, label='Número de pares')

    # Gráfico 4: Tabla de recomendaciones
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Crear tabla de recomendaciones
    recommendations = []

    for threshold, label, color in [(0.7, 'Alta correlación', '#ff4444'),
                                     (0.5, 'Moderada', '#ffaa44'),
                                     (0.3, 'Independencia', '#44ff44')]:
        below = results_df[abs(results_df['correlation']) < threshold]
        if len(below) > 0:
            days = below.iloc[0]['days_gap']
            recommendations.append([label, f"≥ {days:.0f} días", color])
        else:
            recommendations.append([label, "No alcanzado", '#cccccc'])

    table_data = [['Objetivo', 'Días requeridos', '']]
    table_data.extend(recommendations)

    table = ax4.table(cellText=table_data, cellLoc='left',
                     loc='center', bbox=[0.1, 0.3, 0.8, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Estilo de la tabla
    for i in range(len(table_data)):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
            else:
                if j == 2:
                    cell.set_facecolor(table_data[i][2])
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

    ax4.text(0.5, 0.85, 'Recomendaciones de Espaciado Temporal',
            ha='center', va='top', fontsize=14, fontweight='bold', transform=ax4.transAxes)

    ax4.text(0.5, 0.15,
            'Basado en correlación de PnL FWD entre trades\nque comienzan con diferentes gaps temporales',
            ha='center', va='top', fontsize=9, style='italic',
            transform=ax4.transAxes, color='gray')

    plt.suptitle('Análisis de Ventana Temporal para Independencia de Trades',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nGráfico guardado en: {output_file}")
    plt.close()

def generate_final_report(results_df, days_to_independence, min_correlation):
    """
    Genera el informe final con recomendaciones
    """
    report = []
    report.append("=" * 80)
    report.append("ANÁLISIS DE VENTANA TEMPORAL PARA INDEPENDENCIA DE TRADES")
    report.append("=" * 80)
    report.append("")
    report.append("PREGUNTA: ¿Cuántos días deben pasar entre iniciar un trade T1+0 y otro")
    report.append("trade T2+0 para que sus resultados NO estén correlacionados?")
    report.append("")

    report.append("-" * 80)
    report.append("RESPUESTA DIRECTA")
    report.append("-" * 80)
    report.append("")

    if days_to_independence is not None:
        report.append(f"⏰ RECOMENDACIÓN: Esperar al menos {days_to_independence:.0f} días")
        report.append("")
        report.append(f"A partir de ~{days_to_independence:.0f} días de diferencia entre las fechas de inicio,")
        report.append("los trades muestran independencia (correlación < 0.3).")
        report.append("")
        report.append(f"Correlación a {days_to_independence:.0f} días: {min_correlation:.4f}")
    else:
        report.append("⚠️  NO SE ALCANZÓ INDEPENDENCIA COMPLETA en el rango analizado")
        report.append("")
        report.append(f"Correlación mínima observada: {min_correlation:.4f}")
        report.append(f"En gap de: {results_df.loc[results_df['correlation'].idxmin(), 'days_gap']:.0f} días")
        report.append("")
        report.append("Esto sugiere que la correlación persiste incluso con grandes gaps temporales,")
        report.append("aunque se reduce significativamente.")

    report.append("")
    report.append("-" * 80)
    report.append("EVOLUCIÓN DE LA CORRELACIÓN POR GAP TEMPORAL")
    report.append("-" * 80)
    report.append("")

    # Tabla de correlaciones clave
    key_gaps = [0, 1, 3, 7, 14, 21, 30, 45, 60, 90]
    report.append(f"{'Días Gap':<12} {'Correlación':<15} {'N Pares':<12} {'Interpretación':<30}")
    report.append("-" * 80)

    for gap in key_gaps:
        row = results_df[results_df['days_gap'] == gap]
        if len(row) > 0:
            corr = row.iloc[0]['correlation']
            n_pairs = row.iloc[0]['n_pairs']

            if abs(corr) > 0.7:
                interp = "Alta correlación"
            elif abs(corr) > 0.5:
                interp = "Moderada"
            elif abs(corr) > 0.3:
                interp = "Débil"
            else:
                interp = "Independencia ✓"

            report.append(f"{gap:<12} {corr:>7.4f}{'':<8} {int(n_pairs):<12} {interp:<30}")

    report.append("")
    report.append("-" * 80)
    report.append("INTERPRETACIÓN Y RECOMENDACIONES")
    report.append("-" * 80)
    report.append("")

    # Correlación en día 0
    day0_corr = results_df[results_df['days_gap'] == 0]['correlation'].values
    if len(day0_corr) > 0:
        report.append(f"1. DÍA 0 (mismo día): Correlación = {day0_corr[0]:.4f}")
        report.append("   → Trades iniciados el mismo día están ALTAMENTE correlacionados")
        report.append("   → Consistente con el análisis anterior")
        report.append("")

    # Primeros días
    week1_corr = results_df[results_df['days_gap'] <= 7]['correlation'].mean()
    report.append(f"2. PRIMERA SEMANA (0-7 días): Correlación promedio = {week1_corr:.4f}")
    report.append("   → La correlación sigue siendo ALTA")
    report.append("   → NO es suficiente espaciado para diversificación efectiva")
    report.append("")

    # 2-4 semanas
    month1_corr = results_df[(results_df['days_gap'] > 7) & (results_df['days_gap'] <= 30)]['correlation'].mean()
    report.append(f"3. 2-4 SEMANAS (8-30 días): Correlación promedio = {month1_corr:.4f}")
    if month1_corr > 0.5:
        report.append("   → La correlación comienza a disminuir pero sigue siendo MODERADA")
        report.append("   → Diversificación parcial")
    else:
        report.append("   → La correlación ha disminuido significativamente")
        report.append("   → Diversificación efectiva")
    report.append("")

    # Más de 30 días
    month2_corr = results_df[results_df['days_gap'] > 30]['correlation'].mean()
    report.append(f"4. MÁS DE 30 DÍAS: Correlación promedio = {month2_corr:.4f}")
    if month2_corr < 0.3:
        report.append("   → INDEPENDENCIA alcanzada ✓")
        report.append("   → Diversificación completa")
    elif month2_corr < 0.5:
        report.append("   → Correlación DÉBIL")
        report.append("   → Buena diversificación")
    else:
        report.append("   → Todavía existe correlación residual")
        report.append("   → Considerar gaps aún mayores")
    report.append("")

    report.append("-" * 80)
    report.append("ESTRATEGIAS DE ENTRADA RECOMENDADAS")
    report.append("-" * 80)
    report.append("")

    if days_to_independence is not None and days_to_independence <= 45:
        report.append(f"📅 ESTRATEGIA CONSERVADORA: Espaciar trades {int(days_to_independence)}+ días")
        report.append(f"   - Ejemplo: 1 trade cada {int(days_to_independence)} días")
        report.append("   - Garantiza independencia completa entre trades")
        report.append("   - Máxima diversificación temporal")
        report.append("")

        moderate_days = int(days_to_independence * 0.7)
        report.append(f"📅 ESTRATEGIA MODERADA: Espaciar trades {moderate_days}+ días")
        report.append(f"   - Ejemplo: 1 trade cada {moderate_days} días")
        report.append("   - Balance entre frecuencia y diversificación")
        report.append("   - Correlación baja pero no nula")
        report.append("")

        aggressive_days = max(7, int(days_to_independence * 0.4))
        report.append(f"📅 ESTRATEGIA AGRESIVA: Espaciar trades {aggressive_days}+ días")
        report.append(f"   - Ejemplo: 1 trade cada {aggressive_days} días")
        report.append("   - Mayor frecuencia pero menor diversificación")
        report.append("   - Todavía existe correlación moderada")
    else:
        report.append("📅 RECOMENDACIÓN GENERAL:")
        report.append("   - Espaciar trades al menos 30-45 días para reducir correlación")
        report.append("   - Considerar ventanas de 60+ días para independencia más robusta")

    report.append("")
    report.append("-" * 80)
    report.append("EJEMPLOS PRÁCTICOS")
    report.append("-" * 80)
    report.append("")

    if days_to_independence is not None:
        trades_per_year = int(365 / days_to_independence)
        report.append(f"Si entras 1 trade cada {int(days_to_independence)} días:")
        report.append(f"  → Aprox. {trades_per_year} trades independientes por año")
        report.append(f"  → Cada trade tiene su propio 'espacio de mercado'")
        report.append("")

        report.append(f"Si entras 1 trade cada 7 días (semanal):")
        week_corr = results_df[results_df['days_gap'] == 7]['correlation'].values
        if len(week_corr) > 0:
            report.append(f"  → Correlación entre trades consecutivos: ~{week_corr[0]:.2f}")
            if week_corr[0] > 0.5:
                report.append(f"  → ⚠️  ALTA correlación - baja diversificación")
            else:
                report.append(f"  → ✓ Correlación moderada/baja - diversificación aceptable")
        report.append(f"  → Aprox. 52 trades por año")

    report.append("")
    report.append("-" * 80)
    report.append("CONCLUSIÓN FINAL")
    report.append("-" * 80)
    report.append("")

    if days_to_independence is not None and days_to_independence <= 60:
        report.append(f"Para lograr INDEPENDENCIA entre trades (correlación < 0.3):")
        report.append(f"  → Espaciar entradas al menos {int(days_to_independence)} días")
        report.append("")
        report.append("Para REDUCIR correlación de manera significativa:")
        report.append(f"  → Al menos {max(14, int(days_to_independence * 0.5))} días")
    else:
        report.append("La correlación disminuye gradualmente con el tiempo:")
        report.append("  → Espaciar al menos 30 días para correlación moderada")
        report.append("  → Espaciar 60+ días para máxima independencia")

    report.append("")
    report.append("REGLA PRÁCTICA: No inicies todos tus trades en un corto periodo.")
    report.append("Distribuye las entradas a lo largo del tiempo para verdadera diversificación.")
    report.append("")
    report.append("=" * 80)

    report_text = "\n".join(report)

    # Guardar reporte
    with open('INFORME_VENTANA_TEMPORAL_INDEPENDENCIA.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\nInforme guardado en: INFORME_VENTANA_TEMPORAL_INDEPENDENCIA.txt")

    return report_text

def main():
    """Función principal"""
    print("=" * 80)
    print("ANÁLISIS DE VENTANA TEMPORAL PARA INDEPENDENCIA DE TRADES")
    print("=" * 80)

    # 1. Cargar datos
    df = load_data('/home/user/BatmanLT/VIX_combined_mediana.csv')

    # 2. Calcular correlaciones temporales (método día por día)
    print("\nUsando método optimizado de análisis día por día...")
    results_df = calculate_rolling_correlation_by_day(df, max_days=90, top_configs=30)

    if len(results_df) == 0:
        print("\n⚠️  No se pudieron calcular suficientes correlaciones.")
        print("Esto puede deberse a:")
        print("  - Configuraciones muy dispersas temporalmente")
        print("  - Falta de overlap entre fechas")
        return

    # 3. Encontrar umbral de independencia
    days_to_independence, min_correlation = find_independence_threshold(results_df, threshold=0.3)

    # 4. Visualizar decay de correlación
    plot_correlation_decay(results_df)

    # 5. Generar informe final
    generate_final_report(results_df, days_to_independence, min_correlation)

    print("\n" + "=" * 80)
    print("ANÁLISIS COMPLETADO")
    print("=" * 80)
    print("\nArchivos generados:")
    print("  1. correlacion_decay_temporal.png - Visualización del decay de correlación")
    print("  2. INFORME_VENTANA_TEMPORAL_INDEPENDENCIA.txt - Informe completo")
    print("\n")

if __name__ == "__main__":
    main()
