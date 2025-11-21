#!/usr/bin/env python3
"""
Análisis de Ventana Temporal para Independencia de Trades - Versión 2
Método simplificado y más robusto
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

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

def calculate_temporal_correlation_simple(df, pnl_column='PnL_fwd_pts_50', max_days=120):
    """
    Método simplificado: Para cada trade, encuentra otros trades en ventanas temporales
    y calcula la correlación de sus PnLs
    """
    print(f"\n=== ANÁLISIS DE CORRELACIÓN TEMPORAL ===")
    print(f"Analizando correlación entre trades con gaps de 0 a {max_days} días")
    print("Método: Comparación de PnL entre todos los pares de trades")

    # Limpiar datos
    df_clean = df[['dia', 'DTE1/DTE2', pnl_column]].dropna().copy()
    df_clean = df_clean.reset_index(drop=True)

    print(f"Trades con PnL válido: {len(df_clean)}")

    # Definir ventanas de días
    windows = [
        (0, 0, "0 días"),
        (1, 3, "1-3 días"),
        (4, 7, "4-7 días"),
        (8, 14, "8-14 días"),
        (15, 21, "15-21 días"),
        (22, 30, "22-30 días"),
        (31, 45, "31-45 días"),
        (46, 60, "46-60 días"),
        (61, 90, "61-90 días"),
        (91, 120, "91-120 días")
    ]

    results = []

    for min_days, max_days_window, label in windows:
        print(f"\nProcesando ventana: {label}...")

        pnl_pairs = []

        # Muestrear fechas base para eficiencia
        unique_dates = df_clean['dia'].unique()
        sample_size = min(150, len(unique_dates))
        sampled_dates = np.random.choice(unique_dates, size=sample_size, replace=False)

        for date1 in sampled_dates:
            # Convertir a pd.Timestamp si es necesario
            if not isinstance(date1, pd.Timestamp):
                date1 = pd.Timestamp(date1)

            # Calcular fechas objetivo
            target_min = date1 + timedelta(days=min_days)
            target_max = date1 + timedelta(days=max_days_window)

            # Trades en date1
            trades1 = df_clean[df_clean['dia'] == date1]

            # Trades en la ventana objetivo
            trades2 = df_clean[
                (df_clean['dia'] >= target_min) &
                (df_clean['dia'] <= target_max)
            ]

            if len(trades1) == 0 or len(trades2) == 0:
                continue

            # Muestrear pares para comparar
            for _, trade1 in trades1.sample(n=min(5, len(trades1))).iterrows():
                for _, trade2 in trades2.sample(n=min(5, len(trades2))).iterrows():
                    pnl1 = trade1[pnl_column]
                    pnl2 = trade2[pnl_column]

                    days_diff = (trade2['dia'] - trade1['dia']).days

                    if not np.isnan(pnl1) and not np.isnan(pnl2):
                        pnl_pairs.append({
                            'pnl1': pnl1,
                            'pnl2': pnl2,
                            'days_diff': days_diff
                        })

        if len(pnl_pairs) >= 30:  # Mínimo 30 pares
            pairs_df = pd.DataFrame(pnl_pairs)

            # Calcular correlación
            corr, p_value = pearsonr(pairs_df['pnl1'], pairs_df['pnl2'])

            # Calcular también el R²
            r_squared = corr ** 2

            results.append({
                'window_label': label,
                'days_min': min_days,
                'days_max': max_days_window,
                'days_mid': (min_days + max_days_window) / 2,
                'correlation': corr,
                'r_squared': r_squared,
                'p_value': p_value,
                'n_pairs': len(pnl_pairs)
            })

            print(f"  Correlación: {corr:.4f} (R²={r_squared:.4f}, n={len(pnl_pairs)} pares)")
        else:
            print(f"  Datos insuficientes ({len(pnl_pairs)} pares)")

    return pd.DataFrame(results)

def find_independence_day(results_df, thresholds=[0.5, 0.3, 0.2]):
    """
    Encuentra cuántos días se necesitan para diferentes niveles de independencia
    """
    print(f"\n=== IDENTIFICACIÓN DE UMBRALES DE INDEPENDENCIA ===")

    findings = {}

    for threshold in thresholds:
        below_threshold = results_df[abs(results_df['correlation']) < threshold]

        if len(below_threshold) > 0:
            first_window = below_threshold.iloc[0]
            days_range = f"{int(first_window['days_min'])}-{int(first_window['days_max'])}"
            findings[threshold] = {
                'days_min': first_window['days_min'],
                'days_max': first_window['days_max'],
                'days_range': days_range,
                'correlation': first_window['correlation']
            }
            print(f"  Correlación < {threshold}: {days_range} días (r={first_window['correlation']:.4f})")
        else:
            findings[threshold] = None
            print(f"  Correlación < {threshold}: NO alcanzado en el rango analizado")

    return findings

def plot_results(results_df, output_file='ventana_temporal_independencia.png'):
    """
    Crea visualizaciones del análisis
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Gráfico 1: Correlación vs Días (principal)
    ax1 = fig.add_subplot(gs[0:2, 0:2])

    x_pos = range(len(results_df))
    colors = ['red' if abs(c) > 0.7 else 'orange' if abs(c) > 0.5 else 'yellow' if abs(c) > 0.3 else 'green'
              for c in results_df['correlation']]

    bars = ax1.bar(x_pos, results_df['correlation'], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Líneas de referencia
    ax1.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Alta (0.7)')
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Moderada (0.5)')
    ax1.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Independencia (0.3)')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

    ax1.set_xlabel('Ventana Temporal', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Correlación de Pearson', fontsize=13, fontweight='bold')
    ax1.set_title('Decay de Correlación entre Trades por Gap Temporal',
                  fontsize=15, fontweight='bold', pad=20)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(results_df['window_label'], rotation=45, ha='right')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(-0.1, 1.1)

    # Añadir valores en las barras
    for i, (bar, val) in enumerate(zip(bars, results_df['correlation'])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Gráfico 2: R² (varianza explicada)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(x_pos, results_df['r_squared'], 'o-', linewidth=2, markersize=8, color='purple')
    ax2.fill_between(x_pos, 0, results_df['r_squared'], alpha=0.3, color='purple')
    ax2.set_ylabel('R² (Varianza Explicada)', fontsize=11, fontweight='bold')
    ax2.set_title('Poder Predictivo', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(results_df['window_label'], rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    # Gráfico 3: Número de pares analizados
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.bar(x_pos, results_df['n_pairs'], color='steelblue', alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Número de Pares', fontsize=11, fontweight='bold')
    ax3.set_title('Tamaño de Muestra', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(results_df['window_label'], rotation=45, ha='right', fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')

    # Gráfico 4: Tabla de recomendaciones
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    # Crear tabla con recomendaciones
    table_data = [['Gap Temporal', 'Correlación', 'R²', 'Nivel', 'Recomendación']]

    for _, row in results_df.iterrows():
        corr = row['correlation']
        r2 = row['r_squared']

        if abs(corr) > 0.7:
            nivel = 'ALTA ⚠️'
            recom = 'NO usar para diversificación'
            color = '#ffcccc'
        elif abs(corr) > 0.5:
            nivel = 'MODERADA'
            recom = 'Diversificación limitada'
            color = '#ffffcc'
        elif abs(corr) > 0.3:
            nivel = 'DÉBIL'
            recom = 'Diversificación aceptable'
            color = '#ccffcc'
        else:
            nivel = 'INDEPENDIENTE ✓'
            recom = 'Excelente para diversificación'
            color = '#99ff99'

        table_data.append([
            row['window_label'],
            f"{corr:.3f}",
            f"{r2:.3f}",
            nivel,
            recom
        ])

    table = ax4.table(cellText=table_data, cellLoc='center',
                     loc='center', bbox=[0.05, 0.1, 0.9, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Estilo
    for i in range(len(table_data)):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white', fontsize=11)
            else:
                if j == 3:  # Columna de nivel
                    if 'ALTA' in table_data[i][3]:
                        cell.set_facecolor('#ffcccc')
                    elif 'MODERADA' in table_data[i][3]:
                        cell.set_facecolor('#ffffcc')
                    elif 'DÉBIL' in table_data[i][3]:
                        cell.set_facecolor('#ccffcc')
                    else:
                        cell.set_facecolor('#99ff99')
                else:
                    cell.set_facecolor('#f8f8f8' if i % 2 == 0 else 'white')

    ax4.text(0.5, 0.95, 'Tabla de Recomendaciones por Ventana Temporal',
            ha='center', va='top', fontsize=14, fontweight='bold', transform=ax4.transAxes)

    plt.suptitle('Análisis de Ventana Temporal para Independencia de Trades',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nGráfico guardado en: {output_file}")
    plt.close()

def generate_report(results_df, findings):
    """
    Genera el informe final
    """
    report = []
    report.append("=" * 90)
    report.append("ANÁLISIS DE VENTANA TEMPORAL PARA INDEPENDENCIA DE TRADES")
    report.append("=" * 90)
    report.append("")
    report.append("PREGUNTA CENTRAL:")
    report.append("¿Cuántos días deben pasar entre iniciar un trade T1+0 y otro trade T2+0")
    report.append("para que sus resultados NO estén correlacionados?")
    report.append("")

    report.append("=" * 90)
    report.append("RESPUESTA EJECUTIVA")
    report.append("=" * 90)
    report.append("")

    # Encontrar la ventana de independencia
    independent_window = results_df[abs(results_df['correlation']) < 0.3]

    if len(independent_window) > 0:
        first_indep = independent_window.iloc[0]
        report.append(f"⏰ VENTANA DE INDEPENDENCIA: {first_indep['window_label']} días")
        report.append("")
        report.append(f"   → A partir de ~{int(first_indep['days_min'])} días, los trades muestran")
        report.append(f"      INDEPENDENCIA (correlación = {first_indep['correlation']:.4f})")
        report.append("")
        report.append(f"   → La correlación es prácticamente nula")
        report.append(f"   → Los resultados de ambos trades son independientes")
        report.append(f"   → Diversificación completa ✓")
    else:
        min_corr_window = results_df.loc[results_df['correlation'].abs().idxmin()]
        report.append(f"⚠️  INDEPENDENCIA COMPLETA NO ALCANZADA")
        report.append("")
        report.append(f"   → Correlación mínima: {min_corr_window['correlation']:.4f}")
        report.append(f"   → En ventana: {min_corr_window['window_label']} días")
        report.append(f"   → Se requieren gaps mayores a {int(min_corr_window['days_max'])} días")

    report.append("")
    report.append("-" * 90)
    report.append("EVOLUCIÓN DETALLADA DE LA CORRELACIÓN")
    report.append("-" * 90)
    report.append("")

    header = f"{'Ventana Temporal':<20} {'Correlación':<15} {'R²':<12} {'N Pares':<12} {'Interpretación':<25}"
    report.append(header)
    report.append("-" * 90)

    for _, row in results_df.iterrows():
        corr = row['correlation']
        r2 = row['r_squared']
        n = int(row['n_pairs'])

        if abs(corr) > 0.7:
            interp = "ALTA ⚠️"
        elif abs(corr) > 0.5:
            interp = "MODERADA"
        elif abs(corr) > 0.3:
            interp = "DÉBIL"
        else:
            interp = "INDEPENDIENTE ✓"

        report.append(f"{row['window_label']:<20} {corr:>7.4f}{'':<8} {r2:>6.4f}{'':<6} {n:<12} {interp:<25}")

    report.append("")
    report.append("-" * 90)
    report.append("INTERPRETACIÓN POR PERIODO")
    report.append("-" * 90)
    report.append("")

    # Día 0
    day0 = results_df[results_df['days_min'] == 0]
    if len(day0) > 0:
        report.append(f"1. MISMO DÍA (0 días):")
        report.append(f"   Correlación: {day0.iloc[0]['correlation']:.4f}")
        report.append(f"   → Trades iniciados el MISMO día están MUY correlacionados")
        report.append(f"   → Confirma hallazgos del análisis anterior")
        report.append(f"   → NO diversificación")
        report.append("")

    # Primera semana
    week1 = results_df[results_df['days_max'] <= 7]
    if len(week1) > 0:
        avg_corr = week1['correlation'].mean()
        report.append(f"2. PRIMERA SEMANA (0-7 días):")
        report.append(f"   Correlación promedio: {avg_corr:.4f}")
        if avg_corr > 0.7:
            report.append(f"   → Correlación MUY ALTA persiste")
            report.append(f"   → Insuficiente para diversificación")
        elif avg_corr > 0.5:
            report.append(f"   → Correlación ALTA todavía")
            report.append(f"   → Diversificación limitada")
        else:
            report.append(f"   → Correlación comienza a disminuir")
        report.append("")

    # 2-4 semanas
    month1 = results_df[(results_df['days_min'] >= 8) & (results_df['days_max'] <= 30)]
    if len(month1) > 0:
        avg_corr = month1['correlation'].mean()
        report.append(f"3. 2-4 SEMANAS (8-30 días):")
        report.append(f"   Correlación promedio: {avg_corr:.4f}")
        if avg_corr > 0.5:
            report.append(f"   → Correlación todavía MODERADA")
            report.append(f"   → Diversificación parcial")
        elif avg_corr > 0.3:
            report.append(f"   → Correlación DÉBIL")
            report.append(f"   → Diversificación buena")
        else:
            report.append(f"   → INDEPENDENCIA alcanzada ✓")
            report.append(f"   → Diversificación completa")
        report.append("")

    # Más de 30 días
    month2 = results_df[results_df['days_min'] > 30]
    if len(month2) > 0:
        avg_corr = month2['correlation'].mean()
        report.append(f"4. MÁS DE 30 DÍAS:")
        report.append(f"   Correlación promedio: {avg_corr:.4f}")
        if avg_corr < 0.3:
            report.append(f"   → INDEPENDENCIA completa ✓")
            report.append(f"   → Máxima diversificación")
        elif avg_corr < 0.5:
            report.append(f"   → Correlación BAJA")
            report.append(f"   → Excelente diversificación")
        else:
            report.append(f"   → Correlación residual presente")
        report.append("")

    report.append("-" * 90)
    report.append("RECOMENDACIONES ESTRATÉGICAS")
    report.append("-" * 90)
    report.append("")

    # Encontrar umbrales clave
    threshold_30 = results_df[abs(results_df['correlation']) < 0.3]
    threshold_50 = results_df[abs(results_df['correlation']) < 0.5]

    if len(threshold_30) > 0:
        days_30 = int(threshold_30.iloc[0]['days_min'])
        report.append(f"🎯 ESTRATEGIA CONSERVADORA (Independencia completa):")
        report.append(f"   → Espaciar trades {days_30}+ días")
        report.append(f"   → Correlación < 0.3")
        report.append(f"   → Máxima diversificación")
        report.append(f"   → ~{int(365/days_30)} trades independientes por año")
        report.append("")

    if len(threshold_50) > 0:
        days_50 = int(threshold_50.iloc[0]['days_min'])
        report.append(f"📊 ESTRATEGIA MODERADA (Baja correlación):")
        report.append(f"   → Espaciar trades {days_50}+ días")
        report.append(f"   → Correlación < 0.5")
        report.append(f"   → Buena diversificación")
        report.append(f"   → ~{int(365/days_50)} trades por año")
        report.append("")

    report.append(f"⚡ ESTRATEGIA AGRESIVA (Mayor frecuencia):")
    report.append(f"   → Espaciar trades 7-14 días")
    report.append(f"   → Correlación moderada/alta persiste")
    report.append(f"   → Diversificación limitada")
    report.append(f"   → Mayor exposición a condiciones de mercado")
    report.append("")

    report.append("-" * 90)
    report.append("CONCLUSIONES FINALES")
    report.append("-" * 90)
    report.append("")

    if len(threshold_30) > 0:
        days_rec = int(threshold_30.iloc[0]['days_min'])
        report.append(f"✓ Para INDEPENDENCIA COMPLETA entre trades:")
        report.append(f"  → Espaciar entradas al menos {days_rec} días")
        report.append(f"  → Esto garantiza que los resultados no están correlacionados")
        report.append(f"  → Cada trade tiene su propio 'ciclo de mercado'")
    else:
        report.append(f"✓ Para REDUCIR significativamente la correlación:")
        report.append(f"  → Espaciar entradas al menos 30-45 días")
        report.append(f"  → Para independencia completa, considerar 60+ días")

    report.append("")
    report.append("REGLA DE ORO:")
    report.append("  'No pongas todos tus huevos en la misma semana'")
    report.append("")
    report.append("  → Distribuir entradas a lo largo del tiempo es CLAVE")
    report.append("  → La diversificación temporal es TAN importante como la diversificación de DTE")
    report.append("  → El 'cuándo' importa más que el 'qué' configuración DTE uses")
    report.append("")
    report.append("=" * 90)

    report_text = "\n".join(report)

    with open('INFORME_VENTANA_TEMPORAL_INDEPENDENCIA.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\nInforme guardado en: INFORME_VENTANA_TEMPORAL_INDEPENDENCIA.txt")

def main():
    """Función principal"""
    print("=" * 90)
    print("ANÁLISIS DE VENTANA TEMPORAL PARA INDEPENDENCIA DE TRADES")
    print("=" * 90)

    # Cargar datos
    df = load_data('/home/user/BatmanLT/VIX_combined_mediana.csv')

    # Análisis de correlación temporal
    results_df = calculate_temporal_correlation_simple(df, max_days=120)

    if len(results_df) == 0:
        print("\n⚠️  No se pudieron calcular correlaciones")
        return

    # Identificar umbrales
    findings = find_independence_day(results_df)

    # Visualizar
    plot_results(results_df)

    # Generar informe
    generate_report(results_df, findings)

    print("\n" + "=" * 90)
    print("ANÁLISIS COMPLETADO")
    print("=" * 90)
    print("\nArchivos generados:")
    print("  1. ventana_temporal_independencia.png - Visualizaciones completas")
    print("  2. INFORME_VENTANA_TEMPORAL_INDEPENDENCIA.txt - Informe detallado")
    print("\n")

if __name__ == "__main__":
    main()
