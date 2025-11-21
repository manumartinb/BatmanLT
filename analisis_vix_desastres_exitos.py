"""
Análisis de cambios del VIX en operaciones DESASTRE vs ÉXITO
Basado en PnL FWD ventana 25

Autor: Análisis solicitado
Fecha: 2025-11-21
"""

import pandas as pd
import numpy as np

def analizar_cambios_vix():
    """
    Analiza los cambios del VIX en operaciones desastre (<-20%) y éxito (>20%)
    según el PnL FWD ventana 25
    """
    # Leer el archivo CSV
    df = pd.read_csv('VIX_combined_mediana.csv')

    # Convertir las fechas a datetime
    df['dia'] = pd.to_datetime(df['dia'])
    df['dia_fwd_25'] = pd.to_datetime(df['dia_fwd_25'])

    # Crear un diccionario de VIX por fecha
    vix_by_date = df[['dia', 'VIX']].drop_duplicates('dia').set_index('dia')['VIX'].to_dict()

    # Añadir el VIX en la fecha forward
    df['VIX_fwd_25'] = df['dia_fwd_25'].map(vix_by_date)

    # Calcular el cambio del VIX
    df['VIX_change_pts'] = df['VIX_fwd_25'] - df['VIX']
    df['VIX_change_pct'] = ((df['VIX_fwd_25'] - df['VIX']) / df['VIX']) * 100

    # Filtrar operaciones "desastre" (PnL_fwd_pct_25 < -20)
    desastres = df[df['PnL_fwd_pct_25'] < -20].copy()
    desastres_con_vix = desastres.dropna(subset=['VIX_change_pts', 'VIX_change_pct'])

    # Filtrar operaciones "éxito" (PnL_fwd_pct_25 > 20)
    exitos = df[df['PnL_fwd_pct_25'] > 20].copy()
    exitos_con_vix = exitos.dropna(subset=['VIX_change_pts', 'VIX_change_pct'])

    # Crear reporte
    print("=" * 100)
    print(" " * 20 + "ANÁLISIS DE CAMBIOS DEL VIX EN OPERACIONES 'DESASTRE' Y 'ÉXITO'")
    print("=" * 100)

    print(f"\n{'RESUMEN GENERAL':^100}")
    print("-" * 100)
    print(f"  Total de operaciones en el dataset: {len(df):,}")
    print(f"  Operaciones 'DESASTRE' (PnL FWD 25 < -20%): {len(desastres):,} ({len(desastres)/len(df)*100:.1f}%)")
    print(f"  Operaciones 'ÉXITO' (PnL FWD 25 > 20%): {len(exitos):,} ({len(exitos)/len(df)*100:.1f}%)")
    print(f"  Operaciones con datos VIX disponibles:")
    print(f"    - Desastres: {len(desastres_con_vix):,} ({len(desastres_con_vix)/len(desastres)*100:.1f}%)")
    print(f"    - Éxitos: {len(exitos_con_vix):,} ({len(exitos_con_vix)/len(exitos)*100:.1f}%)")

    # DESASTRES
    print(f"\n{'🔴 OPERACIONES DESASTRE (PnL FWD 25 < -20%)':^100}")
    print("=" * 100)

    print(f"\n  Cambio del VIX - Puntos Nominales:")
    print(f"    {'Métrica':<20} {'Valor':>15}")
    print(f"    {'-'*35}")
    print(f"    {'Promedio':<20} {desastres_con_vix['VIX_change_pts'].mean():>14.2f} pts")
    print(f"    {'Mediana':<20} {desastres_con_vix['VIX_change_pts'].median():>14.2f} pts")
    print(f"    {'Desv. Estándar':<20} {desastres_con_vix['VIX_change_pts'].std():>14.2f} pts")
    print(f"    {'Mínimo':<20} {desastres_con_vix['VIX_change_pts'].min():>14.2f} pts")
    print(f"    {'Máximo':<20} {desastres_con_vix['VIX_change_pts'].max():>14.2f} pts")

    print(f"\n  Cambio del VIX - Porcentaje:")
    print(f"    {'Métrica':<20} {'Valor':>15}")
    print(f"    {'-'*35}")
    print(f"    {'Promedio':<20} {desastres_con_vix['VIX_change_pct'].mean():>14.2f}%")
    print(f"    {'Mediana':<20} {desastres_con_vix['VIX_change_pct'].median():>14.2f}%")
    print(f"    {'Desv. Estándar':<20} {desastres_con_vix['VIX_change_pct'].std():>14.2f}%")
    print(f"    {'Mínimo':<20} {desastres_con_vix['VIX_change_pct'].min():>14.2f}%")
    print(f"    {'Máximo':<20} {desastres_con_vix['VIX_change_pct'].max():>14.2f}%")

    print(f"\n  Niveles de VIX:")
    print(f"    {'Métrica':<20} {'Valor':>15}")
    print(f"    {'-'*35}")
    print(f"    {'VIX Inicial Prom.':<20} {desastres_con_vix['VIX'].mean():>14.2f}")
    print(f"    {'VIX Final Prom.':<20} {desastres_con_vix['VIX_fwd_25'].mean():>14.2f}")
    print(f"    {'Cambio Neto':<20} {desastres_con_vix['VIX_fwd_25'].mean() - desastres_con_vix['VIX'].mean():>14.2f} pts")

    # ÉXITOS
    print(f"\n{'🟢 OPERACIONES ÉXITO (PnL FWD 25 > 20%)':^100}")
    print("=" * 100)

    print(f"\n  Cambio del VIX - Puntos Nominales:")
    print(f"    {'Métrica':<20} {'Valor':>15}")
    print(f"    {'-'*35}")
    print(f"    {'Promedio':<20} {exitos_con_vix['VIX_change_pts'].mean():>14.2f} pts")
    print(f"    {'Mediana':<20} {exitos_con_vix['VIX_change_pts'].median():>14.2f} pts")
    print(f"    {'Desv. Estándar':<20} {exitos_con_vix['VIX_change_pts'].std():>14.2f} pts")
    print(f"    {'Mínimo':<20} {exitos_con_vix['VIX_change_pts'].min():>14.2f} pts")
    print(f"    {'Máximo':<20} {exitos_con_vix['VIX_change_pts'].max():>14.2f} pts")

    print(f"\n  Cambio del VIX - Porcentaje:")
    print(f"    {'Métrica':<20} {'Valor':>15}")
    print(f"    {'-'*35}")
    print(f"    {'Promedio':<20} {exitos_con_vix['VIX_change_pct'].mean():>14.2f}%")
    print(f"    {'Mediana':<20} {exitos_con_vix['VIX_change_pct'].median():>14.2f}%")
    print(f"    {'Desv. Estándar':<20} {exitos_con_vix['VIX_change_pct'].std():>14.2f}%")
    print(f"    {'Mínimo':<20} {exitos_con_vix['VIX_change_pct'].min():>14.2f}%")
    print(f"    {'Máximo':<20} {exitos_con_vix['VIX_change_pct'].max():>14.2f}%")

    print(f"\n  Niveles de VIX:")
    print(f"    {'Métrica':<20} {'Valor':>15}")
    print(f"    {'-'*35}")
    print(f"    {'VIX Inicial Prom.':<20} {exitos_con_vix['VIX'].mean():>14.2f}")
    print(f"    {'VIX Final Prom.':<20} {exitos_con_vix['VIX_fwd_25'].mean():>14.2f}")
    print(f"    {'Cambio Neto':<20} {exitos_con_vix['VIX_fwd_25'].mean() - exitos_con_vix['VIX'].mean():>14.2f} pts")

    # COMPARACIÓN
    print(f"\n{'⚖️  COMPARACIÓN: DESASTRES vs ÉXITOS':^100}")
    print("=" * 100)

    diff_pts_mean = exitos_con_vix['VIX_change_pts'].mean() - desastres_con_vix['VIX_change_pts'].mean()
    diff_pts_median = exitos_con_vix['VIX_change_pts'].median() - desastres_con_vix['VIX_change_pts'].median()
    diff_pct_mean = exitos_con_vix['VIX_change_pct'].mean() - desastres_con_vix['VIX_change_pct'].mean()
    diff_pct_median = exitos_con_vix['VIX_change_pct'].median() - desastres_con_vix['VIX_change_pct'].median()

    print(f"\n  Diferencia en Cambio del VIX (Éxitos - Desastres):")
    print(f"    {'Métrica':<30} {'Valor':>20}")
    print(f"    {'-'*50}")
    print(f"    {'Promedio (puntos nom.)':<30} {diff_pts_mean:>19.2f} pts")
    print(f"    {'Mediana (puntos nom.)':<30} {diff_pts_median:>19.2f} pts")
    print(f"    {'Promedio (porcentaje)':<30} {diff_pct_mean:>19.2f}%")
    print(f"    {'Mediana (porcentaje)':<30} {diff_pct_median:>19.2f}%")

    print(f"\n{'CONCLUSIONES':^100}")
    print("=" * 100)

    print(f"\n  1. CAMBIOS PROMEDIO:")
    if diff_pts_mean < 0:
        print(f"     • Los ÉXITOS muestran un descenso del VIX {abs(diff_pts_mean):.2f} pts MAYOR que los desastres")
        print(f"     • Esto sugiere que el VIX tiende a bajar más en operaciones exitosas")
    else:
        print(f"     • Los DESASTRES muestran un descenso del VIX {abs(diff_pts_mean):.2f} pts MAYOR que los éxitos")
        print(f"     • Esto sugiere que el VIX tiende a bajar más en operaciones desastrosas")

    print(f"\n  2. CAMBIOS MEDIANOS:")
    if diff_pts_median < 0:
        print(f"     • Por mediana, los ÉXITOS muestran un descenso del VIX {abs(diff_pts_median):.2f} pts MAYOR")
        print(f"     • La mediana es más robusta frente a valores extremos")
    else:
        print(f"     • Por mediana, los DESASTRES muestran un descenso del VIX {abs(diff_pts_median):.2f} pts MAYOR")

    print(f"\n  3. VOLATILIDAD:")
    print(f"     • Desv. Est. Desastres: {desastres_con_vix['VIX_change_pts'].std():.2f} pts")
    print(f"     • Desv. Est. Éxitos: {exitos_con_vix['VIX_change_pts'].std():.2f} pts")
    if exitos_con_vix['VIX_change_pts'].std() > desastres_con_vix['VIX_change_pts'].std():
        print(f"     • Los ÉXITOS muestran MAYOR volatilidad en el cambio del VIX")
    else:
        print(f"     • Los DESASTRES muestran MAYOR volatilidad en el cambio del VIX")

    print("\n" + "=" * 100)

    # Guardar resultados a CSV
    print("\n  Guardando resultados detallados...")

    desastres_export = desastres_con_vix[['dia', 'VIX', 'dia_fwd_25', 'VIX_fwd_25',
                                          'VIX_change_pts', 'VIX_change_pct',
                                          'PnL_fwd_pct_25', 'PnL_fwd_pts_25']].copy()
    desastres_export['tipo'] = 'DESASTRE'

    exitos_export = exitos_con_vix[['dia', 'VIX', 'dia_fwd_25', 'VIX_fwd_25',
                                     'VIX_change_pts', 'VIX_change_pct',
                                     'PnL_fwd_pct_25', 'PnL_fwd_pts_25']].copy()
    exitos_export['tipo'] = 'EXITO'

    # Combinar y guardar
    resultados = pd.concat([desastres_export, exitos_export], ignore_index=True)
    resultados.to_csv('analisis_vix_desastres_exitos_resultados.csv', index=False)
    print(f"  ✓ Resultados guardados en: analisis_vix_desastres_exitos_resultados.csv")

    # Guardar resumen estadístico
    resumen = pd.DataFrame({
        'Tipo': ['DESASTRE', 'ÉXITO'],
        'N_Operaciones': [len(desastres_con_vix), len(exitos_con_vix)],
        'VIX_Inicial_Prom': [desastres_con_vix['VIX'].mean(), exitos_con_vix['VIX'].mean()],
        'VIX_Final_Prom': [desastres_con_vix['VIX_fwd_25'].mean(), exitos_con_vix['VIX_fwd_25'].mean()],
        'VIX_Change_Pts_Prom': [desastres_con_vix['VIX_change_pts'].mean(), exitos_con_vix['VIX_change_pts'].mean()],
        'VIX_Change_Pts_Mediana': [desastres_con_vix['VIX_change_pts'].median(), exitos_con_vix['VIX_change_pts'].median()],
        'VIX_Change_Pct_Prom': [desastres_con_vix['VIX_change_pct'].mean(), exitos_con_vix['VIX_change_pct'].mean()],
        'VIX_Change_Pct_Mediana': [desastres_con_vix['VIX_change_pct'].median(), exitos_con_vix['VIX_change_pct'].median()],
        'VIX_Change_Pts_Std': [desastres_con_vix['VIX_change_pts'].std(), exitos_con_vix['VIX_change_pts'].std()],
    })
    resumen.to_csv('analisis_vix_resumen_estadistico.csv', index=False)
    print(f"  ✓ Resumen estadístico guardado en: analisis_vix_resumen_estadistico.csv")

    print("\n" + "=" * 100)
    print(" " * 40 + "FIN DEL ANÁLISIS")
    print("=" * 100 + "\n")

if __name__ == "__main__":
    analizar_cambios_vix()
