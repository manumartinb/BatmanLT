# ANÁLISIS DELTA_TOTAL COMO PREDICTOR DE PNL Y PNLDV
## Estrategia Batman - Análisis Cuantitativo Avanzado

---

## RESUMEN EJECUTIVO

**Pregunta de investigación**: ¿Tiene delta_total relación predictiva con el comportamiento de PnL FWD y PNLDV FWD?

**Respuesta directa**:
- **Delta_total es un PREDICTOR MUY POBRE de PnL FWD** (r=-0.059, AUC=0.511)
- **Delta_total tiene FUERTE correlación NEGATIVA con PNLDV FWD** (r=-0.667)
- **Hallazgo contraintuitivo**: Menor delta → Mejor PnL (contradiciendo expectativas direccionales)

**Implicación operativa**: No utilizar delta_total como criterio de entrada/salida basado en expectativas de PnL. Su valor predictivo es equivalente al azar.

---

## 1. CONTEXTO Y DATASET

### Datos Analizados
- **Total operaciones**: 6,463 trades con datos completos de delta_total
- **Ventanas temporales**: T+0, FWD_01, FWD_05, FWD_25, FWD_50
- **Variables objetivo**: PnL_fwd_pts y PnLDV_fwd en cada ventana

### Distribución de Delta_Total
```
Estadísticos Descriptivos:
- Media:     0.084 (ligeramente positiva)
- Mediana:   0.103 (positiva)
- Std Dev:   0.165
- Min:      -1.295
- Max:       1.187

Distribución de Signos:
- Delta Positivo:  6,361 ops (98.4%) ← Estrategia predominantemente alcista
- Delta Negativo:     95 ops (1.5%)
- Delta Neutro:        7 ops (0.1%)
```

**Observación crítica**: El 98.4% de las operaciones tienen delta positivo, indicando una estrategia fuertemente sesgada alcista.

---

## 2. CORRELACIÓN: DELTA_TOTAL vs PNL FWD

### Resultados por Ventana Temporal

| Ventana | Correlación | P-value | Significancia |
|---------|-------------|---------|---------------|
| **T+0** | -0.028 | 0.027 | ✓ Significativa |
| **FWD_01** | -0.034 | 0.006 | ✓ Significativa |
| **FWD_05** | -0.049 | 0.000 | ✓ Significativa |
| **FWD_25** | -0.057 | 0.000 | ✓ Significativa |
| **FWD_50** | -0.059 | 0.000 | ✓ Significativa |

### Interpretación

**Estadísticamente significativo pero PRÁCTICAMENTE IRRELEVANTE**

- Las correlaciones son estadísticamente significativas (p < 0.05) debido al gran tamaño muestral (n=6,463)
- Sin embargo, el coeficiente de correlación r=-0.059 en FWD_50 indica:
  - **R² = 0.0035** → Delta_total explica solo el **0.35%** de la varianza del PnL FWD 50
  - **99.65% de la varianza del PnL** se explica por otros factores

**Conclusión**: Delta_total NO tiene poder predictivo sobre PnL FWD.

---

## 3. CORRELACIÓN: DELTA_TOTAL vs PNLDV FWD

### Resultados por Ventana Temporal

| Ventana | Correlación | P-value | Magnitud |
|---------|-------------|---------|----------|
| **T+0** | **-0.667** | 0.000 | Fuerte Negativa |
| **FWD_01** | -0.603 | 0.000 | Moderada-Fuerte Negativa |
| **FWD_05** | -0.552 | 0.000 | Moderada Negativa |
| **FWD_25** | -0.483 | 0.000 | Moderada Negativa |
| **FWD_50** | -0.461 | 0.000 | Moderada Negativa |

### Interpretación

**CORRELACIÓN FUERTE Y DECRECIENTE EN EL TIEMPO**

En T+0:
- **R² = 0.445** → Delta_total explica el **44.5%** de la varianza de PNLDV en T+0
- Correlación negativa: Mayor delta → Peor PNLDV (más negativo)

Evolución temporal:
- La correlación **decae** de r=-0.667 (T+0) a r=-0.461 (FWD_50)
- Interpretación: El impacto del delta inicial se diluye con el tiempo
- Otros factores (movimiento del subyacente, cambios de volatilidad) ganan importancia

**Conclusión**: Delta_total es un fuerte driver de PNLDV en T+0, pero su influencia disminuye progresivamente.

---

## 4. DELTA_TOTAL COMO PREDICTOR DE DETERIORO GRAVE

### Definición de Deterioro Grave
- **Deterioro Grave**: PnL_fwd_pts_50 < -100 pts
- **Tasa base**: 12.4% de operaciones (803 de 6,463)

### Performance Predictiva

**AUC-ROC = 0.511**

Interpretación:
- AUC = 0.5 → Predictor aleatorio
- AUC = 0.511 → **Prácticamente equivalente al azar**
- Delta_total NO discrimina entre operaciones que deteriorarán y las que no

### Comparación con Predictores Efectivos

| Predictor | AUC | Interpretación |
|-----------|-----|----------------|
| **delta_total** | 0.511 | Inútil |
| **PnL_fwd_pts_25** | 0.956 | Excelente |
| **IV_K3** | 0.933 | Excelente |
| **delta_pnldv_25** | 0.887 | Muy Bueno |

**Conclusión**: Delta_total no debe ser considerado en modelos predictivos de deterioro.

---

## 5. ANÁLISIS POR CATEGORÍAS DE DELTA

### Segmentación por Percentiles

Dividimos delta_total en 3 categorías:
- **Muy Negativo** (P0-P10): delta ≤ -0.062
- **Neutral** (P10-P90): -0.062 < delta < 0.237
- **Muy Positivo** (P90-P100): delta ≥ 0.237

### Performance por Categoría

| Categoría Delta | N Ops | PnL Medio FWD_50 | Tasa Deterioro | PNLDV Medio T+0 |
|-----------------|-------|------------------|----------------|-----------------|
| **Muy Negativo** (P0-P10) | 647 | **+70.24 pts** | 12.5% | -66.8 |
| **Neutral** (P10-P90) | 5,170 | +58.01 pts | 12.0% | -86.5 |
| **Muy Positivo** (P90-P100) | 646 | **+41.22 pts** | 15.0% | -117.5 |

### HALLAZGO CONTRAINTUITIVO

**Menor Delta → Mejor PnL**

1. **Operaciones con delta muy negativo** (P0-P10):
   - PnL medio: **+70.24 pts** (mejor grupo)
   - PNLDV medio: **-66.8** (menos peor caso, menos riesgo)
   - Tasa deterioro: 12.5%

2. **Operaciones con delta muy positivo** (P90-P100):
   - PnL medio: **+41.22 pts** (peor grupo, -41% vs delta negativo)
   - PNLDV medio: **-117.5** (peor caso más severo, mayor riesgo)
   - Tasa deterioro: 15.0% (20% más alta)

**Test Estadístico**:
- Diferencia PnL: +29.02 pts a favor de delta negativo
- T-statistic: -3.087
- P-value: 0.002
- **Conclusión**: Diferencia estadísticamente significativa (p < 0.05)

### Explicación del Fenómeno

**¿Por qué menor delta produce mejor PnL?**

Hipótesis explicativas:
1. **Estrategia Batman es estructuralmente alcista**: Con el 98.4% de ops con delta positivo, las operaciones con delta más bajo (pero aún positivo) pueden tener mejor balance riesgo/recompensa
2. **Menor exposición direccional**: Delta más bajo → menor sensibilidad a movimientos adversos del subyacente
3. **Mejor gestión de PNLDV**: Delta bajo correlaciona con menor PNLDV (menor riesgo de worst-case)
4. **Sesgo de selección temporal**: Operaciones con delta negativo pueden abrirse en condiciones de mercado más favorables

---

## 6. OPERACIONES DELTA-NEUTRAL vs DIRECCIONALES

### Definición de Grupos
- **Delta-neutral**: |delta_total| < 0.10 (aproximadamente neutral)
- **Direccional**: |delta_total| ≥ 0.10

### Resultados Comparativos

| Grupo | N Ops | PnL Medio FWD_50 | Tasa Deterioro | PNLDV Medio |
|-------|-------|------------------|----------------|-------------|
| **Delta-Neutral** | 3,266 (50.5%) | +57.53 pts | 12.7% | -78.5 |
| **Direccional** | 3,197 (49.5%) | +61.47 pts | 12.1% | -95.7 |

**Diferencia**: +3.94 pts a favor de operaciones direccionales

**Test Estadístico**:
- T-statistic: 0.672
- P-value: 0.502
- **Conclusión**: NO hay diferencia estadísticamente significativa (p > 0.05)

**Implicación**: La neutralidad direccional (delta-hedging) NO mejora los resultados en esta estrategia Batman.

---

## 7. ANÁLISIS POR QUINTILES DE DELTA

### Distribución de PnL por Quintil

| Quintil | Rango Delta | N Ops | PnL Medio | Tasa Deterioro |
|---------|-------------|-------|-----------|----------------|
| **Q1** (más negativo) | < -0.015 | 1,293 | +67.92 pts | 11.6% |
| **Q2** | -0.015 a 0.056 | 1,293 | +60.39 pts | 11.5% |
| **Q3** | 0.056 a 0.103 | 1,293 | +56.55 pts | 12.5% |
| **Q4** | 0.103 a 0.169 | 1,292 | +58.88 pts | 12.4% |
| **Q5** (más positivo) | > 0.169 | 1,292 | +50.61 pts | 14.0% |

### Patrón Observado

**Tendencia clara**: A medida que delta aumenta, el PnL medio **disminuye**

- Q1 (delta más bajo): **+67.92 pts**
- Q5 (delta más alto): **+50.61 pts**
- **Diferencia**: -17.31 pts (-25.5% peor performance)

**Tasa de deterioro**:
- Q1: 11.6% (más seguro)
- Q5: 14.0% (más arriesgado)

---

## 8. REGRESIÓN LINEAL: DELTA → PNL FWD 50

### Modelo Simple

```
PnL_fwd_pts_50 = β₀ + β₁ × delta_total + ε
```

**Resultados**:
- Intercepto (β₀): +62.74 pts
- Pendiente (β₁): -47.26 pts por unidad de delta
- R²: **0.0035** (0.35% de varianza explicada)
- P-value: < 0.001 (estadísticamente significativo)

**Interpretación**:
- Por cada **+0.10** de incremento en delta → PnL disminuye **-4.73 pts**
- Ejemplo: Delta de 0.0 → PnL esperado = +62.74 pts
- Ejemplo: Delta de 0.2 → PnL esperado = +53.29 pts

**Conclusión**: Relación estadísticamente significativa pero magnitud del efecto es MUY PEQUEÑA (R²=0.35%).

---

## 9. CONCLUSIONES Y RECOMENDACIONES

### Conclusión General

**Delta_total NO es un predictor útil de PnL FWD en la estrategia Batman**

Evidencia:
1. Correlación prácticamente nula con PnL FWD (r=-0.059)
2. AUC-ROC = 0.511 (equivalente a predictor aleatorio)
3. Solo explica 0.35% de varianza del PnL

### Hallazgos Relevantes

#### 1. Delta y PNLDV: Fuerte Relación Negativa
- Mayor delta → Peor PNLDV (r=-0.667 en T+0)
- Implicación: Operaciones con delta alto tienen peor worst-case scenario
- Recomendación: **Monitorear delta para gestión de riesgo de PNLDV**, NO para predicción de PnL

#### 2. Fenómeno Contraintuitivo: Menor Delta → Mejor PnL
- Operaciones con delta bajo (P0-P10): +70.24 pts
- Operaciones con delta alto (P90-P100): +41.22 pts
- Diferencia: **+29 pts** (estadísticamente significativa, p=0.002)
- Posible explicación: Mejor balance riesgo/recompensa con menor exposición direccional

#### 3. Delta-Neutral NO Mejora Performance
- Delta-neutral: +57.53 pts
- Direccional: +61.47 pts
- Diferencia NO significativa (p=0.502)
- Recomendación: **No hedgear delta como estrategia sistemática**

### Recomendaciones Operativas

#### ✅ QUÉ HACER CON DELTA_TOTAL

1. **Gestión de Riesgo de PNLDV**:
   - Usar delta para estimar riesgo de worst-case
   - Delta alto → PNLDV más negativo → Mayor exposición al riesgo

2. **NO usar como filtro de entrada**:
   - Delta NO predice éxito/fracaso de la operación
   - No rechazar operaciones por delta alto/bajo basado en expectativas de PnL

3. **Monitoreo informativo**:
   - Incluir delta en dashboard como métrica de exposición direccional
   - NO usarlo como trigger de cierre

#### ❌ QUÉ NO HACER

1. **NO usar delta como predictor de deterioro**:
   - AUC=0.511 → Equivalente a lanzar una moneda
   - Existen predictores mucho mejores (PnL_fwd_pts_25, IV_K3)

2. **NO hedgear delta sistemáticamente**:
   - No mejora resultados (p=0.502)
   - Aumenta costos de transacción sin beneficio claro

3. **NO asumir "mayor delta = mejor PnL"**:
   - La evidencia muestra lo contrario
   - Relación es débil pero negativa (r=-0.059)

### Integración con Hallazgos Previos

| Predictor | AUC | Uso Recomendado |
|-----------|-----|-----------------|
| **PnL_fwd_pts_25** | 0.956 | ✅ Criterio primario de cierre |
| **IV_K3 (T+0)** | 0.933 | ✅ Filtro de entrada |
| **delta_pnldv_25** | 0.887 | ✅ Confirmación de deterioro |
| **delta_total** | 0.511 | ❌ No usar para predicción PnL |

### Respuesta Final a la Pregunta Original

**"¿Tiene relación delta como predictor del comportamiento de PnL FWD y PNLDV FWD?"**

**Respuesta**:
- **Para PnL FWD**: NO. Delta_total NO tiene poder predictivo útil (r=-0.059, AUC=0.511)
- **Para PNLDV FWD**: SÍ. Delta_total tiene fuerte correlación negativa (r=-0.667), útil para evaluar riesgo de worst-case, NO para predecir PnL final

**Recomendación final**: Descarte delta_total como variable predictiva en modelos de deterioro de PnL. Manténgalo solo como métrica de gestión de riesgo de PNLDV.

---

## 10. VISUALIZACIONES GENERADAS

**Archivo**: `analisis_delta_total_predictor.png`

Incluye 9 paneles:
1. Distribución de delta_total (histograma)
2. Delta vs PnL FWD_50 (scatter plot)
3. Delta vs PNLDV FWD_50 (scatter plot)
4. Correlación delta-PnL por ventana temporal
5. Correlación delta-PNLDV por ventana temporal
6. Curvas ROC para predicción de deterioro
7. PnL medio por quintil de delta
8. Tasa de deterioro por quintil de delta
9. Comparación delta-neutral vs direccional

---

## ANEXO: COMPARACIÓN CON ANÁLISIS PREVIOS

### Ranking de Predictores de Deterioro Grave PnL FWD 50

| # | Predictor | AUC | Tipo | Ventana |
|---|-----------|-----|------|---------|
| 1 | PnL_fwd_pts_25 | 0.956 | FWD | W=25 |
| 2 | IV_K3 | 0.933 | T+0 | T+0 |
| 3 | delta_pnldv_25 | 0.887 | Derivada FWD | W=25 |
| 4 | PnLDV_fwd_25 | 0.853 | FWD | W=25 |
| 5 | theta_total | 0.742 | T+0 | T+0 |
| ... | ... | ... | ... | ... |
| **40** | **delta_total** | **0.511** | **T+0** | **T+0** |

**Posición**: Delta_total ocupa el **último lugar** entre los 40 predictores analizados.

---

**Fecha de análisis**: 2025-11-21
**Dataset**: PNLDV.csv (7,092 operaciones totales, 6,463 con delta_total válido)
**Script**: `analisis_delta_total_predictor.py`
