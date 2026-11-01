"""
Data quality checks for the SNR / IGAC property-transactions dataset (Medellín).

The DataFrame is assumed to be ALREADY LOADED (e.g. from Parquet). This module never reads files.

Quick start
-----------
    import pandas as pd
    from data_quality_checks import run_quality_checks

    df = pd.read_parquet("your_file.parquet")          # <- your own loading code
    res = run_quality_checks(df, export_path="reporte_calidad.xlsx")

    res.tables["completitud"]      # completeness per column
    res.tables["duplicados"]       # duplicate checks
    res.tables["reglas"]           # validity / consistency rules
    res.tables["outliers_metodos"] # outlier thresholds per method
    res.tables["categorias"]       # value categories (volume + value)
    res.df                         # copy of df + new columns (VALOR_CATEGORIA, VALOR_OUTLIER, ...)
    print("\n".join(res.findings))  # plain-language summary for the data owners

Report labels are in Spanish because the audience is the data owner (IGAC / SNR).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Configuration (all assumptions live here, so they are easy to review with the owners)
# --------------------------------------------------------------------------------------

EXPECTED_COLUMNS = [
    "PK", "MATRICULA", "FECHA_RADICA_TEXTO", "FECHA_APERTURA_TEXTO", "YEAR_RADICA", "ORIP",
    "DIVIPOLA", "DEPARTAMENTO", "MUNICIPIO", "TIPO_PREDIO_ZONA", "CATEGORIA_RURALIDAD",
    "NUM_ANOTACION", "ESTADO_FOLIO", "FOLIOS_DERIVADOS", "DINAMICA_INMOBILIARIA", "COD_NATUJUR",
    "NOMBRE_NATUJUR", "NUMERO_CATASTRAL", "NUMERO_CATASTRAL_ANTIGUO", "DOCUMENTO_JUSTIFICATIVO",
    "COUNT_A", "COUNT_DE", "PREDIOS_NUEVOS", "TIENE_VALOR", "TIENE_MAS_DE_UN_VALOR", "VALOR",
]

# API field names / accent variants -> canonical column name
COLUMN_ALIASES = {
    "DINAMICA_2024": "DINAMICA_INMOBILIARIA",
    "CATEGORIA_RURALIDAD_2024": "CATEGORIA_RURALIDAD",
}

# Values that mean "no information" even though they are not technically null
PLACEHOLDERS = {"", "NAN", "NONE", "NULL", "N/A", "NA", "SIN INFORMACION", "[]", "{}"}

# Columns that are legitimately empty in many rows -> completeness must be read with context
CONDITIONAL_COLUMNS = {
    "ESTADO_FOLIO": "Puede venir vacío según el estado del folio.",
    "FOLIOS_DERIVADOS": "Solo se llena cuando el acto genera folios derivados (desenglobe, venta parcial, etc.).",
    "NUMERO_CATASTRAL": "Solo disponible si el folio está cruzado con catastro.",
    "NUMERO_CATASTRAL_ANTIGUO": "Solo existe para predios con numeración histórica.",
    "VALOR": "Solo aplica a actos con valor (TIENE_VALOR = 1). Ver completitud condicional.",
}

# Plausibility limit for VALOR in COP. ~1 monthly minimum wage: below this a property sale is
# very unlikely (possible typing error, partial rights, or value expressed in thousands).
MIN_PLAUSIBLE_VALUE = 1_000_000
MAX_PLAUSIBLE_VALUE = None  # e.g. 1e12 to cap implausibly large values; None = no cap

# Illustrative fixed cut-offs in COP (4 categories). VALIDATE WITH THE DATA OWNERS before using.
FIXED_BINS_COP = {
    4: [0, 150e6, 350e6, 700e6, np.inf],
    3: [0, 200e6, 500e6, np.inf],
}
CATEGORY_LABELS = {
    4: ["1-Bajo", "2-Medio bajo", "3-Medio alto", "4-Alto"],
    3: ["1-Bajo", "2-Medio", "3-Alto"],
}

ESTADO_FOLIO_VALID = {"ACTIVO", "CERRADO", "EN CUSTODIA"}
ZONA_VALID = {"URBANO", "RURAL", "SIN INFORMACION"}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def _fix_mojibake(s: str) -> str:
    """'DinÃ¡mica' -> 'Dinámica' (UTF-8 text that was decoded as latin-1)."""
    try:
        return s.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the canonical upper-case, accent-free names used in this module."""
    mapping = {}
    for c in df.columns:
        n = _strip_accents(_fix_mojibake(str(c))).strip().upper().replace(" ", "_")
        mapping[c] = COLUMN_ALIASES.get(n, n)
    return df.rename(columns=mapping)


def _norm_text(s: pd.Series) -> pd.Series:
    """Upper-case, trimmed, accent-free string version of a series (NaN preserved)."""
    out = s.astype("string").str.strip().str.upper()
    return out.str.normalize("NFKD").str.replace(r"[\u0300-\u036f]", "", regex=True)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _pct(n: int | float, total: int | float) -> float:
    return round(100 * n / total, 2) if total else 0.0


def _severity_from_completeness(p: float) -> str:
    if p >= 99:
        return "Excelente"
    if p >= 95:
        return "Bueno"
    if p >= 80:
        return "Aceptable"
    if p >= 50:
        return "Deficiente"
    return "Crítico"


# --------------------------------------------------------------------------------------
# 1. Schema
# --------------------------------------------------------------------------------------

def schema_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in EXPECTED_COLUMNS:
        rows.append({"columna": c, "estado": "OK" if c in df.columns else "FALTANTE",
                     "dtype": str(df[c].dtype) if c in df.columns else None})
    for c in df.columns:
        if c not in EXPECTED_COLUMNS:
            rows.append({"columna": c, "estado": "NO ESPERADA (no está en el diccionario)",
                         "dtype": str(df[c].dtype)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 2. Completeness
# --------------------------------------------------------------------------------------

def completeness_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per column: nulls, placeholder values ('SIN INFORMACIÓN', '', '[]'...), completeness %.
    - completitud_pct          = % of rows with a real, non-placeholder value.
    - completitud_condicional  = for VALOR only: % filled among rows where TIENE_VALOR == 1.
    """
    n = len(df)
    rows = []
    for c in df.columns:
        s = df[c]
        n_null = int(s.isna().sum())
        n_placeholder = 0
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            try:
                txt = _norm_text(s)
                n_placeholder = int((s.notna() & txt.isin(PLACEHOLDERS).fillna(False)).sum())
            except Exception:  # unhashable values (lists/dicts)
                txt = s.astype(str)
                n_placeholder = int((s.notna() & txt.isin(["", "[]", "{}"])).sum())
        n_missing = n_null + n_placeholder
        comp = _pct(n - n_missing, n)

        cond = None
        if c == "VALOR" and "TIENE_VALOR" in df.columns:
            applies = _num(df["TIENE_VALOR"]) == 1
            cond = _pct(int((applies & s.notna()).sum()), int(applies.sum()))

        try:
            n_unique = int(s.nunique(dropna=True))
        except TypeError:
            n_unique = int(s.astype(str).nunique())

        base = cond if cond is not None else comp
        rows.append({
            "columna": c,
            "dtype": str(s.dtype),
            "filas": n,
            "nulos": n_null,
            "valores_placeholder": n_placeholder,
            "faltantes_total": n_missing,
            "completitud_pct": comp,
            "completitud_condicional_pct": cond,
            "valores_unicos": n_unique,
            "valores_unicos_pct": _pct(n_unique, n),
            "clasificacion": _severity_from_completeness(base),
            "nota": CONDITIONAL_COLUMNS.get(c, ""),
        })
    out = pd.DataFrame(rows).sort_values("completitud_pct").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------------------
# 3. Duplicates
# --------------------------------------------------------------------------------------

def _dup_mask(df: pd.DataFrame, cols=None) -> pd.Series:
    """All rows that belong to a duplicated group (keep=False). Robust to unhashable cells."""
    try:
        return df.duplicated(subset=cols, keep=False)
    except TypeError:
        sub = df if cols is None else df[list(cols)]
        return sub.astype(str).duplicated(keep=False)


def duplicates_report(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Returns (summary, detail) where detail maps check-name -> rows involved in duplicates.

    Checks
      * PK repeated                 -> PK should be unique per transaction.
      * Full-row duplicate          -> identical in every column.
      * MATRICULA + NUM_ANOTACION   -> the same annotation of the same folio appears twice.
      * Same act, same value        -> same folio, filing date, legal act, document and value under
                                       different PKs (possible double registration).
    NOTE: the same DOCUMENTO_JUSTIFICATIVO across different MATRICULAS is NORMAL (one deed can
    cover several properties), so it is not flagged.
    """
    n = len(df)
    checks = {
        "PK repetida": ["PK"],
        "Fila completa duplicada": None,
        "MATRICULA + NUM_ANOTACION repetidos": ["MATRICULA", "NUM_ANOTACION"],
        "Mismo folio, fecha, acto, documento y valor": [
            "MATRICULA", "FECHA_RADICA_TEXTO", "COD_NATUJUR", "DOCUMENTO_JUSTIFICATIVO", "VALOR"],
    }
    summary, detail = [], {}
    for name, cols in checks.items():
        if cols is not None and not all(c in df.columns for c in cols):
            summary.append({"verificacion": name, "filas_involucradas": None, "grupos_duplicados": None,
                            "filas_sobrantes": None, "pct_filas": None,
                            "estado": "NO EVALUADA (faltan columnas)"})
            continue
        mask = _dup_mask(df, cols)
        involved = int(mask.sum())
        try:
            extra = int(df.duplicated(subset=cols).sum())          # rows beyond the first of each group
        except TypeError:                                          # unhashable cells (lists/dicts)
            extra = int((df if cols is None else df[cols]).astype(str).duplicated().sum())
        groups = 0
        if involved:
            sub = df.loc[mask] if cols is None else df.loc[mask, cols]
            groups = int(sub.astype(str).drop_duplicates().shape[0])
        summary.append({"verificacion": name, "filas_involucradas": involved, "grupos_duplicados": groups,
                        "filas_sobrantes": extra, "pct_filas": _pct(involved, n),
                        "estado": "OK" if involved == 0 else "REVISAR"})
        if involved:
            sort_cols = cols if cols is not None else [df.columns[0]]
            detail[name] = df.loc[mask].sort_values(sort_cols)
    return pd.DataFrame(summary), detail


# --------------------------------------------------------------------------------------
# 4. Validity / consistency rules
# --------------------------------------------------------------------------------------

def validity_report(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Returns (summary, masks). masks[rule] is a boolean Series marking violating rows."""
    results, masks = [], {}

    def add(rule, severity, desc, cols, fn):
        if not all(c in df.columns for c in cols):
            return
        try:
            m = fn().fillna(False).astype(bool)
        except Exception as e:  # never let one rule break the whole report
            results.append({"regla": rule, "severidad": severity, "descripcion": desc,
                            "filas_con_error": None, "pct_filas": None, "estado": f"ERROR AL EVALUAR: {e}"})
            return
        k = int(m.sum())
        masks[rule] = m
        results.append({"regla": rule, "severidad": severity, "descripcion": desc, "filas_con_error": k,
                        "pct_filas": _pct(k, len(df)), "estado": "OK" if k == 0 else "REVISAR"})

    def neq(a, b):
        return (a != b) & a.notna() & b.notna()

    # --- parsed helpers --------------------------------------------------------------
    d_rad = pd.to_datetime(df["FECHA_RADICA_TEXTO"], errors="coerce") if "FECHA_RADICA_TEXTO" in df else None
    d_ape = pd.to_datetime(df["FECHA_APERTURA_TEXTO"], errors="coerce") if "FECHA_APERTURA_TEXTO" in df else None
    pk = None
    if "PK" in df:
        parts = df["PK"].astype("string").str.split("-", expand=True)
        if parts.shape[1] >= 6:
            pk = parts

    # --- dates -----------------------------------------------------------------------
    add("FECHA_RADICA_INVALIDA", "Crítica", "Fecha de radicación con valor pero no interpretable como fecha.",
        ["FECHA_RADICA_TEXTO"], lambda: df["FECHA_RADICA_TEXTO"].notna() & d_rad.isna())
    add("FECHA_APERTURA_INVALIDA", "Crítica", "Fecha de apertura con valor pero no interpretable como fecha.",
        ["FECHA_APERTURA_TEXTO"], lambda: df["FECHA_APERTURA_TEXTO"].notna() & d_ape.isna())
    add("FECHA_RADICA_FUTURA_O_ANTIGUA", "Crítica", "Fecha de radicación posterior a hoy o anterior a 1900.",
        ["FECHA_RADICA_TEXTO"], lambda: (d_rad > pd.Timestamp.today()) | (d_rad < pd.Timestamp("1900-01-01")))
    add("YEAR_RADICA_vs_FECHA", "Crítica", "YEAR_RADICA no coincide con el año de FECHA_RADICA_TEXTO.",
        ["YEAR_RADICA", "FECHA_RADICA_TEXTO"], lambda: neq(_num(df["YEAR_RADICA"]), d_rad.dt.year.astype("float")))
    add("APERTURA_POSTERIOR_A_RADICA", "Informativa",
        "Apertura posterior a la radicación. No siempre es error (sentencias, correcciones, trámites de fin de año).",
        ["FECHA_RADICA_TEXTO", "FECHA_APERTURA_TEXTO"], lambda: d_ape > d_rad)

    # --- PK consistency --------------------------------------------------------------
    add("PK_FORMATO", "Advertencia", "PK no tiene el formato DIVIPOLA-ORIP-MATRICULA-ANOTACION-NATUJUR-AÑO.",
        ["PK"], lambda: df["PK"].notna() & ~df["PK"].astype("string").str.match(r"^\d{5}-\d{3}-\d+-\d+-\d+-\d{4}$"))
    if pk is not None:
        add("PK_ANIO_vs_YEAR_RADICA", "Advertencia", "El año dentro de la PK no coincide con YEAR_RADICA.",
            ["PK", "YEAR_RADICA"], lambda: neq(_num(pk[5]), _num(df["YEAR_RADICA"])))
        add("PK_ANOTACION_vs_NUM_ANOTACION", "Advertencia", "El número de anotación en la PK no coincide con NUM_ANOTACION.",
            ["PK", "NUM_ANOTACION"], lambda: neq(_num(pk[3]), _num(df["NUM_ANOTACION"])))
        add("PK_NATUJUR_vs_COD_NATUJUR", "Advertencia", "El código de naturaleza jurídica en la PK no coincide con COD_NATUJUR.",
            ["PK", "COD_NATUJUR"], lambda: neq(_num(pk[4]), _num(df["COD_NATUJUR"])))

    # --- geography -------------------------------------------------------------------
    add("MATRICULA_vs_ORIP", "Advertencia", "Los 3 primeros dígitos de MATRICULA no coinciden con el código ORIP.",
        ["MATRICULA", "ORIP"],
        lambda: df["MATRICULA"].astype("string").str.split("-").str[0].str.zfill(3)
        .ne(_num(df["ORIP"]).astype("Int64").astype("string").str.zfill(3)) & df["MATRICULA"].notna() & df["ORIP"].notna())
    add("DIVIPOLA_SIN_CERO_INICIAL", "Informativa",
        "DIVIPOLA con 4 dígitos (se perdió el cero inicial, p. ej. 5001 en lugar de 05001). Debe guardarse como texto.",
        ["DIVIPOLA"], lambda: df["DIVIPOLA"].astype("string").str.strip().str.len().eq(4))
    add("DIVIPOLA_LONGITUD_INVALIDA", "Crítica", "DIVIPOLA no tiene 4 o 5 dígitos numéricos.",
        ["DIVIPOLA"], lambda: df["DIVIPOLA"].notna() &
        ~df["DIVIPOLA"].astype("string").str.strip().str.fullmatch(r"\d{4,5}"))
    add("TIPO_PREDIO_ZONA_FUERA_DE_DOMINIO", "Advertencia", "Valor distinto de URBANO / RURAL / SIN INFORMACIÓN.",
        ["TIPO_PREDIO_ZONA"], lambda: df["TIPO_PREDIO_ZONA"].notna() & ~_norm_text(df["TIPO_PREDIO_ZONA"]).isin(ZONA_VALID))
    add("ESTADO_FOLIO_FUERA_DE_DOMINIO", "Advertencia", "Valor distinto de ACTIVO / CERRADO / EN CUSTODIA.",
        ["ESTADO_FOLIO"], lambda: df["ESTADO_FOLIO"].notna() & ~_norm_text(df["ESTADO_FOLIO"]).isin(ESTADO_FOLIO_VALID))

    # --- binary flags ----------------------------------------------------------------
    for c in ["DINAMICA_INMOBILIARIA", "PREDIOS_NUEVOS", "TIENE_VALOR", "TIENE_MAS_DE_UN_VALOR"]:
        add(f"{c}_NO_BINARIA", "Crítica", f"{c} debe ser 0 o 1.", [c],
            lambda c=c: df[c].notna() & ~_num(df[c]).isin([0, 1]))

    # --- numeric sanity --------------------------------------------------------------
    add("NUM_ANOTACION_NO_POSITIVA", "Crítica", "NUM_ANOTACION debe ser un entero positivo.",
        ["NUM_ANOTACION"], lambda: _num(df["NUM_ANOTACION"]) <= 0)
    add("COUNT_A_NEGATIVO", "Crítica", "COUNT_A (receptores) no puede ser negativo.",
        ["COUNT_A"], lambda: _num(df["COUNT_A"]) < 0)
    add("COUNT_DE_NEGATIVO", "Crítica", "COUNT_DE (otorgantes) no puede ser negativo.",
        ["COUNT_DE"], lambda: _num(df["COUNT_DE"]) < 0)
    add("NUMERO_CATASTRAL_LONGITUD", "Advertencia", "NUMERO_CATASTRAL presente pero sin 30 dígitos.",
        ["NUMERO_CATASTRAL"],
        lambda: (df["NUMERO_CATASTRAL"].notna() & ~df["NUMERO_CATASTRAL"].astype("string").str.replace(r"\D", "", regex=True).str.len().eq(30))
        if not pd.api.types.is_numeric_dtype(df["NUMERO_CATASTRAL"]) else df["NUMERO_CATASTRAL"].notna())

    # --- VALOR consistency -----------------------------------------------------------
    v = _num(df["VALOR"]) if "VALOR" in df else None
    add("VALOR_NO_NUMERICO", "Crítica", "VALOR con contenido que no se puede convertir a número.",
        ["VALOR"], lambda: df["VALOR"].notna() & v.isna())
    add("VALOR_NO_POSITIVO", "Crítica", "VALOR menor o igual a cero.", ["VALOR"], lambda: v <= 0)
    add("TIENE_VALOR_1_SIN_VALOR", "Crítica", "TIENE_VALOR = 1 pero VALOR está vacío.",
        ["TIENE_VALOR", "VALOR"], lambda: (_num(df["TIENE_VALOR"]) == 1) & v.isna())
    add("TIENE_VALOR_0_CON_VALOR", "Advertencia", "TIENE_VALOR = 0 pero VALOR está diligenciado.",
        ["TIENE_VALOR", "VALOR"], lambda: (_num(df["TIENE_VALOR"]) == 0) & v.notna())
    add("MAS_DE_UN_VALOR_SIN_VALOR", "Advertencia", "TIENE_MAS_DE_UN_VALOR = 1 pero TIENE_VALOR = 0.",
        ["TIENE_VALOR", "TIENE_MAS_DE_UN_VALOR"],
        lambda: (_num(df["TIENE_MAS_DE_UN_VALOR"]) == 1) & (_num(df["TIENE_VALOR"]) == 0))
    add("VALOR_BAJO_RANGO_PLAUSIBLE", "Advertencia",
        f"VALOR positivo pero menor a {MIN_PLAUSIBLE_VALUE:,.0f} COP (posible error de digitación o derechos parciales).",
        ["VALOR"], lambda: (v > 0) & (v < MIN_PLAUSIBLE_VALUE))

    order = {"Crítica": 0, "Advertencia": 1, "Informativa": 2}
    summary = pd.DataFrame(results)
    if not summary.empty:
        summary = summary.sort_values(by=["severidad", "filas_con_error"], key=lambda s: s.map(order) if s.name == "severidad" else -s.fillna(0)).reset_index(drop=True)
    return summary, masks


# --------------------------------------------------------------------------------------
# 5. Outliers in VALOR
# --------------------------------------------------------------------------------------

def _tukey_bounds(x: pd.Series, k: float) -> tuple[float, float]:
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def outlier_analysis(
    df: pd.DataFrame,
    value_col: str = "VALOR",
    k: float = 1.5,
    k_extreme: float = 3.0,
    z_thresh: float = 3.5,
    min_plausible: float | None = MIN_PLAUSIBLE_VALUE,
    max_plausible: float | None = MAX_PLAUSIBLE_VALUE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (stats, methods, flags).

    Property prices are strongly right-skewed, so the PRIMARY method is Tukey's IQR on log10(VALOR).
    Two cross-check methods are reported: raw IQR and modified z-score (MAD) on log10(VALOR).
    `flags` has the same index as df and these columns:
        VALOR_OUTLIER          'Normal' | 'Atípico bajo' | 'Atípico alto' | NaN (no valid value)
        VALOR_OUTLIER_EXTREMO  True when beyond k_extreme * IQR (log scale)
        VALOR_OUT_IQR_RAW, VALOR_OUT_IQR_LOG, VALOR_OUT_MAD   per-method booleans
        VALOR_N_METODOS        how many of the 3 methods flag the row (0-3)
        VALOR_FUERA_RANGO_PLAUSIBLE
    """
    v = _num(df[value_col])
    valid = v.where(v > 0)
    x = valid.dropna()
    flags = pd.DataFrame(index=df.index)
    if len(x) < 4:
        empty = pd.DataFrame([{"nota": "Datos insuficientes (<4 valores positivos) para analizar atípicos."}])
        return empty, empty, flags

    lx = np.log10(x)

    # descriptive statistics
    q = x.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    stats = pd.DataFrame({"estadistico": [
        "n válidos (>0)", "n nulos", "n <= 0", "mínimo", "P1", "P5", "P25 (Q1)", "mediana", "P75 (Q3)", "P95", "P99",
        "máximo", "promedio", "desviación estándar", "coef. variación", "asimetría (skew)", "promedio / mediana"],
        "valor": [len(x), int(v.isna().sum()), int((v <= 0).sum()), x.min(), q[0.01], q[0.05], q[0.25], q[0.5],
                  q[0.75], q[0.95], q[0.99], x.max(), x.mean(), x.std(), x.std() / x.mean(), x.skew(),
                  x.mean() / x.median()]})

    # method 1: IQR on log scale (primary)
    lo_l, hi_l = _tukey_bounds(lx, k)
    lo_e, hi_e = _tukey_bounds(lx, k_extreme)
    # method 2: IQR on raw scale
    lo_r, hi_r = _tukey_bounds(x, k)
    # method 3: modified z-score on log scale
    med = lx.median()
    mad = (lx - med).abs().median()

    lv = np.log10(valid)
    flags["VALOR_OUT_IQR_LOG"] = (lv < lo_l) | (lv > hi_l)
    flags["VALOR_OUT_IQR_RAW"] = (valid < lo_r) | (valid > hi_r)
    if mad > 0:
        z = 0.6745 * (lv - med) / mad
        flags["VALOR_OUT_MAD"] = z.abs() > z_thresh
    else:
        flags["VALOR_OUT_MAD"] = False
    flags["VALOR_OUT_MAD"] = flags["VALOR_OUT_MAD"].fillna(False)
    flags[["VALOR_OUT_IQR_LOG", "VALOR_OUT_IQR_RAW"]] = flags[["VALOR_OUT_IQR_LOG", "VALOR_OUT_IQR_RAW"]].fillna(False)
    flags["VALOR_N_METODOS"] = flags[["VALOR_OUT_IQR_LOG", "VALOR_OUT_IQR_RAW", "VALOR_OUT_MAD"]].sum(axis=1).astype(int)

    label = pd.Series(np.where(lv < lo_l, "Atípico bajo", np.where(lv > hi_l, "Atípico alto", "Normal")), index=df.index)
    flags["VALOR_OUTLIER"] = label.where(valid.notna())
    flags["VALOR_OUTLIER_EXTREMO"] = ((lv < lo_e) | (lv > hi_e)).fillna(False)

    plaus = pd.Series(False, index=df.index)
    if min_plausible is not None:
        plaus |= (valid < min_plausible)
    if max_plausible is not None:
        plaus |= (valid > max_plausible)
    flags["VALOR_FUERA_RANGO_PLAUSIBLE"] = plaus.fillna(False)

    def row(name, lo, hi, mask_lo, mask_hi, note):
        return {"metodo": name, "limite_inferior": lo, "limite_superior": hi,
                "n_bajos": int(mask_lo.sum()), "n_altos": int(mask_hi.sum()),
                "pct_del_total_valido": _pct(int(mask_lo.sum() + mask_hi.sum()), len(x)), "nota": note}

    methods = pd.DataFrame([
        row(f"IQR sobre log10 (k={k}) [principal]", 10 ** lo_l, 10 ** hi_l, x < 10 ** lo_l, x > 10 ** hi_l,
            "Recomendado: los precios son asimétricos."),
        row(f"IQR extremo sobre log10 (k={k_extreme})", 10 ** lo_e, 10 ** hi_e, x < 10 ** lo_e, x > 10 ** hi_e,
            "Casos muy probablemente erróneos o excepcionales."),
        row(f"IQR sobre escala original (k={k})", lo_r, hi_r, x < lo_r, x > hi_r,
            "Referencia: tiende a marcar solo valores altos."),
        row(f"Z-score modificado (MAD) sobre log10 (|z|>{z_thresh})",
            10 ** (med - z_thresh * mad / 0.6745) if mad > 0 else np.nan,
            10 ** (med + z_thresh * mad / 0.6745) if mad > 0 else np.nan,
            (flags["VALOR_OUT_MAD"] & (lv < med)).reindex(x.index), (flags["VALOR_OUT_MAD"] & (lv > med)).reindex(x.index),
            "Robusto a valores extremos."),
        row("Rango plausible (regla de negocio)", min_plausible, max_plausible,
            (x < min_plausible) if min_plausible is not None else pd.Series(False, index=x.index),
            (x > max_plausible) if max_plausible is not None else pd.Series(False, index=x.index),
            "Parámetro configurable (MIN_PLAUSIBLE_VALUE / MAX_PLAUSIBLE_VALUE)."),
    ])
    return stats, methods, flags


def outlier_rows(df_enriched: pd.DataFrame, value_col: str = "VALOR", max_rows: int = 5000) -> pd.DataFrame:
    """Rows worth reviewing: flagged by >=2 methods, extreme, or outside the plausible range."""
    need = {"VALOR_N_METODOS", "VALOR_OUTLIER_EXTREMO", "VALOR_FUERA_RANGO_PLAUSIBLE"}
    if not need.issubset(df_enriched.columns):
        return pd.DataFrame()
    m = (df_enriched["VALOR_N_METODOS"] >= 2) | df_enriched["VALOR_OUTLIER_EXTREMO"] | df_enriched["VALOR_FUERA_RANGO_PLAUSIBLE"]
    keep = [c for c in ["PK", "MATRICULA", "YEAR_RADICA", "NOMBRE_NATUJUR", "TIENE_MAS_DE_UN_VALOR", value_col,
                        "VALOR_OUTLIER", "VALOR_OUTLIER_EXTREMO", "VALOR_N_METODOS", "VALOR_FUERA_RANGO_PLAUSIBLE",
                        "DOCUMENTO_JUSTIFICATIVO"] if c in df_enriched.columns]
    return df_enriched.loc[m, keep].sort_values(value_col, ascending=False).head(max_rows)


# --------------------------------------------------------------------------------------
# 6. Value categories
# --------------------------------------------------------------------------------------

def categorize_value(
    df: pd.DataFrame,
    value_col: str = "VALOR",
    n_categories: int = 4,
    method: str = "quantile",
    by: list[str] | str | None = None,
    fixed_bins: list[float] | None = None,
) -> pd.Series:
    """
    Build an ordered categorical column from VALOR.

    method="quantile": equal-volume categories (each holds ~1/n of the transactions). With n=4 these
                       are quartiles: Bajo / Medio bajo / Medio alto / Alto. Use `by=["YEAR_RADICA"]`
                       to compare each transaction against its own year (removes the price-inflation effect).
    method="fixed":    absolute COP thresholds (FIXED_BINS_COP or your own `fixed_bins`). Easier to
                       explain to the owners, but the thresholds must be agreed with them.
    Rows with null or non-positive VALOR get NaN (reported separately as 'Sin valor').
    """
    if n_categories not in CATEGORY_LABELS:
        raise ValueError("n_categories must be 3 or 4")
    labels = CATEGORY_LABELS[n_categories]
    v = _num(df[value_col])
    v = v.where(v > 0)

    if method == "quantile":
        if by:
            keys = [by] if isinstance(by, str) else list(by)
            pct = v.groupby([df[k] for k in keys]).rank(pct=True, method="average")
        else:
            pct = v.rank(pct=True, method="average")
        cat = pd.cut(pct, bins=np.linspace(0, 1, n_categories + 1), labels=labels, include_lowest=True)
    elif method == "fixed":
        bins = fixed_bins or FIXED_BINS_COP[n_categories]
        if len(bins) != n_categories + 1:
            raise ValueError(f"fixed_bins needs {n_categories + 1} edges")
        cat = pd.cut(v, bins=bins, labels=labels, include_lowest=True, right=True)
    else:
        raise ValueError("method must be 'quantile' or 'fixed'")
    return cat.astype(pd.CategoricalDtype(labels, ordered=True))


def category_summary(df: pd.DataFrame, cat_col: str = "VALOR_CATEGORIA", value_col: str = "VALOR") -> pd.DataFrame:
    """Volume (number of transactions) and value (COP) per category, plus observed value ranges."""
    v = _num(df[value_col])
    g = pd.DataFrame({"cat": df[cat_col], "v": v})
    out = (g.groupby("cat", observed=False)["v"]
           .agg(transacciones="size", valor_minimo="min", valor_mediana="median", valor_promedio="mean",
                valor_maximo="max", valor_total="sum").reset_index().rename(columns={"cat": "categoria"}))
    out["pct_transacciones"] = (100 * out["transacciones"] / max(out["transacciones"].sum(), 1)).round(2)
    out["pct_valor_total"] = (100 * out["valor_total"] / max(out["valor_total"].sum(), 1)).round(2)
    n_missing = int(df[cat_col].isna().sum())
    if n_missing:
        out = pd.concat([out, pd.DataFrame([{"categoria": "Sin valor", "transacciones": n_missing,
                                             "pct_transacciones": _pct(n_missing, len(df))}])], ignore_index=True)
    out["categoria"] = out["categoria"].astype(str)
    return out


def category_by_group(df: pd.DataFrame, group_col: str, cat_col: str = "VALOR_CATEGORIA") -> pd.DataFrame:
    """Transactions per category for each value of group_col (e.g. YEAR_RADICA, NOMBRE_NATUJUR)."""
    if group_col not in df.columns:
        return pd.DataFrame()
    return pd.crosstab(df[group_col], df[cat_col], dropna=False).reset_index()


# --------------------------------------------------------------------------------------
# 7. Orchestration + findings
# --------------------------------------------------------------------------------------

@dataclass
class QualityResults:
    df: pd.DataFrame                                   # input + new columns
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    duplicate_rows: dict[str, pd.DataFrame] = field(default_factory=dict)
    rule_masks: dict[str, pd.Series] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)


def build_findings(tables: dict[str, pd.DataFrame], n_rows: int) -> list[str]:
    out = [f"Registros analizados: {n_rows:,}."]
    comp = tables["completitud"]
    free = comp[~comp["columna"].isin(CONDITIONAL_COLUMNS)]
    bad = free[free["completitud_pct"] < 95]
    if bad.empty:
        out.append("Completitud: todas las columnas de diligenciamiento obligatorio superan el 95%.")
    else:
        out.append("Completitud: columnas por debajo del 95% -> " +
                   "; ".join(f"{r.columna} ({r.completitud_pct:.1f}%)" for r in bad.itertuples()) + ".")
    cond = comp[comp["columna"] == "VALOR"]
    if not cond.empty and pd.notna(cond["completitud_condicional_pct"].iloc[0]):
        out.append(f"VALOR: {cond['completitud_condicional_pct'].iloc[0]:.1f}% diligenciado en los actos que declaran valor "
                   f"(TIENE_VALOR = 1); {cond['completitud_pct'].iloc[0]:.1f}% sobre el total de registros.")
    sparse = comp[comp["columna"].isin(CONDITIONAL_COLUMNS) & (comp["columna"] != "VALOR") & (comp["completitud_pct"] < 50)]
    if not sparse.empty:
        out.append("Columnas de llenado condicional con baja completitud (confirmar con el dueño si es esperado): " +
                   "; ".join(f"{r.columna} ({r.completitud_pct:.1f}%)" for r in sparse.itertuples()) + ".")

    dup = tables["duplicados"]
    flagged = dup[dup["estado"] == "REVISAR"]
    out.append("Duplicados: " + ("sin hallazgos." if flagged.empty else
               "; ".join(f"{r.verificacion}: {int(r.filas_involucradas):,} filas ({r.pct_filas}%)" for r in flagged.itertuples()) + "."))

    rules = tables["reglas"]
    if not rules.empty:
        viol = rules[rules["filas_con_error"].fillna(0) > 0]
        crit = viol[viol["severidad"] == "Crítica"]
        out.append(f"Reglas de validez: {len(viol)} de {len(rules)} reglas con hallazgos; {len(crit)} críticas." +
                   ("" if crit.empty else " Críticas: " + "; ".join(f"{r.regla} ({int(r.filas_con_error):,})" for r in crit.itertuples()) + "."))

    om = tables.get("outliers_metodos")
    if om is not None and "metodo" in om.columns:
        p = om.iloc[0]
        out.append(f"Valores atípicos (IQR sobre log10): {int(p.n_bajos):,} bajos (< {p.limite_inferior:,.0f}) y "
                   f"{int(p.n_altos):,} altos (> {p.limite_superior:,.0f}) COP.")
    cs = tables.get("categorias")
    if cs is not None and not cs.empty:
        out.append("Categorías de valor -> " + "; ".join(
            f"{r.categoria}: {int(r.transacciones):,} transacciones" for r in cs.itertuples()) + ".")
    return out


def run_quality_checks(
    df: pd.DataFrame,
    value_col: str = "VALOR",
    n_categories: int = 4,
    category_method: str = "quantile",
    category_by: list[str] | str | None = None,
    fixed_bins: list[float] | None = None,
    export_path: str | None = None,
) -> QualityResults:
    """Run every check on an already-loaded DataFrame. See module docstring."""
    df = normalize_columns(df).copy()
    tables: dict[str, pd.DataFrame] = {}

    tables["esquema"] = schema_report(df)
    tables["completitud"] = completeness_report(df)
    tables["duplicados"], dup_rows = duplicates_report(df)
    tables["reglas"], masks = validity_report(df)

    enriched = df
    if value_col in df.columns:
        stats, methods, flags = outlier_analysis(df, value_col)
        tables["outliers_estadisticos"], tables["outliers_metodos"] = stats, methods
        enriched = pd.concat([df, flags], axis=1)
        enriched["VALOR_CATEGORIA"] = categorize_value(df, value_col, n_categories, category_method, category_by, fixed_bins)
        tables["outliers_filas"] = outlier_rows(enriched, value_col)
        tables["categorias"] = category_summary(enriched, "VALOR_CATEGORIA", value_col)
        for g in ["YEAR_RADICA", "NOMBRE_NATUJUR"]:
            t = category_by_group(enriched, g)
            if not t.empty:
                tables[f"categorias_x_{g.lower()}"] = t

    res = QualityResults(df=enriched, tables=tables, duplicate_rows=dup_rows, rule_masks=masks)
    res.findings = build_findings(tables, len(df))
    if export_path:
        export_report(res, export_path)
    return res


def export_report(res: QualityResults, path: str) -> None:
    """Write all tables to one Excel workbook that can be handed to the data owners."""
    sheet_names = {
        "esquema": "Esquema", "completitud": "Completitud", "duplicados": "Duplicados", "reglas": "Reglas_validez",
        "outliers_estadisticos": "Outliers_estadisticos", "outliers_metodos": "Outliers_metodos",
        "outliers_filas": "Outliers_filas", "categorias": "Categorias_valor",
        "categorias_x_year_radica": "Categorias_x_anio", "categorias_x_nombre_natujur": "Categorias_x_acto",
    }
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame({"Hallazgos principales": res.findings}).to_excel(xw, sheet_name="Resumen", index=False)
        for key, df_ in res.tables.items():
            if df_ is None or df_.empty:
                continue
            df_.to_excel(xw, sheet_name=sheet_names.get(key, key)[:31], index=False)
        for name, d in res.duplicate_rows.items():          # first 5,000 duplicate rows per check
            d.head(5000).to_excel(xw, sheet_name=("Dup_" + name)[:31].replace("/", "-").replace("+", "y"), index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in list(col)[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 70)
            ws.freeze_panes = "A2"
