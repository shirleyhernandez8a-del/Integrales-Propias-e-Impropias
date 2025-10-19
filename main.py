import streamlit as st
import sympy as sp
from sympy import limit, oo, Symbol, integrate, latex, lambdify, sqrt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import subprocess  # <-- agregado para comprobar java
import shlex       # <-- ayuda a parsear comandos seguro
import mpmath as mp  # respaldo numérico cuando SymPy no alcanza

# Inicializar session_state para gráfica persistente
if "show_graph" not in st.session_state:
    st.session_state.show_graph = False
if "saved_f" not in st.session_state:
    st.session_state.saved_f = ""
if "saved_a" not in st.session_state:
    st.session_state.saved_a = ""
if "saved_b" not in st.session_state:
    st.session_state.saved_b = ""

st.set_page_config(
    page_title="Solver de Integrales Impropias y Propias Detallado",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Estilos de fondo y contenedores (Tema Claro) */
    .main {background-color: #f0f8ff;}
    .stApp {background-color: #f0f8ff;}
    /* ---------------------------------------------------------------------- */
    /* FIX CONTRASTE GENERAL PARA EL MODO CLARO Y DARK MODE */
    /* ---------------------------------------------------------------------- */
    
    /* Estilos de encabezado y botones */
    h1 {color: #1e3a8a; text-align: center; font-family: 'Arial Black';}
    .stButton > button {background-color: #3b82f6; color: white; border-radius: 10px;}
    
    /* 1. CUERPO PRINCIPAL - Asegurar texto oscuro (#1e3a8a) sobre fondo claro */
    .stTextInput label, .stCheckbox label, 
    .stApp p, .stApp h2, .stApp h3 {
        color: #1e3a8a !important; 
    }
    .stTextInput input {
        color: #1e3a8a !important;
        background-color: white !important; 
    }
    .katex-display .base {
        color: #000000 !important; 
    }
    /* El texto de las alertas en el cuerpo principal debe ser oscuro para el contraste */
    .stApp .stAlert p, .stApp .stAlert h3, .stApp .stAlert * {
        color: #1e3a8a !important; 
    }

    /* 2. BARRA LATERAL - FIX CRÍTICO: Forzar el color a azul claro vibrante (#1E90FF)
        para asegurar visibilidad y consistencia con el cuerpo principal en dark mode. */
    .sidebar .sidebar-content h1, 
    .sidebar .sidebar-content h2, 
    .sidebar .sidebar-content h3, 
    .sidebar .sidebar-content p,
    .sidebar .sidebar-content label,
    /* Selector para el texto dentro de st.write y st.markdown */
    .sidebar .sidebar-content div[data-testid*="stMarkdownContainer"] * ,
    .sidebar .sidebar-content div[data-testid*="stHeader"] * ,
    .sidebar .sidebar-content div[data-testid*="stText"] *
    {
        /* Forzamos color azul claro vibrante para buena legibilidad */
        color: #1E90FF !important; 
    }
    /* Aseguramos también el contraste de las alertas en el sidebar */
    .sidebar .sidebar-content .stAlert p, 
    .sidebar .sidebar-content .stAlert h3, 
    .sidebar .sidebar-content .stAlert *
    {
        color: #1E90FF !important;
    }
    /* -------------------------------------------------------------------------------------- */
    </style>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<h1 style='text-align: center; color: #1e3a8a;'>🧮 Solver de Integrales Impropias y Propias - Paso a Paso Detallado</h1>",
    unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #4b5563; font-size: 18px;'>Ingresa la función y límites. La app explica **cada subpaso** antes de la respuesta final: tipo, cálculo de antiderivada, evaluación del límite y análisis de convergencia. ¡Visualiza el área y confirma la convergencia! 🎓</p>",
    unsafe_allow_html=True)
st.markdown("---")


def find_singularities(f, a_val, b_val, x):
    """
    Intenta encontrar puntos de singularidad reales dentro del intervalo (a, b)
    o en los límites (a o b). Devuelve una lista de singularidades encontradas
    (puede estar vacía).
    """
    sing_set = set()

    # Si la función no es simbólica, retornamos vacío
    try:
        f = sp.simplify(f)
    except Exception:
        return []

    # 1) Buscamos polos a partir del denominador si existe
    try:
        denom = sp.denom(f)
        if denom != 1:
            sols = sp.solveset(sp.Eq(denom, 0), x, domain=sp.S.Reals)
            for s in sols:
                try:
                    if s.is_real:
                        sing_set.add(sp.simplify(s))
                except Exception:
                    pass
    except Exception:
        pass

    # 2) Buscamos puntos donde la función no está definida (zoo, nan)
    try:
        problematic = sp.solveset(sp.Eq(sp.sympify(f).as_numer_denom()[0], sp.nan), x, domain=sp.S.Reals)
        # este paso es menos fiable; lo dejamos como intento
    except Exception:
        pass

    # 3) Revisar raíces de radicandos en potencias fraccionarias de denominador par (ej sqrt)
    try:
        # detecta sqrt(...) y otros exponentes fraccionarios con denominador par
        for sub in sp.preorder_traversal(f):
            if isinstance(sub, sp.Pow):
                exp = sub.args[1]
                base = sub.args[0]
                if exp.is_Rational and exp.q % 2 == 0:
                    # base >= 0 requerido
                    sols = sp.solveset(sp.Eq(base, 0), x, domain=sp.S.Reals)
                    for s in sols:
                        sing_set.add(sp.simplify(s))
    except Exception:
        pass

    # 4) Filtrar por el intervalo si a_val y b_val son finitos
    filtered = []
    try:
        a = float(a_val) if a_val != -oo and a_val != oo and getattr(a_val, "is_number", False) else None
    except Exception:
        a = None
    try:
        b = float(b_val) if b_val != -oo and b_val != oo and getattr(b_val, "is_number", False) else None
    except Exception:
        b = None

    for s in sing_set:
        try:
            sval = float(s)
            if (a is None or sval >= a) and (b is None or sval <= b):
                filtered.append(sp.simplify(s))
        except Exception:
            # si no se puede convertir a float, solo incluir si a/b son simbólicos
            filtered.append(sp.simplify(s))

    # eliminar duplicados y ordenar
    unique = sorted(list(set(filtered)), key=lambda z: float(z) if getattr(z, "is_number", False) else 0)
    return unique


def check_for_singularities_mode(f, a_val, b_val, x):
    """
    Verifica si la integral es impropia debido a límites infinitos o singularidades.
    Retorna el modo ('proper', 'infinite_upper', etc.) y una singularidad si procede.
    """
    # Detectar límites infinitos primero
    if a_val == -oo and b_val == oo:
        return "infinite_both", None
    if b_val == oo:
        return "infinite_upper", None
    if a_val == -oo:
        return "infinite_lower", None

    # Buscar singularidades dentro del intervalo o en los extremos
    singulars = find_singularities(f, a_val, b_val, x)

    if len(singulars) == 0:
        return "proper", None

    # Si hay singularidades múltiples, preferimos marcar internal_singular si están dentro
    for s in singulars:
        try:
            if s == a_val:
                return "singular_lower", s
            if s == b_val:
                return "singular_upper", s
            # Si está estrictamente entre a y b
            if getattr(a_val, "is_number", False) and getattr(b_val, "is_number", False):
                if float(a_val) < float(s) < float(b_val):
                    return "internal_singular", s
        except Exception:
            # si conversión falla, intentamos comparar simbólicamente
            try:
                if sp.Lt(a_val, s) and sp.Lt(s, b_val):
                    return "internal_singular", s
            except Exception:
                pass

    # si ninguna condición anterior, devolvemos singular internal por defecto en el primero
    return "internal_singular", singulars[0] if singulars else None


def clean_divergence_result(result):
    """
    Limpia resultado de SymPy si es infinito o confuso.
    """
    if result is oo:
        return oo
    if result is -oo:
        return -oo
    # SymPy a veces devuelve zoo o expresiones con -oo incrustado
    try:
        if getattr(result, "is_infinite", False):
            s = str(result)
            if s.startswith('-'):
                return -oo
            return oo
    except Exception:
        pass
    if result is sp.nan:
        return sp.nan
    return result


def numeric_integral_backup(f_sym, a_val, b_val, x_sym):
    """
    Intento numérico robusto con mpmath en caso SymPy no entregue resultado simbólico.
    Devuelve (value, converges_bool). Para integrales impropias, intentamos dividir
    en sub-intervalos y usar quad con límites apropiados.
    """
    # Convertir a función mpmath (mediante lambdify con 'mpmath')
    try:
        f_mp = sp.lambdify(x_sym, f_sym, modules=['mpmath'])
    except Exception:
        try:
            f_mp = sp.lambdify(x_sym, f_sym, modules=['numpy', 'mpmath'])
        except Exception:
            return (None, False)

    mp.mp.dps = 50  # precisión alta

    # Manejo de límites infinitos y singularidades simples:
    try:
        # Con mpmath, usar mp.quad con límites adecuados
        if a_val == -oo and b_val == oo:
            # dividir en (-inf,0) y (0,inf) (si 0 no es singular)
            try:
                val1 = mp.quad(f_mp, [-mp.inf, 0])
                val2 = mp.quad(f_mp, [0, mp.inf])
                return (val1 + val2, True)
            except Exception:
                # Intentar dividir en -A..A y aumentar A hasta convergencia (poco robusto pero intentamos)
                try:
                    for A in [10, 50, 200, 1000]:
                        try:
                            val = mp.quad(f_mp, [-A, A])
                            return (val, True)
                        except Exception:
                            continue
                except Exception:
                    return (None, False)
        elif b_val == oo:
            try:
                val = mp.quad(f_mp, [float(a_val), mp.inf])
                return (val, True)
            except Exception:
                # intentar aumentar límite superior finito
                try:
                    for B in [10, 50, 200, 1000]:
                        try:
                            val = mp.quad(f_mp, [float(a_val), B])
                            return (val, True)
                        except Exception:
                            continue
                except Exception:
                    return (None, False)
        elif a_val == -oo:
            try:
                val = mp.quad(f_mp, [-mp.inf, float(b_val)])
                return (val, True)
            except Exception:
                try:
                    for A in [10, 50, 200, 1000]:
                        try:
                            val = mp.quad(f_mp, [-A, float(b_val)])
                            return (val, True)
                        except Exception:
                            continue
                except Exception:
                    return (None, False)
        else:
            # límites finitos normales (podemos intentar detectar singularidades internas)
            try:
                val = mp.quad(f_mp, [float(a_val), float(b_val)])
                return (val, True)
            except Exception:
                # si falla, intentar dividir en subintervalos evitando puntos singulares
                return (None, False)
    except Exception:
        return (None, False)


def resolver_integral(f_str, a_str, b_str, var='x'):
    try:
        x = Symbol(var)
        # Reemplazos seguros
        f_str_sympify = f_str.replace('E', 'exp(1)').replace('sqrt(', 'sqrt(')

        # Validación temprana y mensaje amigable si falla sympify
        try:
            f = sp.sympify(f_str_sympify)
        except Exception as e:
            st.error(f"Entrada inválida para f(x): {e}. Ejemplos válidos: x**2, 1/x**2, sqrt(x), exp(x), log(x).")
            return

        # parse límites
        try:
            a = sp.sympify(a_str)
        except Exception:
            st.error("Entrada inválida para límite inferior 'a'. Usa números o 'oo'/'-oo'.")
            return
        try:
            b = sp.sympify(b_str)
        except Exception:
            st.error("Entrada inválida para límite superior 'b'. Usa números o 'oo'/'-oo'.")
            return

        # Mostrar header de análisis
        st.subheader("📊 Análisis Completo Paso a Paso")

        # Variables auxiliares
        lim_val_1_display = None
        lim_val_2_display = None
        final_res_step_by_step = None
        numeric_backup_used = False

        # Tipo de integral
        mode, c = check_for_singularities_mode(f, a, b, x)
        analysis_notes = []

        st.write("**Paso 1: Identificación del Tipo de Integral**")

        c_latex = latex(c) if c is not None else "c"

        if mode == "internal_singular":
            analysis_notes.append(f"Esta es una integral impropia por **singularidad interna** (discontinuidad en $c={c_latex}$), donde ${latex(a)} < {c_latex} < {latex(b)}$.")
            analysis_notes.append("Se debe dividir en dos integrales impropias:")
            st.latex(r"\int_{" + latex(a) + "}^{" + latex(b) + r"} f(x) dx = \lim_{t_1 \to " + c_latex + r"^-} \int_{" + latex(a) + "}^{t_1} f(x) dx + \lim_{t_2 \to " + c_latex + r"^+} \int_{t_2}^{" + latex(b) + r"} f(x) dx")
            analysis_notes.append("Si una de las dos partes diverge, la integral completa **DIVERGE**.")
        elif mode == "infinite_both":
            analysis_notes.append("Esta es una integral impropia por **límite infinito doble** ($-\infty$ a $\infty$).")
            analysis_notes.append("Se resuelve dividiendo en dos integrales en un punto arbitrario $c$ (usamos $c=0$ por simplicidad):")
            st.latex(r"\int_{-\infty}^{\infty} f(x) \, dx = \lim_{t_1 \to -\infty} \int_{t_1}^{0} f(x) \, dx + \lim_{t_2 \to \infty} \int_{0}^{t_2} f(x) \, dx")
            analysis_notes.append("Si una de las dos partes diverge, la integral completa **DIVERGE**.")
        elif mode == "infinite_upper":
            analysis_notes.append("Esta es una integral impropia por **límite infinito superior**. Se resuelve como:")
            st.latex(r"\int_{" + latex(a) + r"}^\infty f(x) \, dx = \lim_{t \to \infty} \int_{" + latex(a) + r"}^t f(x) \, dx")
            analysis_notes.append("Se evaluará $F(t)-F(a)$ y se tomará el límite $t \to \infty$.")
        elif mode == "infinite_lower":
            analysis_notes.append("Esta es una integral impropia por **límite infinito inferior**. Se resuelve como:")
            st.latex(r"\int_{-\infty}^{" + latex(b) + r"} f(x) \, dx = \lim_{t \to -\infty} \int_t^{" + latex(b) + r"} f(x) \, dx")
            analysis_notes.append("Se evaluará $F(b)-F(t)$ y se tomará el límite $t \to -\infty$.")
        elif mode == "singular_lower":
            analysis_notes.append(f"Esta es una integral impropia por **singularidad en el límite inferior** (discontinuidad en $a={latex(a)}$). Se resuelve como:")
            st.latex(r"\int_{" + latex(a) + "}^{" + latex(b) + r"} f(x) \, dx = \lim_{\epsilon \to " + latex(a) + r"^{+}} \int_{\epsilon}^{" + latex(b) + r"} f(x) \, dx")
            analysis_notes.append("Se evaluará $F(b)-F(\epsilon)$ y se tomará el límite $\epsilon \to " + latex(a) + r"^{+}$.")
        elif mode == "singular_upper":
            analysis_notes.append(f"Esta es una integral impropia por **singularidad en el límite superior** (discontinuidad en $b={latex(b)}$). Se resuelve como:")
            st.latex(r"\int_{" + latex(a) + "}^{" + latex(b) + r"} f(x) \, dx = \lim_{\epsilon \to " + latex(b) + r"^{-}} \int_{" + latex(a) + r"}^{\epsilon} f(x) \, dx")
            analysis_notes.append("Se evaluará $F(\epsilon)-F(a)$ y se tomará el límite $\epsilon \to " + latex(b) + r"^{-}$.")
        else:
            analysis_notes.append("Esta es una **integral propia** (límites finitos y función continua en el intervalo de integración). Se calcula $F(b) - F(a)$ directamente.")

        for note in analysis_notes:
            st.markdown(note)

        st.write("**Función dada**:")
        st.latex(f"f(x) = {latex(f)}")
        st.write(f"**Límites de Integración**: de ${latex(a)}$ a ${latex(b)}$")

        # --- APLICACIÓN: Encontrar la antiderivada simbólica (si es posible) ---
        try:
            F = sp.integrate(f, x)
            st.write("**Paso 2: Encontrar la Antiderivada Indefinida $F(x)$**")
            st.latex(r"\int f(x) dx = F(x) = " + latex(F) + r" + C")
            st.markdown(f"**Nota**: En la integral definida, la constante $C$ se cancela.")
        except Exception:
            # Si falla, aún continuamos con el análisis usando límites o respaldo numérico
            F = None
            st.warning("SymPy no pudo calcular la antiderivada simbólicamente. Continuaremos con límites y/o evaluación numérica de respaldo.")

        # --- EVALUACIÓN DE LÍMITES Y PARTES SEGÚN MODO ---
        t = Symbol('t')
        epsilon = Symbol('epsilon')
        lim_val = None

        st.write("**Paso 3 & 4: Evaluación y Cálculo Explícito del Límite**")

        # Función auxiliar para calcular límite simbólico o numérico de forma segura
        def safe_limit(expr, var_sym, point, dir=None):
            try:
                if dir is None:
                    return limit(expr, var_sym, point)
                else:
                    return limit(expr, var_sym, point, dir=dir)
            except Exception:
                # Intento de evaluación numérica si falla el límite simbólico
                try:
                    if point == oo:
                        # evaluar expr para valores grandes
                        f_num = sp.lambdify(var_sym, expr, 'mpmath')
                        for R in [1e2, 1e3, 1e4]:
                            try:
                                v = f_num(R)
                                if mp.isfinite(v):
                                    return mp.mpf(v)
                            except Exception:
                                continue
                    elif point == -oo:
                        f_num = sp.lambdify(var_sym, expr, 'mpmath')
                        for R in [-1e2, -1e3, -1e4]:
                            try:
                                v = f_num(R)
                                if mp.isfinite(v):
                                    return mp.mpf(v)
                            except Exception:
                                continue
                    else:
                        f_num = sp.lambdify(var_sym, expr, 'mpmath')
                        # evaluar cercano desde ambos lados si es finito
                        for delta in [1e-6, 1e-4, 1e-2]:
                            try:
                                v1 = f_num(float(point) - delta)
                                v2 = f_num(float(point) + delta)
                                if mp.isfinite(v1) and mp.isfinite(v2) and abs(v1 - v2) < 1e-6:
                                    return mp.mpf((v1 + v2) / 2)
                            except Exception:
                                continue
                except Exception:
                    pass
                return sp.nan

        if mode == "proper":
            if F is not None:
                F_b = F.subs(x, b)
                F_a = F.subs(x, a)
                expr = F_b - F_a
                st.markdown(r"Aplicamos el Teorema Fundamental del Cálculo:")
                st.latex(r"\int_{" + latex(a) + "}^{" + latex(b) + r"} f(x) \, dx = F(" + latex(b) + ") - F(" + latex(a) + ")")
                st.latex(r"= \left[ " + latex(F_b) + r" \right] - \left[ " + latex(F_a) + r" \right]")
                final_res_step_by_step = sp.simplify(expr)
                st.latex(r"= " + latex(final_res_step_by_step))
            else:
                # Si no hay antiderivada simbólica, usamos backup numérico
                num_val, conv_flag = numeric_integral_backup(f, a, b, x)
                if conv_flag:
                    numeric_backup_used = True
                    final_res_step_by_step = mp.mpf(num_val)
                    st.markdown("Se usó evaluación numérica de respaldo (mpmath) para la integral propia.")
                    st.latex(r"\text{Valor numérico aproximado: } " + latex(sp.N(final_res_step_by_step)))
                else:
                    final_res_step_by_step = sp.nan

        elif mode == "infinite_upper":
            if F is not None:
                expr_t_a = F.subs(x, t) - F.subs(x, a)
                lim_val = safe_limit(expr_t_a, t, oo)
                final_res_step_by_step = lim_val
                st.markdown(r"Sustituimos el límite superior infinito con $t$:") 
                st.latex(r"\lim_{t \to \infty} \left[ " + latex(sp.simplify(expr_t_a)) + r" \right] = " + latex(clean_divergence_result(lim_val)))
            else:
                # Respaldo numérico
                num_val, conv_flag = numeric_integral_backup(f, a, oo, x)
                if conv_flag:
                    numeric_backup_used = True
                    final_res_step_by_step = mp.mpf(num_val)
                    st.markdown("Se usó evaluación numérica de respaldo (mpmath) para la integral impropia (cima infinita).")
                    st.latex(r"\text{Valor numérico aproximado: } " + latex(sp.N(final_res_step_by_step)))
                else:
                    final_res_step_by_step = sp.nan

        elif mode == "infinite_lower":
            if F is not None:
                expr_b_t = F.subs(x, b) - F.subs(x, t)
                lim_val = safe_limit(expr_b_t, t, -oo)
                final_res_step_by_step = lim_val
                st.markdown(r"Sustituimos el límite inferior infinito con $t$:") 
                st.latex(r"\lim_{t \to -\infty} \left[ " + latex(sp.simplify(expr_b_t)) + r" \right] = " + latex(clean_divergence_result(lim_val)))
            else:
                num_val, conv_flag = numeric_integral_backup(f, -oo, b, x)
                if conv_flag:
                    numeric_backup_used = True
                    final_res_step_by_step = mp.mpf(num_val)
                    st.markdown("Se usó evaluación numérica de respaldo (mpmath) para la integral impropia (inferior infinito).")
                    st.latex(r"\text{Valor numérico aproximado: } " + latex(sp.N(final_res_step_by_step)))
                else:
                    final_res_step_by_step = sp.nan

        elif mode == "singular_lower":
            if F is not None:
                expr_b_eps = F.subs(x, b) - F.subs(x, epsilon)
                lim_val = safe_limit(expr_b_eps, epsilon, a, dir='+')
                final_res_step_by_step = lim_val
                st.markdown(r"Sustituimos el límite inferior singular con $\epsilon$ y tomamos el límite lateral $\epsilon \to a^{+}$:") 
                st.latex(r"\lim_{\epsilon \to " + latex(a) + r"^{+}} \left[ " + latex(sp.simplify(expr_b_eps)) + r" \right] = " + latex(clean_divergence_result(lim_val)))
            else:
                # intentar integral numérica evitando extremo singular
                try:
                    a_num = float(a)
                    b_num = float(b)
                    # integrar desde a+delta a b
                    for delta in [1e-6, 1e-4, 1e-2]:
                        try:
                            f_mp = sp.lambdify(x, f, 'mpmath')
                            val = mp.quad(f_mp, [a_num + delta, b_num])
                            final_res_step_by_step = val
                            numeric_backup_used = True
                            break
                        except Exception:
                            continue
                except Exception:
                    final_res_step_by_step = sp.nan

        elif mode == "singular_upper":
            if F is not None:
                expr_eps_a = F.subs(x, epsilon) - F.subs(x, a)
                lim_val = safe_limit(expr_eps_a, epsilon, b, dir='-')
                final_res_step_by_step = lim_val
                st.markdown(r"Sustituimos el límite superior singular con $\epsilon$ y tomamos el límite lateral $\epsilon \to b^{-}$:") 
                st.latex(r"\lim_{\epsilon \to " + latex(b) + r"^{-}} \left[ F(" + latex(b) + r") - F(\epsilon) \right] = " + latex(clean_divergence_result(lim_val)))
            else:
                try:
                    a_num = float(a)
                    b_num = float(b)
                    for delta in [1e-6, 1e-4, 1e-2]:
                        try:
                            f_mp = sp.lambdify(x, f, 'mpmath')
                            val = mp.quad(f_mp, [a_num, b_num - delta])
                            final_res_step_by_step = val
                            numeric_backup_used = True
                            break
                        except Exception:
                            continue
                except Exception:
                    final_res_step_by_step = sp.nan

        elif mode == "internal_singular":
            # dividir en 2 con c
            t1, t2 = Symbol('t1'), Symbol('t2')
            c_val = c if c is not None else 0

            if F is not None:
                # Parte 1
                F_c1 = F.subs(x, t1) - F.subs(x, a)
                st.markdown("**Desarrollo detallado de la Parte 1**:")
                st.latex(r"\int_{" + latex(a) + "}^{" + latex(c_val) + r"} f(x)\,dx = F\left(" + latex(c_val) + r"^{-}\right) - F\left(" + latex(a) + r"\right)")
                st.latex(r"= \lim_{t_1 \to " + latex(c_val) + "^{-}} \left(" + latex(F.subs(x, t1)) + " - " + latex(F.subs(x, a)) + r"\right)")
                lim_val_1 = safe_limit(F_c1, t1, c_val, dir='-')
                lim_val_1_display = clean_divergence_result(lim_val_1)

                st.markdown(f"**Parte 1: Límite de $\\int_{{{latex(a)}}}^{{{latex(c_val)}}} f(x) dx$ (límite izquierdo)**")
                st.latex(r"\lim_{t_1 \to " + (latex(c_val) if c is not None else str(c_val)) + r"^-} \left[ F(t_1) - F(" + latex(a) + r") \right] = " + latex(lim_val_1_display))

                # Parte 2
                F_c2 = F.subs(x, b) - F.subs(x, t2)
                st.markdown("**Desarrollo detallado de la Parte 2**:")
                st.latex(r"\int_{" + latex(c_val) + "}^{" + latex(b) + r"} f(x)\,dx = F\left(" + latex(b) + r"\right) - F\left(" + latex(c_val) + r"^{+}\right)")
                st.latex(r"= \lim_{t_2 \to " + latex(c_val) + "^{+}} \left(" + latex(F.subs(x, b)) + " - " + latex(F.subs(x, t2)) + r"\right)")
                lim_val_2 = safe_limit(F_c2, t2, c_val, dir='+')
                lim_val_2_display = clean_divergence_result(lim_val_2)
                st.markdown(f"**Parte 2: Límite de $\\int_{{{latex(c_val)}}}^{{{latex(b)}}} f(x) dx$ (límite derecho)**")
                st.latex(r"\lim_{t_2 \to " + (latex(c_val) if c is not None else str(c_val)) + r"^{+}} \left[ F(" + latex(b) + r") - F(t_2) \right] = " + latex(lim_val_2_display))

                # evaluar convergencia
                is_div_1 = (getattr(lim_val_1_display, "is_infinite", False) or lim_val_1_display is sp.nan)
                is_div_2 = (getattr(lim_val_2_display, "is_infinite", False) or lim_val_2_display is sp.nan)

                if is_div_1 or is_div_2:
                    if is_div_1:
                        final_res_step_by_step = lim_val_1
                    if is_div_2:
                        final_res_step_by_step = lim_val_2
                else:
                    final_res_step_by_step = lim_val_1 + lim_val_2
            else:
                # Si F no pudo obtenerse, intentamos respaldo numérico dividiendo en c
                try:
                    a_num = float(a)
                    b_num = float(b)
                    c_num = float(c_val)
                    num1, conv1 = numeric_integral_backup(f, a_num, c_num, x)
                    num2, conv2 = numeric_integral_backup(f, c_num, b_num, x)
                    if conv1 and conv2:
                        numeric_backup_used = True
                        final_res_step_by_step = mp.mpf(num1) + mp.mpf(num2)
                    else:
                        final_res_step_by_step = sp.nan
                except Exception:
                    final_res_step_by_step = sp.nan

        # Si no fue cubierto, usamos lim_val simple
        if final_res_step_by_step is None:
            final_res_step_by_step = lim_val

        final_res_clean = clean_divergence_result(final_res_step_by_step)

        # --- ANÁLISIS DE CONVERGENCIA FINAL ---
        try:
            # Intento simbólico directo (mayor confianza)
            res_full = integrate(f, (x, a, b))
        except Exception:
            res_full = sp.nan
            st.warning("SymPy no pudo computar la integral simbólica completa. Se intentó una aproximación numérica si fue posible.")

        st.write("**Paso 5: Análisis de Convergencia (Conclusión Final)**")

        is_finite = False
        try:
            # Si res_full es numérico o simbólico finito
            is_finite = res_full.is_finite
        except Exception:
            try:
                if str(res_full).lower() not in ['oo', 'zoo', 'nan', 'infinity'] and sp.N(res_full).is_real:
                    is_finite = True
            except Exception:
                is_finite = False

        # Si SymPy indica finito pero el cálculo por límites da infinito, priorizamos límites
        if is_finite:
            # Comprobación especial para singularidad interna donde límites laterales divergen
            if mode == "internal_singular" and ((isinstance(lim_val_1_display, sp.Expr) and getattr(lim_val_1_display, "is_infinite", False)) or (isinstance(lim_val_2_display, sp.Expr) and getattr(lim_val_2_display, "is_infinite", False))):
                 st.error("❌ **La integral DIVERGE** (no converge).")
                 try:
                     st.write(f"**Aclaración Importante**: Uno o ambos límites laterales resultaron en $\\pm \\infty$ (Parte 1: ${latex(lim_val_1_display)}$, Parte 2: ${latex(lim_val_2_display)}$). Aunque SymPy pueda devolver un valor principal de Cauchy, la integral es DIVERGENTE porque no existe la suma de las partes.")
                 except Exception:
                     pass
            else:
                # Informe de convergencia
                if numeric_backup_used:
                    st.success(f"✅ **La integral CONVERGE**. Valor numérico aproximado (respaldo): ${sp.N(final_res_clean)}$.")
                else:
                    st.success(f"✅ **La integral CONVERGE** a un valor finito: ${latex(res_full)}$.")
                st.write(f"**Explicación detallada**: El límite o la suma de límites es finito (${latex(final_res_step_by_step)}$).")
                st.success("✅ ¡Cálculo completado exitosamente! La integral converge.", icon="🎯")
                st.info("Revisa los pasos 3 y 4 para ver el proceso matemático completo.")
                st.markdown("""
                <div id="confetti-holder"></div>
                <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
                <script>
                const duration = 3 * 1000;
                const end = Date.now() + duration;
                (function frame() {
                    confetti({ particleCount: 5, angle: 60, spread: 55, origin: { x: 0 }, colors: ['#3b82f6', '#60a5fa', '#93c5fd'] });
                    confetti({ particleCount: 5, angle: 120, spread: 55, origin: { x: 1 }, colors: ['#3b82f6', '#60a5fa', '#93c5fd'] });
                    if (Date.now() < end) {
                        requestAnimationFrame(frame);
                    }
                }());
                </script>
                """, unsafe_allow_html=True)
        else:
            st.error("❌ **La integral DIVERGE** (no converge).")

            if mode == "internal_singular":
                try:
                    st.markdown(f"**Resultado de la Parte 1 (Límite Izquierdo)**: ${latex(lim_val_1_display)}$")
                    st.markdown(f"**Resultado de la Parte 2 (Límite Derecho)**: ${latex(lim_val_2_display)}$")
                except Exception:
                    pass
                if (isinstance(lim_val_1_display, sp.Expr) and getattr(lim_val_1_display, "is_infinite", False)) and (isinstance(lim_val_2_display, sp.Expr) and getattr(lim_val_2_display, "is_infinite", False)) and (str(lim_val_1_display).startswith('-') and not str(lim_val_2_display).startswith('-')):
                     st.info("⚠️ **Aclaración de Signos**: La integral diverge porque el límite izquierdo es $\\mathbf{-\\infty}$ y el límite derecho es $\\mathbf{+\\infty}$.")
                else:
                    st.write(f"El resultado final muestra divergencia: ${latex(final_res_clean)}$")
            elif mode == "infinite_both":
                try:
                    st.markdown(f"**Resultado de la Parte 1 ($-\infty$ a $0$)**: ${latex(lim_val_1_display)}$")
                    st.markdown(f"**Resultado de la Parte 2 ($0$ a $\\infty$)**: ${latex(lim_val_2_display)}$")
                except Exception:
                    pass
                st.write(f"El resultado del límite divergente fue: ${latex(final_res_clean)}$")
            else:
                st.write(f"El resultado del límite es: ${latex(final_res_clean)}$")

            st.write("**Explicación detallada**: El límite (o la suma de límites) resultó en $\\pm \\infty$ o no existe; por tanto la integral diverge.")

    except Exception as e:
        st.error(
            f"❌ Error en el cálculo: {str(e)}. Tips: Usa **'x'** como variable, **`**` para potencias (ej. x**2), **x**(1/3) para $\\sqrt[3]{x}$, **sqrt(x)** para $\\sqrt{x}$, **oo** para $\\infty$, **log()** para $\\ln()$, **exp(x)** para $e^x$ o **E** para $e$. Ejemplo: 1/sqrt(1+x)."
        )


with st.sidebar:
    # Cambié solo el render del header para forzar color legible (azul claro vibrante #1E90FF)
    st.markdown("<h2 style='color:#1E90FF; margin-bottom:0.2rem;'>⚙️ Configuración y Ayuda</h2>", unsafe_allow_html=True)
    # Cambié solo esto para que la guía tenga color forzado y sea legible en dark mode
    st.markdown("<h3 style='color:#1E90FF; margin-top:0.5rem;'>📝 Guía de Sintaxis</h3>", unsafe_allow_html=True)
    st.write(
        "- **f(x)**: La función debe usar **x** como variable (ej. `1/x**2`)."
    )
    st.write("- **a / b**: Límite inferior/superior.")
    st.write("- **Potencias**: Usa **`**` (ej. `x**2`).")
    # Instrucción de raíces actualizada para claridad.
    st.write("- **Raíces**: Usa **sqrt(x)** para $\\sqrt{x}$ o potencias fraccionarias (ej. `x**(1/3)` para $\\sqrt[3]{x}$).")
    st.write("- **Infinito**: Usa **oo** para $+\\infty$ o **-oo** para $-\\infty$.")
    st.write("- **Funciones**: Usa **log(x)** para $\\ln(x)$, **exp(x)** para $e^x$, y **E** para la constante de Euler.")

    st.markdown(
        f"""
        <div style='background-color:#eef2ff; color:#1e3a8a; padding:10px; border-radius:8px; font-weight:600;'>
        💡 <strong>Nota del Desarrollador</strong>
        <br>1. El sistema identifica si es **Propia** o **Impropia** (y el tipo).
        <br>2. Primero se calcula la **Antiderivada** **F(x)**.
        <br>3. Luego se aplica el **Límite** correspondiente (a **
