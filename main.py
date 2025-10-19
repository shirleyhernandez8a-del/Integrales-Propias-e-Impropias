import streamlit as st
import sympy as sp
from sympy import limit, oo, Symbol, integrate, latex, lambdify, sqrt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import shlex
import mpmath as mp

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
    .main {background-color: #f0f8ff;}
    .stApp {background-color: #f0f8ff;}
    h1 {color: #1e3a8a; text-align: center; font-family: 'Arial Black';}
    .stButton > button {background-color: #3b82f6; color: white; border-radius: 10px;}
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
    .stApp .stAlert p, .stApp .stAlert h3, .stApp .stAlert * {
        color: #1e3a8a !important; 
    }
    .sidebar .sidebar-content h1, 
    .sidebar .sidebar-content h2, 
    .sidebar .sidebar-content h3, 
    .sidebar .sidebar-content p,
    .sidebar .sidebar-content label,
    .sidebar .sidebar-content div[data-testid*="stMarkdownContainer"] * ,
    .sidebar .sidebar-content div[data-testid*="stHeader"] * ,
    .sidebar .sidebar-content div[data-testid*="stText"] *
    {
        color: #1E90FF !important; 
    }
    .sidebar .sidebar-content .stAlert p, 
    .sidebar .sidebar-content .stAlert h3, 
    .sidebar .sidebar-content .stAlert *
    {
        color: #1E90FF !important;
    }
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

    try:
        f = sp.simplify(f)
    except Exception:
        return []

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

    try:
        for sub in sp.preorder_traversal(f):
            if isinstance(sub, sp.Pow):
                exp = sub.args[1]
                base = sub.args[0]
                if exp.is_Rational and exp.q % 2 == 0:
                    sols = sp.solveset(sp.Eq(base, 0), x, domain=sp.S.Reals)
                    for s in sols:
                        sing_set.add(sp.simplify(s))
    except Exception:
        pass

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
            filtered.append(sp.simplify(s))

    unique = sorted(list(set(filtered)), key=lambda z: float(z) if getattr(z, "is_number", False) else 0)
    return unique


def check_for_singularities_mode(f, a_val, b_val, x):
    if a_val == -oo and b_val == oo:
        return "infinite_both", None
    if b_val == oo:
        return "infinite_upper", None
    if a_val == -oo:
        return "infinite_lower", None

    singulars = find_singularities(f, a_val, b_val, x)

    if len(singulars) == 0:
        return "proper", None

    for s in singulars:
        try:
            if s == a_val:
                return "singular_lower", s
            if s == b_val:
                return "singular_upper", s
            if getattr(a_val, "is_number", False) and getattr(b_val, "is_number", False):
                if float(a_val) < float(s) < float(b_val):
                    return "internal_singular", s
        except Exception:
            try:
                if sp.Lt(a_val, s) and sp.Lt(s, b_val):
                    return "internal_singular", s
            except Exception:
                pass

    return "internal_singular", singulars[0] if singulars else None


def clean_divergence_result(result):
    if result is oo:
        return oo
    if result is -oo:
        return -oo
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
    try:
        f_mp = sp.lambdify(x_sym, f_sym, modules=['mpmath'])
    except Exception:
        try:
            f_mp = sp.lambdify(x_sym, f_sym, modules=['numpy', 'mpmath'])
        except Exception:
            return (None, False)

    mp.mp.dps = 50

    try:
        if a_val == -oo and b_val == oo:
            try:
                val1 = mp.quad(f_mp, [-mp.inf, 0])
                val2 = mp.quad(f_mp, [0, mp.inf])
                return (val1 + val2, True)
            except Exception:
                return (None, False)
        elif b_val == oo:
            try:
                val = mp.quad(f_mp, [float(a_val), mp.inf])
                return (val, True)
            except Exception:
                return (None, False)
        elif a_val == -oo:
            try:
                val = mp.quad(f_mp, [-mp.inf, float(b_val)])
                return (val, True)
            except Exception:
                return (None, False)
        else:
            try:
                val = mp.quad(f_mp, [float(a_val), float(b_val)])
                return (val, True)
            except Exception:
                return (None, False)
    except Exception:
        return (None, False)


def resolver_integral(f_str, a_str, b_str, var='x'):
    try:
        x = Symbol(var)
        f_str_sympify = f_str.replace('E', 'exp(1)').replace('sqrt(', 'sqrt(')

        try:
            f = sp.sympify(f_str_sympify)
        except Exception as e:
            st.error(f"Entrada inválida para f(x): {e}. Ejemplos válidos: x**2, 1/x**2, sqrt(x), exp(x), log(x).")
            return

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

        # ============ VALIDACIÓN DEL DOMINIO (ADVERTENCIA, NO BLOQUEO) ============
        domain_warning = False
        domain_message = ""
        
        # Verificar raíces de índice par (como sqrt) que pueden generar valores complejos
        for sub in sp.preorder_traversal(f):
            if isinstance(sub, sp.Pow):
                exp = sub.args[1]
                base = sub.args[0]
                # Si tiene exponente fraccionario con denominador par (raíz par)
                if exp.is_Rational and exp.q % 2 == 0:
                    # Verificar si el intervalo incluye valores donde base < 0
                    if a != -oo and a.is_number and float(a) < 0:
                        try:
                            test_val = base.subs(x, a)
                            if test_val.is_number and float(test_val) < 0:
                                domain_warning = True
                                domain_message = f"⚠️ **Nota sobre el Dominio**: La función contiene una raíz de índice par que puede generar **valores complejos** en parte del intervalo (x < 0). SymPy trabajará con números complejos si es necesario."
                                break
                        except:
                            pass
        
        if domain_warning:
            st.warning(domain_message)
        # ===========================================================================

        st.subheader("📊 Análisis Completo Paso a Paso")

        lim_val_1_display = None
        lim_val_2_display = None
        final_res_step_by_step = None
        numeric_backup_used = False

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

        try:
            F = sp.integrate(f, x)
            st.write("**Paso 2: Encontrar la Antiderivada Indefinida $F(x)$**")
            st.latex(r"\int f(x) dx = F(x) = " + latex(F) + r" + C")
            st.markdown(f"**Nota**: En la integral definida, la constante $C$ se cancela.")
        except Exception:
            F = None
            st.warning("SymPy no pudo calcular la antiderivada simbólicamente. Continuaremos con límites y/o evaluación numérica de respaldo.")

        t = Symbol('t')
        epsilon = Symbol('epsilon')
        lim_val = None

        st.write("**Paso 3 & 4: Evaluación y Cálculo Explícito del Límite**")

        def safe_limit(expr, var_sym, point, dir=None):
            try:
                if dir is None:
                    return limit(expr, var_sym, point)
                else:
                    return limit(expr, var_sym, point, dir=dir)
            except Exception:
                try:
                    if point == oo:
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
                try:
                    a_num = float(a)
                    b_num = float(b)
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
            t1, t2 = Symbol('t1'), Symbol('t2')
            c_val = c if c is not None else 0

            if F is not None:
                F_c1 = F.subs(x, t1) - F.subs(x, a)
                st.markdown("**Desarrollo detallado de la Parte 1**:")
                st.latex(r"\int_{" + latex(a) + "}^{" + latex(c_val) + r"} f(x)\,dx = F\left(" + latex(c_val) + r"^{-}\right) - F\left(" + latex(a) + r"\right)")
                st.latex(r"= \lim_{t_1 \to " + latex(c_val) + "^{-}} \left(" + latex(F.subs(x, t1)) + " - " + latex(F.subs(x, a)) + r"\right)")
                lim_val_1 = safe_limit(F_c1, t1, c_val, dir='-')
                lim_val_1_display = clean_divergence_result(lim_val_1)

                st.markdown(f"**Parte 1: Límite de $\\int_{{{latex(a)}}}^{{{latex(c_val)}}} f(x) dx$ (límite izquierdo)**")
                st.latex(r"\lim_{t_1 \to " + (latex(c_val) if c is not None else str(c_val)) + r"^-} \left[ F(t_1) - F(" + latex(a) + r") \right] = " + latex(lim_val_1_display))

                F_c2 = F.subs(x, b) - F.subs(x, t2)
                st.markdown("**Desarrollo detallado de la Parte 2**:")
                st.latex(r"\int_{" + latex(c_val) + "}^{" + latex(b) + r"} f(x)\,dx = F\left(" + latex(b) + r"\right) - F\left(" + latex(c_val) + r"^{+}\right)")
                st.latex(r"= \lim_{t_2 \to " + latex(c_val) + "^{+}} \left(" + latex(F.subs(x, b)) + " - " + latex(F.subs(x, t2)) + r"\right)")
                lim_val_2 = safe_limit(F_c2, t2, c_val, dir='+')
                lim_val_2_display = clean_divergence_result(lim_val_2)
                st.markdown(f"**Parte 2: Límite de $\\int_{{{latex(c_val)}}}^{{{latex(b)}}} f(x) dx$ (límite derecho)**")
                st.latex(r"\lim_{t_2 \to " + (latex(c_val) if c is not None else str(c_val)) + r"^{+}} \left[ F(" + latex(b) + r") - F(t_2) \right] = " + latex(lim_val_2_display))

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

        if final_res_step_by_step is None:
            final_res_step_by_step = lim_val

        final_res_clean = clean_divergence_result(final_res_step_by_step)

        try:
            res_full = integrate(f, (x, a, b))
        except Exception:
            res_full = sp.nan
            st.warning("SymPy no pudo computar la integral simbólica completa. Se intentó una aproximación numérica si fue posible.")

        st.write("**Paso 5: Análisis de Convergencia (Conclusión Final)**")

        is_finite = False
        try:
            is_finite = res_full.is_finite
        except Exception:
            try:
                if str(res_full).lower() not in ['oo', 'zoo', 'nan', 'infinity'] and sp.N(res_full).is_real:
                    is_finite = True
            except Exception:
                is_finite = False

        if is_finite:
            if mode == "internal_singular" and ((isinstance(lim_val_1_display, sp.Expr) and getattr(lim_val_1_display, "is_infinite", False)) or (isinstance(lim_val_2_display, sp.Expr) and getattr(lim_val_2_display, "is_infinite", False))):
                 st.error("❌ **La integral DIVERGE** (no converge).")
                 try:
                     st.write(f"**Aclaración Importante**: Uno o ambos límites laterales resultaron en $\\pm \\infty$ (Parte 1: ${latex(lim_val_1_display)}$, Parte 2: ${latex(lim_val_2_display)}$). Aunque SymPy pueda devolver un valor principal de Cauchy, la integral es DIVERGENTE porque no existe la suma de las partes.")
                 except Exception:
                     pass
            else:
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
    st.markdown("<h2 style='color:#1E90FF; margin-bottom:0.2rem;'>⚙️ Configuración y Ayuda</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#1E90FF; margin-top:0.5rem;'>📝 Guía de Sintaxis</h3>", unsafe_allow_html=True)
    st.write(
        "- **f(x)**: La función debe usar **x** como variable (ej. `1/x**2`)."
    )
    st.write("- **a / b**: Límite inferior/superior.")
    st.write("- **Potencias**: Usa **`**` (ej. `x**2`).")
    st.write("- **Raíces**: Usa **sqrt(x)** para $\\sqrt{x}$ o potencias fraccionarias (ej. `x**(1/3)` para $\\sqrt[3]{x}$).")
    st.write("- **Infinito**: Usa **oo** para $+\\infty$ o **-oo** para $-\\infty$.")
    st.write("- **Funciones**: Usa **log(x)** para $\\ln(x)$, **exp(x)** para $e^x$, y **E** para la constante de Euler.")

    st.markdown(
        f"""
        <div style='background-color:#eef2ff; color:#1e3a8a; padding:10px; border-radius:8px; font-weight:600;'>
        💡 <strong>Nota del Desarrollador</strong>
        <br>1. El sistema identifica si es **Propia** o **Impropia** (y el tipo).
        <br>2. Primero se calcula la **Antiderivada** **F(x)**.
               <br>3. Luego se aplica el **Límite** correspondiente (a **t** o **épsilon**).
        <br>4. La **convergencia** se declara solo si el límite final es **finito**.
        <br>5. La respuesta final está **validada con SymPy** para máxima confianza.
        </div>
        """,
        unsafe_allow_html=True
    )

    modo = st.selectbox("✨ Opciones de Gráfica",
                        ["Estándar", "Avanzado (con Gráfica Auto)"],
                        index=0, key="modo_select")
    if modo == "Avanzado (con Gráfica Auto)":
        st.checkbox("Activar gráfica automática al resolver", value=True, key="sidebar_auto_graf")

    st.markdown("---")
    st.markdown("### 🔎 Comprobar Java (en esta máquina)")
    st.write("Si corres Streamlit en la misma PC donde está NetBeans, pulsa el botón y te diré la versión de Java.")
    if st.button("Comprobar java -version"):
        try:
            cmd = shlex.split("java -version")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out, err = proc.communicate(timeout=5)
            output = err.strip() if err.strip() != "" else out.strip()
            if output == "":
                st.warning("No se obtuvo salida al ejecutar `java -version`. Asegúrate de que 'java' esté en el PATH del sistema.")
            else:
                st.code(output)
                if "17" in output or "17." in output:
                    st.success("Perfecto — tu Java parece ser JDK 17 (ok para el proyecto).")
                else:
                    st.info("La versión detectada puede no ser Java 17. Si no es 17, instala Temurin / Adoptium JDK 17 para mayor compatibilidad.")
        except FileNotFoundError:
            st.error("No se encontró el ejecutable 'java' en esta máquina. Es probable que no esté instalado o no esté en el PATH.")
        except subprocess.TimeoutExpired:
            st.error("La comprobación tardó demasiado y fue cancelada.")
        except Exception as e:
            st.error(f"Ocurrió un error al comprobar Java: {e}")

tab1, tab2 = st.tabs(["🚀 Resolver Manual", "🧪 Ejemplos Rápidos"])

with tab1:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        f_expr = st.text_input("🔢 f(x):",
                               value="1/x**(5/3)",
                               help="Ej: x**(1/3) | Escribe libremente",
                               key="input_fx")
    with col2:
        a_lim = st.text_input(
            "📏 a (inferior):",
            value="-1",
            help="Ej: 0 (singularidad), 1, o cualquier número",
            key="input_a")
    with col3:
        b_lim = st.text_input("📏 b (superior):",
                              value="1",
                              help="Ej: oo (infinito), 1, o cualquier número",
                              key="input_b")

    progress_bar = st.progress(0)
    if st.button("🔍 Resolver con Detalle Completo", type="primary", key="resolver_detalle_btn"):
        for i in range(100):
            progress_bar.progress(i + 1)
        st.session_state.saved_f = f_expr
        st.session_state.saved_a = a_lim
        st.session_state.saved_b = b_lim
        resolver_integral(f_expr, a_lim, b_lim)
        if modo == "Avanzado (con Gráfica Auto)":
            st.session_state.show_graph = True

    st.session_state.show_graph = st.checkbox(
        "📈 Mostrar Gráfica de f(x) (Área Bajo la Curva Visualizada)",
        value=st.session_state.show_graph,
        key="graph_checkbox"
    )

    if st.session_state.show_graph and st.session_state.saved_f != "":
        try:
            x_sym = Symbol('x')
            f_str_graph = st.session_state.saved_f.replace('E', 'exp(1)').replace('sqrt(', 'sqrt(')
            f = sp.sympify(f_str_graph)
            a = sp.sympify(st.session_state.saved_a)
            b = sp.sympify(st.session_state.saved_b)
            fig, ax = plt.subplots(figsize=(10, 6))

            start = -10.0 if a == -oo else float(a) if hasattr(a, "is_number") and a.is_number else -1.0
            end = 10.0 if b == oo else float(b) if hasattr(b, "is_number") and b.is_number else 1.0

            if start < 0 and end > 0:
                x_vals_neg = np.linspace(start, -0.01, 100)
                x_vals_pos = np.linspace(0.01, end, 100)
                x_vals = np.concatenate((x_vals_neg, [np.nan], x_vals_pos))
            else:
                if start == 0: start = 0.01
                if end == 0: end = -0.01
                if start >= end: end = start + 5.0
                x_vals = np.linspace(start, end, 200)

            f_np = lambdify(x_sym, f, 'numpy')
            y_vals_raw = f_np(x_vals)
            y_vals = np.real(y_vals_raw)
            y_vals[~np.isfinite(y_vals)] = np.nan
            y_vals = np.clip(y_vals, -100, 100)

            ax.plot(x_vals, y_vals, color='#3b82f6', linewidth=2, label=f"f(x) = {st.session_state.saved_f}")
            mask = np.isfinite(y_vals)
            if np.any(mask):
                ax.fill_between(x_vals[mask], 0, y_vals[mask], alpha=0.3, color='#3b82f6', label='Área bajo la curva')

            if a != -oo and hasattr(a, "is_number") and a.is_number:
                ax.axvline(float(a), color='r', linestyle='--', label=f'Límite inferior: {a}', linewidth=2)
            if b != oo and hasattr(b, "is_number") and b.is_number:
                ax.axvline(float(b), color='g', linestyle='--', label=f'Límite superior: {b}', linewidth=2)

            ax.axhline(0, color='black', linewidth=0.5)
            ax.set_title("🔍 Gráfica Interactiva: Visualiza el Área de la Integral", fontsize=16, color='#1e3a8a')
            ax.set_xlabel("x", fontsize=12)
            ax.set_ylabel("f(x)", fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

            y_min = np.nanmin(y_vals)
            y_max = np.nanmax(y_vals)
            if np.isnan(y_min) or np.isnan(y_max) or y_max - y_min < 1:
                ax.set_ylim(-5, 5)
            else:
                ax.set_ylim(max(-100, y_min), min(100, y_max))

            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"❌ Error al generar gráfica: {e}")

with tab2:
    st.markdown("### Ejemplos Clásicos de Integrales Impropias")

    col_ej1, col_ej2, col_ej3 = st.columns(3)
    with col_ej1:
        with st.expander("Ej1: ∫ 1/x² dx de 1 a ∞ (Converge)"):
            st.write("**Función:** 1/x² | **Límites:** a=1, b=∞")
            if st.button("Resolver Ejemplo 1", key="ej1"):
                st.session_state.saved_f = "1/x**2"
                st.session_state.saved_a = "1"
                st.session_state.saved_b = "oo"
                resolver_integral("1/x**2", "1", "oo")
                if modo == "Avanzado (con Gráfica Auto)": st.session_state.show_graph = True

    with col_ej2:
        with st.expander("Ej2: ∫ 1/√x dx de 0 a 1 (Converge)"):
            st.write("**Función:** 1/√x | **Límites:** a=0, b=1")
            if st.button("Resolver Ejemplo 2", key="ej2"):
                st.session_state.saved_f = "1/sqrt(x)"
                st.session_state.saved_a = "0"
                st.session_state.saved_b = "1"
                resolver_integral("1/sqrt(x)", "0", "1")
                if modo == "Avanzado (con Gráfica Auto)": st.session_state.show_graph = True

    with col_ej3:
        with st.expander("Ej3: ∫ 1/x dx de 1 a ∞ (Diverge)"):
            st.write("**Función:** 1/x | **Límites:** a=1, b=∞")
            if st.button("Resolver Ejemplo 3", key="ej3"):
                st.session_state.saved_f = "1/x"
                st.session_state.saved_a = "1"
                st.session_state.saved_b = "oo"
                resolver_integral("1/x", "1", "oo")
                if modo == "Avanzado (con Gráfica Auto)": st.session_state.show_graph = True

    st.markdown("---")

    col_ej4, col_ej5, col_ej6 = st.columns(3)
    with col_ej4:
        with st.expander("Ej4: ∫ ln(x) dx de 0 a 1 (Converge)"):
            st.write("**Función:** ln(x) | **Límites:** a=0, b=1")
            if st.button("Resolver Ejemplo 4", key="ej4"):
                st.session_state.saved_f = "log(x)"
                st.session_state.saved_a = "0"
                st.session_state.saved_b = "1"
                resolver_integral("log(x)", "0", "1")
                if modo == "Avanzado (con Gráfica Auto)": st.session_state.show_graph = True

    with col_ej5:
        with st.expander("Ej5: ∫ 1/x^(5/3) dx de -1 a 1 (Diverge)"):
            st.write("**Función:** 1/x^(5/3) | **Límites:** a=-1, b=1")
            if st.button("Resolver Ejemplo 5", key="ej5"):
                st.session_state.saved_f = "1/x**(5/3)"
                st.session_state.saved_a = "-1"
                st.session_state.saved_b = "1"
                resolver_integral("1/x**(5/3)", "-1", "1")
                if modo == "Avanzado (con Gráfica Auto)": st.session_state.show_graph = True

    with col_ej6:
        with st.expander("Ej6: ∫ x² dx de 0 a 2 (Converge)"):
            st.write("**Función:** x² | **Límites:** a=0, b=2")
            if st.button("Resolver Ejemplo 6", key="ej6"):
                st.session_state.saved_f = "x**2"
                st.session_state.saved_a = "0"
                st.session_state.saved_b = "2"
                resolver_integral("x**2", "0", "2")
                if modo == "Avanzado (con Gráfica Auto)": st.session_state.show_graph = True
