import streamlit as st
import sympy as sp
from sympy import limit, oo, Symbol, integrate, latex, lambdify, exp, E, pi, sqrt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import subprocess  # <-- agregado para comprobar java
import shlex       # <-- ayuda a parsear comandos seguro

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
    Intenta encontrar un punto de singularidad real en el intervalo [a, b] o en sus límites.
    Retorna la singularidad (si es simple y única) o None.
    """
    # Robustecemos acceso a is_number para evitar AttributeError
    try:
        a = float(a_val) if hasattr(a_val, "is_number") and a_val.is_number and a_val != oo and a_val != -oo else None
    except Exception:
        a = None
    try:
        b = float(b_val) if hasattr(b_val, "is_number") and b_val.is_number and b_val != oo and b_val != -oo else None
    except Exception:
        b = None
    
    if a is None or b is None or a >= b:
        return None # No se pueden chequear singularidades internas si hay límites infinitos o rango inválido.

    singularities = []
    
    # Intento 1: Usar sp.poles (más preciso)
    try:
        poles = sp.poles(f, x)
        for pole in poles:
            if pole.is_real:
                val = float(pole)
                if a <= val <= b:
                    singularities.append(val)
    except Exception:
        # Fallback si sp.poles falla
        pass
    
    # Intento 2: Chequeo heurístico para x=0 (común)
    if a is not None and b is not None and a < 0 < b:
        try:
            # Subir a 0^n para probar límites si la simple sustitución falla
            f_at_0 = f.subs(x, 0)
            if not sp.re(f_at_0).is_finite:
                if 0 not in singularities:
                    singularities.append(0)
        except Exception:
            pass

    # Eliminamos duplicados
    unique_singularities = sorted(list(set(singularities)))
    
    # Si encontramos una sola singularidad, la retornamos
    if len(unique_singularities) == 1:
        return sp.sympify(unique_singularities[0])
    
    # Si encontramos múltiples o ninguna singularidad, retornamos None
    return None

def check_for_singularities_mode(f, a_val, b_val, x):
    """
    Verifica si la integral es impropia debido a límites infinitos o singularidades.
    Retorna el modo ('proper', 'infinite_upper', etc.) y la singularidad 'c'.
    """
    c = find_singularities(f, a_val, b_val, x)
    
    # 1. Chequeo de Límites Infinitos
    if a_val == -oo and b_val == oo:
        return "infinite_both", c
    if b_val == oo:
        return "infinite_upper", c
    if a_val == -oo:
        return "infinite_lower", c

    # 2. Chequeo de Singularidades
    if c is not None:
        if c == a_val:
            return "singular_lower", c
        elif c == b_val:
            return "singular_upper", c
        elif hasattr(c, "is_number") and c.is_number and a_val < c < b_val:
            return "internal_singular", c
            
    # Si no es impropia por límites o singularidades detectadas
    return "proper", None
    
def clean_divergence_result(result):
    """
    Limpia el resultado de SymPy si es infinito y contiene términos complejos o confusos.
    Asegura que solo se muestre 'oo' o '-oo' cuando el resultado diverge.
    """
    # Si SymPy ya lo identificó como infinito, lo limpiamos para evitar símbolos como oo*(-1)**(1/3)
    if result is oo:
        return oo
    if result is -oo:
        return -oo
        
    # Verificar si el resultado es infinito, incluyendo los casos zoo (complejo infinito) o Infinity
    is_infinite = getattr(result, "is_infinite", False)
    
    # Patrones que indican divergencia negativa o compleja que debe ser simplificada a -oo
    if is_infinite and ('-' in str(result) or '(-1)' in str(result) or '(-oo)' in str(result) or 'zoo' in str(result)):
        return -oo
    elif is_infinite:
        return oo
        
    # Si es NaN, es una divergencia no definida
    if result is sp.nan:
        return sp.nan
        
    return result

def resolver_integral(f_str, a_str, b_str, var='x'):
    try:
        x = Symbol(var)
        # Reemplazar 'E' con 'exp(1)' y 'sqrt' con 'sp.sqrt' para SymPy de forma más robusta
        # Esto soluciona el error 'Symbol' object has no attribute 'sqrt'
        f_str_sympify = f_str.replace('E', 'exp(1)').replace('sqrt(', 'sp.sqrt(')
        
        # Validación temprana y mensaje amigable si falla sympify
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

        st.subheader("📊 Análisis Completo Paso a Paso")
        
        # Inicializamos los resultados parciales para los modos complejos
        lim_val_1_display = None
        lim_val_2_display = None
        final_res_step_by_step = None
        
        # --- LÓGICA DE DETECCIÓN DE TIPO MÁS ROBUSTA ---
        mode, c = check_for_singularities_mode(f, a, b, x)
        analysis_notes = []
        
        st.write("**Paso 1: Identificación del Tipo de Integral**")
        
        # preparar latex seguro para c
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

        # --- CÁLCULO DE LA ANTIDERIVADA ---
        F = sp.integrate(f, x)
        st.write("**Paso 2: Encontrar la Antiderivada Indefinida $F(x)$**")
        st.latex(r"\int f(x) dx = F(x) = " + latex(F) + r" + C")
        st.markdown(f"**Nota**: En la integral definida, la constante $C$ se cancela.")

        # --- APLICACIÓN DE LÍMITES DETALLADA (Paso 3 y 4) ---
        t = Symbol('t')
        epsilon = Symbol('epsilon')
        lim_val = None # Inicializamos lim_val para los modos simples

        st.write("**Paso 3 & 4: Evaluación y Cálculo Explícito del Límite**")

        if mode == "proper":
            F_b = F.subs(x, b)
            F_a = F.subs(x, a)
            expr = F_b - F_a
            st.markdown(r"Aplicamos el Teorema Fundamental del Cálculo:")
            st.latex(r"\int_{" + latex(a) + "}^{" + latex(b) + r"} f(x) \, dx = F(" + latex(b) + ") - F(" + latex(a) + ")")
            st.latex(r"= \left[ " + latex(F_b) + r" \right] - \left[ " + latex(F_a) + r" \right]")
            final_res_step_by_step = sp.simplify(expr)
            st.latex(r"= " + latex(final_res_step_by_step))
        
        elif mode == "infinite_upper":
            expr_t_a = F.subs(x, t) - F.subs(x, a)
            lim_val = limit(expr_t_a, t, oo)
            final_res_step_by_step = lim_val
            st.markdown(r"Sustituimos el límite superior infinito con $t$:") 
            st.latex(r"\lim_{t \to \infty} \left[ " + latex(sp.simplify(expr_t_a)) + r" \right] = " + latex(clean_divergence_result(lim_val)))

        elif mode == "infinite_lower":
            expr_b_t = F.subs(x, b) - F.subs(x, t)
            lim_val = limit(expr_b_t, t, -oo)
            final_res_step_by_step = lim_val
            st.markdown(r"Sustituimos el límite inferior infinito con $t$:") 
            st.latex(r"\lim_{t \to -\infty} \left[ " + latex(sp.simplify(expr_b_t)) + r" \right] = " + latex(clean_divergence_result(lim_val)))

        elif mode == "singular_lower":
            expr_b_eps = F.subs(x, b) - F.subs(x, epsilon)
            lim_val = limit(expr_b_eps, epsilon, a, dir='+')
            final_res_step_by_step = lim_val
            st.markdown(r"Sustituimos el límite inferior singular con $\epsilon$ y tomamos el límite lateral $\epsilon \to a^{+}$:") 
            st.latex(r"\lim_{\epsilon \to " + latex(a) + r"^{+}} \left[ " + latex(sp.simplify(expr_b_eps)) + r" \right] = " + latex(clean_divergence_result(lim_val)))

        elif mode == "singular_upper":
            expr_eps_a = F.subs(x, epsilon) - F.subs(x, a)
            lim_val = limit(expr_eps_a, epsilon, b, dir='-')
            final_res_step_by_step = lim_val
            st.markdown(r"Sustituimos el límite superior singular con $\epsilon$ y tomamos el límite lateral $\epsilon \to b^{-}$:") 
            st.latex(r"\lim_{\epsilon \to " + latex(b) + r"^{-}} \left[ F(" + latex(b) + r") - F(\epsilon) \right] = " + latex(clean_divergence_result(lim_val)))

        elif mode == "internal_singular":
            t1, t2 = Symbol('t1'), Symbol('t2')
            c_val = c if c is not None else 0 # Usar 0 si no se encuentra c para dividir

            # Parte 1: a hasta c (límite inferior t1 -> c-)
            F_c1 = F.subs(x, t1) - F.subs(x, a)

            # Mostrar el desarrollo paso a paso de la primera parte
            st.markdown("**Desarrollo detallado de la Parte 1**:")
            st.latex(r"\int_{" + latex(a) + "}^{" + latex(c_val) + r"} f(x)\,dx = F\left(" + latex(c_val) + r"^{-}\right) - F\left(" + latex(a) + r"\right)")
            st.latex(r"= \lim_{t_1 \to " + latex(c_val) + "^{-}} \left[" + latex(F) + r"\right]_{x=" + latex(a) + "}^{x=t_1}")
            st.latex(r"= \lim_{t_1 \to " + latex(c_val) + "^{-}} \left(" + latex(F.subs(x, t1)) + " - " + latex(F.subs(x, a)) + r"\right)")

            lim_val_1 = limit(F_c1, t1, c_val, dir='-')
            lim_val_1_display = clean_divergence_result(lim_val_1) # Se limpia el resultado aquí

            st.markdown(f"**Parte 1: Límite de $\\int_{{{latex(a)}}}^{{{latex(c_val)}}} f(x) dx$ (límite izquierdo)**")
            st.latex(r"\lim_{t_1 \to " + (latex(c_val) if c is not None else str(c_val)) + r"^-} \left[ F(t_1) - F(" + latex(a) + r") \right] = " + latex(lim_val_1_display))

            # Chequeo explícito de divergencia en la primera parte
            is_div_1 = (getattr(lim_val_1_display, "is_infinite", False) or lim_val_1_display is sp.nan)

            # Parte 2: c hasta b (límite superior t2 -> c+)
            F_c2 = F.subs(x, b) - F.subs(x, t2)

            # Mostrar el desarrollo paso a paso de la segunda parte
            st.markdown("**Desarrollo detallado de la Parte 2**:")
            st.latex(r"\int_{" + latex(c_val) + "}^{" + latex(b) + r"} f(x)\,dx = F\left(" + latex(b) + r"\right) - F\left(" + latex(c_val) + r"^{+}\right)")
            st.latex(r"= \lim_{t_2 \to " + latex(c_val) + "^{+}} \left[" + latex(F) + r"\right]_{x=t_2}^{x=" + latex(b) + "}")
            st.latex(r"= \lim_{t_2 \to " + latex(c_val) + "^{+}} \left(" + latex(F.subs(x, b)) + " - " + latex(F.subs(x, t2)) + r"\right)")

            lim_val_2 = limit(F_c2, t2, c_val, dir='+')
            lim_val_2_display = clean_divergence_result(lim_val_2) # Se limpia el resultado aquí

            st.markdown(f"**Parte 2: Límite de $\\int_{{{latex(c_val)}}}^{{{latex(b)}}} f(x) dx$ (límite derecho)**")
            st.latex(r"\lim_{t_2 \to " + (latex(c_val) if c is not None else str(c_val)) + r"^{+}} \left[ F(" + latex(b) + r") - F(t_2) \right] = " + latex(lim_val_2_display))

            # Chequeo explícito de divergencia en la segunda parte
            is_div_2 = (getattr(lim_val_2_display, "is_infinite", False) or lim_val_2_display is sp.nan)

            # Si una o ambas divergen, el resultado final diverge
            if is_div_1 or is_div_2:
                if is_div_1:
                    final_res_step_by_step = lim_val_1
                if is_div_2:
                    final_res_step_by_step = lim_val_2

            # Si ambos convergen, se suman
            if not is_div_1 and not is_div_2:
                final_res_step_by_step = lim_val_1 + lim_val_2


        elif mode == "infinite_both":
            t1, t2 = Symbol('t1'), Symbol('t2')
            c_val = 0 # Usamos 0 como punto de división

            # Parte 1: -oo hasta 0 (límite inferior t1 -> -oo)
            F_inf1 = F.subs(x, 0) - F.subs(x, t1)
            lim_val_1 = limit(F_inf1, t1, -oo)
            lim_val_1_display = clean_divergence_result(lim_val_1) # Se limpia el resultado aquí
            
            st.markdown(f"**Parte 1: Límite de $\\int_{{-\\infty}}^{{0}} f(x) dx$**")
            st.latex(r"\lim_{t_1 \to -\infty} \left[ F(0) - F(t_1) \right] = " + latex(lim_val_1_display))
            
            is_div_1 = (getattr(lim_val_1_display, "is_infinite", False) or lim_val_1_display is sp.nan)
            
            # Parte 2: 0 hasta oo (límite superior t2 -> oo)
            F_inf2 = F.subs(x, t2) - F.subs(x, 0)
            lim_val_2 = limit(F_inf2, t2, oo)
            lim_val_2_display = clean_divergence_result(lim_val_2) # Se limpia el resultado aquí

            st.markdown(f"**Parte 2: Límite de $\\int_{{0}}^{{\\infty}} f(x) dx$**")
            st.latex(r"\lim_{t_2 \to \infty} \left[ F(t_2) - F(0) \right] = " + latex(lim_val_2_display))
            
            is_div_2 = (getattr(lim_val_2_display, "is_infinite", False) or lim_val_2_display is sp.nan)

            if is_div_1 or is_div_2:
                # Si una o ambas divergen, el resultado final diverge
                if is_div_1: final_res_step_by_step = lim_val_1
                if is_div_2: final_res_step_by_step = lim_val_2

            if not is_div_1 and not is_div_2:
                final_res_step_by_step = lim_val_1 + lim_val_2
        
        # Si no es un caso complejo que ya fue procesado, usamos el resultado de los límites simples
        if final_res_step_by_step is None:
            final_res_step_by_step = lim_val
        
        # Hacemos una última limpieza al resultado final antes de la conclusión
        final_res_clean = clean_divergence_result(final_res_step_by_step)


        # --- ANÁLISIS DE CONVERGENCIA (USA EL RESULTADO DIRECTO DE SYMPY PARA MAXIMIZAR ROBUSTEZ) ---
        
        # Este es el cálculo final y robusto que asegura la respuesta correcta.
        try:
            res_full = integrate(f, (x, a, b))
        except Exception:
            # Si SymPy no devuelve simbólicamente la integral, marcamos como indefinido simbólicamente.
            res_full = sp.nan
            st.warning("SymPy no pudo computar la integral simbólica completa. El resultado simbólico quedó indefinido (se intentó una aproximación en otros pasos).")

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
            # Caso especial: integral impar de singularidad, divergente en realidad (ej. 1/x**(5/3) de -1 a 1).
            # Comprobamos que si es una singularidad interna y el límite da infinito, diverge
            if mode == "internal_singular" and ((isinstance(lim_val_1_display, sp.Expr) and getattr(lim_val_1_display, "is_infinite", False)) or (isinstance(lim_val_2_display, sp.Expr) and getattr(lim_val_2_display, "is_infinite", False))):
                 st.error("❌ **La integral DIVERGE** (no converge).")
                 try:
                     st.write(f"**Aclaración Importante**: Uno o ambos límites laterales resultaron en $\\pm \\infty$ (Parte 1: ${latex(lim_val_1_display)}$, Parte 2: ${latex(lim_val_2_display)}$). Aunque SymPy pueda devolver un valor principal de Cauchy ($0$ en este caso), la integral es propiamente **DIVERGENTE** porque la función no es continua en el intervalo.")
                 except Exception:
                     pass
            else:
                st.success(
                    f"✅ **La integral CONVERGE** a un valor finito: ${latex(res_full)}$."
                )
                st.write(
                    f"**Explicación detallada**: El límite (o la suma de los límites en casos divididos) es finito (${latex(final_res_step_by_step)}$), por lo que el área bajo la curva es acotada."
                )
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
                
                # Aclaración de signos neutra
                if (isinstance(lim_val_1_display, sp.Expr) and getattr(lim_val_1_display, "is_infinite", False)) and (isinstance(lim_val_2_display, sp.Expr) and getattr(lim_val_2_display, "is_infinite", False)) and (str(lim_val_1_display).startswith('-') and not str(lim_val_2_display).startswith('-')):
                     st.info("⚠️ **Aclaración de Signos**: La integral diverge porque el límite izquierdo es $\\mathbf{-\\infty}$ y el límite derecho es $\\mathbf{+\\infty}$. Como al menos una de las partes es infinita, la integral total **DIVERGE**.")
                else:
                    st.write(f"El resultado del límite divergente fue: ${latex(final_res_clean)}$")

            elif mode == "infinite_both":
                try:
                    st.markdown(f"**Resultado de la Parte 1 ($-\infty$ a $0$)**: ${latex(lim_val_1_display)}$")
                    st.markdown(f"**Resultado de la Parte 2 ($0$ a $\\infty$)**: ${latex(lim_val_2_display)}$")
                except Exception:
                    pass
                st.write(f"El resultado del límite divergente fue: ${latex(final_res_clean)}$")

            else:
                # Muestra el resultado limpio de divergencia
                st.write(f"El resultado del límite es: ${latex(final_res_clean)}$")
                 
            st.write(
                "**Explicación detallada**: El límite (o la suma de los límites en casos divididos) resultó en $\\pm \infty$ o no existe, lo que significa que el área crece sin cota (ilimitada)."
            )

    except Exception as e:
        # Mensaje de error actualizado
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
    
    # ----------------------------------------------------------------------------------
    # Tip Pro ajustado, eliminando LaTeX para F(x), t, epsilon, y límites.
    # ----------------------------------------------------------------------------------
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
    # ----------------------------------------------------------------------------------

    modo = st.selectbox("✨ Opciones de Gráfica",
                        ["Estándar", "Avanzado (con Gráfica Auto)"],
                        index=0, key="modo_select")
    if modo == "Avanzado (con Gráfica Auto)":
        st.checkbox("Activar gráfica automática al resolver", value=True, key="sidebar_auto_graf")

    # ---------- NUEVO: Comprobador de Java (NO cambia nada más) ----------
    st.markdown("---")
    st.markdown("### 🔎 Comprobar Java (en esta máquina)")
    st.write("Si corres Streamlit en la misma PC donde está NetBeans, pulsa el botón y te diré la versión de Java.")
    if st.button("Comprobar java -version"):
        try:
            # usamos shlex.split por seguridad en distintos OS
            cmd = shlex.split("java -version")
            # java -version imprime en stderr en muchos sistemas -> capturamos ambos
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out, err = proc.communicate(timeout=5)
            # Preferimos mostrar err si existe (es donde java -version suele escribir)
            output = err.strip() if err.strip() != "" else out.strip()
            if output == "":
                st.warning("No se obtuvo salida al ejecutar `java -version`. Asegúrate de que 'java' esté en el PATH del sistema.")
            else:
                st.code(output)
                # Mensaje amigable: interpretar la versión
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
    # --------------------------------------------------------------------

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

    # NOTE: st.progress does not accept a 'key' argument — así que lo dejamos sin key.
    progress_bar = st.progress(0)
    if st.button("🔍 Resolver con Detalle Completo", type="primary", key="resolver_detalle_btn"):
        for i in range(100):
            progress_bar.progress(i + 1)
        # Guarda datos en session_state para gráfica persistente
        st.session_state.saved_f = f_expr
        st.session_state.saved_a = a_lim
        st.session_state.saved_b = b_lim
        resolver_integral(f_expr, a_lim, b_lim)
        # Auto gráfica si modo avanzado
        if modo == "Avanzado (con Gráfica Auto)":
            st.session_state.show_graph = True

    # Checkbox persistente para gráfica
    st.session_state.show_graph = st.checkbox(
        "📈 Mostrar Gráfica de f(x) (Área Bajo la Curva Visualizada)",
        value=st.session_state.show_graph,
        key="graph_checkbox"
    )
    # Bloque de gráfica si checkbox marcado
    if st.session_state.show_graph and st.session_state.saved_f != "":
        try:
            x_sym = Symbol('x')
            # Usamos el string original guardado para la gráfica, pero lo pre-procesamos si usa 'E' o 'sqrt'
            f_str_graph = st.session_state.saved_f.replace('E', 'exp(1)').replace('sqrt(', 'sp.sqrt(')
            f = sp.sympify(f_str_graph)
            a = sp.sympify(st.session_state.saved_a)
            b = sp.sympify(st.session_state.saved_b)
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Manejo seguro de start/end para la gráfica
            try:
                # Si el límite inferior es infinito negativo, ajustamos el inicio
                start = -10.0 if a == -oo else (float(a) if hasattr(a, "is_number") and a.is_number and a != 0 else -1.0)
            except Exception:
                start = -1.0
            try:
                # Si el límite superior es infinito positivo, ajustamos el final
                end = 10.0 if b == oo else (float(b) if hasattr(b, "is_number") and b.is_number and b != 0 else 1.0)
            except Exception:
                end = 1.0
                
            # Ajuste extra para singularidades: Evitar empezar en 0 o terminar en 0
            # Si el rango incluye 0, separamos las curvas
            if start < 0 and end > 0:
                x_vals_neg = np.linspace(start, -0.01, 100)
                x_vals_pos = np.linspace(0.01, end, 100)
                x_vals = np.concatenate((x_vals_neg, [np.nan], x_vals_pos)) # Separador en 0
            else:
                if start == 0: start = 0.01
                if end == 0: end = -0.01
                # Aseguramos que 'start' sea menor que 'end'
                if start >= end:
                    end = start + 5.0
                x_vals = np.linspace(start, end, 200)
            
            # Evaluar f(x) numéricamente con lambdify y manejo robusto de errores
            try:
                f_np = lambdify(x_sym, f, 'numpy')
                y_vals_raw = f_np(x_vals)

                # --- FIX: Manejo Robusto para Gráficas (Evita errores de Dominio/Números Complejos) ---
                y_vals = np.real(y_vals_raw)
                y_vals[~np.isfinite(y_vals)] = np.nan
                max_y_limit = 100.0
                y_vals = np.clip(y_vals, -max_y_limit, max_y_limit)
                # -----------------------------------------------------------------------------------

            except Exception as e:
                st.error(f"❌ Error de Dominio en Gráfica: {e}. Esto sucede cuando la función (ej. $\\sqrt{{x}}$ o $1/x$) se evalúa fuera de su dominio. Mostrando solo el eje.")
                y_vals = np.zeros_like(x_vals) # Fallback

            ax.plot(x_vals,
                    y_vals,
                    label=f"f(x) = {st.session_state.saved_f}",
                    color='#3b82f6',
                    linewidth=2)
            # Sombreado para área bajo la curva
            x_fill = x_vals[np.isfinite(y_vals)]
            y_fill = y_vals[np.isfinite(y_vals)]
            if x_fill.size > 0:
                ax.fill_between(x_fill,
                                0,
                                y_fill,
                                alpha=0.3,
                                color='#3b82f6',
                                label='Área aproximada')
            
            # Límites (ajustados para infinito)
            if a != -oo and hasattr(a, "is_number") and a.is_number:
                ax.axvline(float(a),
                           color='r',
                           linestyle='--',
                           label=f'Límite inferior: {a}',
                           linewidth=2)
            if b != oo and hasattr(b, "is_number") and b.is_number:
                ax.axvline(float(b),
                           color='g',
                           linestyle='--',
                           label=f'Límite superior: {b}',
                           linewidth=2)

            # Eje Y en 0
            ax.axhline(0, color='black', linewidth=0.5)

            ax.set_title(
                "🔍 Gráfica Interactiva: Visualiza el Área de la Integral",
                fontsize=16,
                color='#1e3a8a')
            ax.set_xlabel("x", fontsize=12)
            ax.set_ylabel("f(x)", fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            try:
                y_min = np.nanmin(y_vals)
                y_max = np.nanmax(y_vals)
                if np.isnan(y_min) or np.isnan(y_max) or y_max - y_min < 1:
                    ax.set_ylim(-5, 5)
                else:
                    # Limitar el eje Y para evitar gráficos ilegibles debido a picos infinitos
                    ax.set_ylim(max(-100, y_min), min(100, y_max))
            except Exception:
                ax.set_ylim(-5, 5)

            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"❌ Error al generar gráfica: {e}. Verifica función simple.")

with tab2:
    st.markdown("### Ejemplos Clásicos de Integrales Impropias")
    
    col_ej1, col_ej2, col_ej3 = st.columns(3)
    with col_ej1:
        with st.expander("Ej1: $\\int 1/x^2 dx$ de 1 a $\\infty$ (Converge)"):
            st.write("**Función**: $1/x^2$ | **Límites**: $a=1, b=\\infty$")
            if st.button("Resolver Ejemplo 1", key="ej1"):
                st.session_state.saved_f = "1/x**2"
                st.session_state.saved_a = "1"
                st.session_state.saved_b = "oo"
                resolver_integral("1/x**2", "1", "oo")
                if modo == "Avanzado (con Gráfica Auto)": st.session_state.show_graph = True
    with col_ej2:
        with st.expander("Ej2: $\\int 1/\\sqrt{x} dx$ de 0 a 1 (Singular, Converge)"):
            st.write("**Función**: $1/\\sqrt{x}$ | **Límites**: $a=0, b=1$")
            if st.button("Resolver Ejemplo 2", key="ej2"):
                st.session_state.saved_f = "1/sqrt(x)" 
                st.session_state.saved_a = "0"
                st.session_state.saved_b = "1"
                resolver_integral("1/sqrt(x)", "0", "1")
                if modo == "Avanzado (con Gráfica Auto)": st.session_state.show_graph = True
    with col_ej3:
        with st.expander("Ej3: $\\int 1/x dx$ de 1 a $\\infty$ (Diverge)"):
            st.write("**Función**: $1/x$ | **Límites**: $a=1, b=\\infty$")
            if st.button("Resolver Ejemplo 3", key="ej3"):
                st.session_state.saved_f = "1/x"
                st.session_state.saved_a = "1"
                st.session_state.saved_b = "oo"
                resolver_integral("1/x", "1", "oo")
                if modo == "Avanzado (con Gráfica Auto)": st.session_state.show_graph = True

    st.markdown("---") # Separador visual para la segunda fila

    col_ej4, col_ej5, col_ej6 = st.columns(3)
    with col_ej4:
        with st.expander("Ej4: $\\int \ln(x) dx$ de 0 a 1 (Singular, Converge)"):
            st.write("**Función**: $\\ln(x)$ | **Límites**: $a=0, b=1$")
            if st.button("Resolver Ejemplo 4", key="ej4"):
                st.session_state.saved_f = "log(x)" 
                st.session_state.saved_a = "0"
                st.session_state.saved_b = "1"
                resolver_integral("log(x)", "0", "1")
                if modo == "Avanzado (con Gráfica Auto)": st.session_state.show_graph = True
    with col_ej5:
        with st.expander("Ej5: $\\int 1/x^{5/3} dx$ de -1 a 1 (Singularidad Interna, DIVERGE)"):
            st.write("**Función**: $1/x^{5/3}$ | **Límites**: $a=-1, b=1$")
            if st.button("Resolver Ejemplo 5", key="ej5"):
                st.session_state.saved_f = "1/x**(5/3)" 
                st.session_state.saved_a = "-1"
                st.session_state.saved_b = "1"
                resolver_integral("1/x**(5/3)", "-1", "1")
                if modo == "Avanzado (con Gráfica Auto)": st.session_state.show_graph = True
    with col_ej6:
        with st.expander("Ej6: $\\int x^2 dx$ de 0 a 2 (Integral Propia, Converge)"):
            st.write("**Función**: $x^2$ | **Límites**: $a=0, b=2$")
            if st.button("Resolver Ejemplo 6", key="ej6"):
                st.session_state.saved_f = "x**2" 
                st.session_state.saved_a = "0"
                st.session_state.saved_b = "2"
                resolver_integral("x**2", "0", "2")
                if modo == "Avanzado (con Gráfica Auto)": st.session_state.show_graph = True
