import sympy as sp

from sympy import limit, oo, Symbol, integrate, latex, lambdify, exp, E, pi, sqrt

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np



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

    

    a = float(a_val) if a_val.is_number and a_val != oo and a_val != -oo else None

    b = float(b_val) if b_val.is_number and b_val != oo and b_val != -oo else None

    

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

        elif c.is_number and a_val < c < b_val:

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

    is_infinite = result.is_infinite

    

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

        

        f = sp.sympify(f_str_sympify)

        a = sp.sympify(a_str)

        b = sp.sympify(b_str)



        st.subheader("📊 Análisis Completo Paso a Paso")

        

        # Inicializamos los resultados parciales para los modos complejos

        lim_val_1_display = None

        lim_val_2_display = None

        final_res_step_by_step = None

        

        # --- LÓGICA DE DETECCIÓN DE TIPO MÁS ROBUSTA ---

        mode, c = check_for_singularities_mode(f, a, b, x)

        analysis_notes = []

        

        st.write("**Paso 1: Identificación del Tipo de Integral**")

        

        if mode == "internal_singular":

            analysis_notes.append(f"Esta es una integral impropia por **singularidad interna** (discontinuidad en $c={latex(c)}$), donde ${latex(a)} < {latex(c)} < {latex(b)}$.")

            analysis_notes.append("Se debe dividir en dos integrales impropias:")

            st.latex(r"\int_{" + latex(a) + "}^{" + latex(b) + r"} f(x) dx = \lim_{t_1 \to " + latex(c) + r"^-} \int_{" + latex(a) + "}^{t_1} f(x) dx + \lim_{t_2 \to " + latex(c) + r"^+} \int_{t_2}^{" + latex(b) + r"} f(x) dx")

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

            st.latex(r"\lim_{t \to \infty} \left[ F(t) - F(" + latex(a) + r") \right] = \lim_{t \to \infty} \left[ \left(" + latex(F.subs(x, t)) + r"\right) - \left(" + latex(F.subs(x, a)) + r"\right) \right]")

            st.latex(r"\lim_{t \to \infty} \left[ " + latex(sp.simplify(expr_t_a)) + r" \right] = " + latex(clean_divergence_result(lim_val)))



        elif mode == "infinite_lower":

            expr_b_t = F.subs(x, b) - F.subs(x, t)

            lim_val = limit(expr_b_t, t, -oo)

            final_res_step_by_step = lim_val

            st.markdown(r"Sustituimos el límite inferior infinito con $t$:")

            st.latex(r"\lim_{t \to -\infty} \left[ F(" + latex(b) + r") - F(t) \right] = \lim_{t \to -\infty} \left[ \left(" + latex(F.subs(x, b)) + r"\right) - \left(" + latex(F.subs(x, t)) + r"\right) \right]")

            st.latex(r"\lim_{t \to -\infty} \left[ " + latex(sp.simplify(expr_b_t)) + r" \right] = " + latex(clean_divergence_result(lim_val)))



        elif mode == "singular_lower":

            expr_b_eps = F.subs(x, b) - F.subs(x, epsilon)

            lim_val = limit(expr_b_eps, epsilon, a, dir='+')

            final_res_step_by_step = lim_val

            st.markdown(r"Sustituimos el límite inferior singular con $\epsilon$ y tomamos el límite lateral $\epsilon \to a^{+}$:")

            st.latex(r"\lim_{\epsilon \to " + latex(a) + r"^{+}} \left[ F(" + latex(b) + r") - F(\epsilon) \right] = \lim_{\epsilon \to " + latex(a) + r"^{+}} \left[ \left(" + latex(F.subs(x, b)) + r"\right) - \left(" + latex(F.subs(x, epsilon)) + r"\right) \right]")

            st.latex(r"\lim_{\epsilon \to " + latex(a) + r"^{+}} \left[ " + latex(sp.simplify(expr_b_eps)) + r" \right] = " + latex(clean_divergence_result(lim_val)))



        elif mode ==
