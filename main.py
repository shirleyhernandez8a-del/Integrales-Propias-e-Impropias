# main.py - Versión parcheada por ChatGPT (Colochita)
import streamlit as st
import sympy as sp
from sympy import limit, oo, Symbol, integrate, latex, lambdify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Inicializar session_state
# -------------------------
if "show_graph" not in st.session_state:
    st.session_state.show_graph = False
if "saved_f" not in st.session_state:
    st.session_state.saved_f = ""
if "saved_a" not in st.session_state:
    st.session_state.saved_a = ""
if "saved_b" not in st.session_state:
    st.session_state.saved_b = ""

# -------------------------
# Config de la página
# -------------------------
st.set_page_config(
    page_title="Solver de Integrales Impropias Detallado",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos (mantengo tu CSS)
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
    .katex-display .base { color: #000000 !important; }
    .stApp .stAlert p, .stApp .stAlert h3, .stApp .stAlert * { color: #1e3a8a !important; }
    .sidebar .sidebar-content h1,
    .sidebar .sidebar-content h2,
    .sidebar .sidebar-content h3,
    .sidebar .sidebar-content p,
    .sidebar .sidebar-content label,
    .sidebar .sidebar-content div[data-testid*="stMarkdownContainer"] *,
    .sidebar .sidebar-content div[data-testid*="stHeader"] *,
    .sidebar .sidebar-content div[data-testid*="stText"] * {
        color: #1E90FF !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<h1 style='text-align: center; color: #1e3a8a;'>🧮 Solver de Integrales Impropias - Paso a Paso Detallado</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #4b5563; font-size: 18px;'>Ingresa la función y límites. La app explica <strong>cada subpaso</strong> antes de la respuesta final: tipo, cálculo de antiderivada, evaluación del límite y análisis de convergencia.</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------
# Funciones auxiliares
# -------------------------
def find_singularities(f, a_val, b_val, x):
    try:
        a = float(a_val) if a_val.is_number and a_val != oo and a_val != -oo else None
    except Exception:
        a = None
    try:
        b = float(b_val) if b_val.is_number and b_val != oo and b_val != -oo else None
    except Exception:
        b = None

    if a is None or b is None or a >= b:
        return None

    singularities = []
    try:
        poles = sp.poles(f, x)
        for pole in poles:
            if pole.is_real:
                val = float(pole)
                if a <= val <= b:
                    singularities.append(val)
    except Exception:
        pass

    # heurístico para 0
    try:
        if a is not None and b is not None and a < 0 < b:
            f_at_0 = f.subs(x, 0)
            if not sp.re(f_at_0).is_finite:
                if 0 not in singularities:
                    singularities.append(0)
    except Exception:
        pass

    unique_singularities = sorted(list(set(singularities)))
    if len(unique_singularities) == 1:
        return sp.sympify(unique_singularities[0])
    return None

def check_for_singularities_mode(f, a_val, b_val, x):
    c = find_singularities(f, a_val, b_val, x)
    if a_val == -oo and b_val == oo:
        return "infinite_both", c
    if b_val == oo:
        return "infinite_upper", c
    if a_val == -oo:
        return "infinite_lower", c

    if c is not None:
        if c == a_val:
            return "singular_lower", c
        elif c == b_val:
            return "singular_upper", c
        elif c.is_number and a_val < c < b_val:
            return "internal_singular", c

    return "proper", None

def clean_divergence_result(result):
    if result is oo:
        return oo
    if result is -oo:
        return -oo
    try:
        is_infinite = result.is_infinite
    except Exception:
        is_infinite = False
    if is_infinite and ('-' in str(result) or '(-1)' in str(result) or '(-oo)' in str(result) or 'zoo' in str(result)):
        return -oo
    elif is_infinite:
        return oo
    if result is sp.nan:
        return sp.nan
    return result

# -------------------------
# Lógica principal
# -------------------------
def resolver_integral(f_str, a_str, b_str, var='x'):
    try:
        x = Symbol(var)
        # Preprocesado básico
        f_str_sympify = f_str.replace('E', 'exp(1)').replace('sqrt(', 'sp.sqrt(')
        f = sp.sympify(f_str_sympify)
        a = sp.sympify(a_str)
        b = sp.sympify(b_str)

        st.subheader("📊 Análisis Completo Paso a Paso")

        lim_val_1_display = None
        lim_val_2_display = None
        final_res_step_by_step = None

        mode, c = check_for_singularities_mode(f, a, b, x)
        analysis_notes = []

        st.write("**Paso 1: Identificación del Tipo de Integral**")

        if mode == "internal_singular":
            analysis_notes.append(f"Integral impropia por singularidad interna en $c={latex(c)}$.")
            st.latex(r"\int_{" + latex(a) + "}^{" + latex(b) + r"} f(x) dx = \lim_{t_1 \to " + latex(c) + r"^-} \int_{" + latex(a) + "}^{t_1} f(x) dx + \lim_{t_2 \to " + latex(c) + r"^+} \int_{t_2}^{" + latex(b) + r"} f(x) dx")
        elif mode == "infinite_both":
            analysis_notes.append("Integral con límites $-\infty$ a $\infty$; se divide en dos.")
            st.latex(r"\int_{-\infty}^{\infty} f(x) \, dx = \lim_{t_1 \to -\infty} \int_{t_1}^{0} f(x) \, dx + \lim_{t_2 \to \infty} \int_{0}^{t_2} f(x) \, dx")
        elif mode == "infinite_upper":
            analysis_notes.append("Límite superior infinito.")
            st.latex(r"\int_{" + latex(a) + r"}^\infty f(x) \, dx = \lim_{t \to \infty} \int_{" + latex(a) + r"}^t f(x) \, dx")
        elif mode == "infinite_lower":
            analysis_notes.append("Límite inferior infinito.")
            st.latex(r"\int_{-\infty}^{" + latex(b) + r"} f(x) \, dx = \lim_{t \to -\infty} \int_t^{" + latex(b) + r"} f(x) \, dx")
        elif mode == "singular_lower":
            analysis_notes.append(f"Singularidad en límite inferior $a={latex(a)}$.")
            st.latex(r"\int_{" + latex(a) + "}^{" + latex(b) + r"} f(x) \, dx = \lim_{\epsilon \to " + latex(a) + r"^{+}} \int_{\epsilon}^{" + latex(b) + r"} f(x) \, dx")
        elif mode == "singular_upper":
            analysis_notes.append(f"Singularidad en límite superior $b={latex(b)}$.")
            st.latex(r"\int_{" + latex(a) + "}^{" + latex(b) + r"} f(x) \, dx = \lim_{\epsilon \to " + latex(b) + r"^{-}} \int_{" + latex(a) + r"}^{\epsilon} f(x) \, dx")
        else:
            analysis_notes.append("Integral propia (continua en el intervalo).")

        for note in analysis_notes:
            st.markdown(note)

        st.write("**Función dada**:")
        st.latex(f"f(x) = {latex(f)}")
        st.write(f"**Límites de Integración**: de ${latex(a)}$ a ${latex(b)}$")

        # Antiderivada
        F = sp.integrate(f, x)
        st.write("**Paso 2: Antiderivada**")
        st.latex(r"\int f(x) dx = F(x) = " + latex(F) + r" + C")

        # Evaluación por casos
        t = Symbol('t')
        epsilon = Symbol('epsilon')
        lim_val = None

        st.write("**Paso 3 & 4: Evaluación y Límites**")

        if mode == "proper":
            F_b = F.subs(x, b)
            F_a = F.subs(x, a)
            expr = F_b - F_a
            final_res_step_by_step = sp.simplify(expr)
            st.latex(r"\int_{" + latex(a) + "}^{" + latex(b) + r"} f(x) \, dx = " + latex(final_res_step_by_step))

        elif mode == "infinite_upper":
            expr_t_a = F.subs(x, t) - F.subs(x, a)
            lim_val = limit(expr_t_a, t, oo)
            final_res_step_by_step = lim_val
            st.latex(r"\lim_{t \to \infty} \left[ " + latex(sp.simplify(expr_t_a)) + r" \right] = " + latex(clean_divergence_result(lim_val)))

        elif mode == "infinite_lower":
            expr_b_t = F.subs(x, b) - F.subs(x, t)
            lim_val = limit(expr_b_t, t, -oo)
            final_res_step_by_step = lim_val
            st.latex(r"\lim_{t \to -\infty} \left[ " + latex(sp.simplify(expr_b_t)) + r" \right] = " + latex(clean_divergence_result(lim_val)))

        elif mode == "singular_lower":
            expr_b_eps = F.subs(x, b) - F.subs(x, epsilon)
            lim_val = limit(expr_b_eps, epsilon, a, dir='+')
            final_res_step_by_step = lim_val
            st.latex(r"\lim_{\epsilon \to " + latex(a) + r"^{+}} \left[ " + latex(sp.simplify(expr_b_eps)) + r" \right] = " + latex(clean_divergence_result(lim_val)))

        elif mode == "singular_upper":
            expr_eps_a = F.subs(x, epsilon) - F.subs(x, a)
            lim_val = limit(expr_eps_a, epsilon, b, dir='-')
            final_res_step_by_step = lim_val
            st.latex(r"\lim_{\epsilon \to " + latex(b) + r"^{-}} \left[ " + latex(sp.simplify(expr_eps_a)) + r" \right] = " + latex(clean_divergence_result(lim_val)))

        elif mode == "internal_singular":
            t1, t2 = Symbol('t1'), Symbol('t2')
            c_val = c if c is not None else 0
            F_c1 = F.subs(x, t1) - F.subs(x, a)
            lim_val_1 = limit(F_c1, t1, c_val, dir='-')
            lim_val_1_display = clean_divergence_result(lim_val_1)
            st.markdown(f"**Parte 1 (a -> c^-):** {latex(lim_val_1_display)}")

            F_c2 = F.subs(x, b) - F.subs(x, t2)
            lim_val_2 = limit(F_c2, t2, c_val, dir='+')
            lim_val_2_display = clean_divergence_result(lim_val_2)
            st.markdown(f"**Parte 2 (c^+ -> b):** {latex(lim_val_2_display)}")

            is_div_1 = (lim_val_1_display is oo or lim_val_1_display is -oo or lim_val_1_display is sp.nan)
            is_div_2 = (lim_val_2_display is oo or lim_val_2_display is -oo or lim_val_2_display is sp.nan)

            if is_div_1 or is_div_2:
                if is_div_1:
                    final_res_step_by_step = lim_val_1
                if is_div_2:
                    final_res_step_by_step = lim_val_2
            else:
                final_res_step_by_step = lim_val_1 + lim_val_2

        elif mode == "infinite_both":
            t1, t2 = Symbol('t1'), Symbol('t2')
            # -inf -> 0
            F_inf1 = F.subs(x, 0) - F.subs(x, t1)
            lim_val_1 = limit(F_inf1, t1, -oo)
            lim_val_1_display = clean_divergence_result(lim_val_1)
            st.markdown(f"**Parte 1 (-inf -> 0):** {latex(lim_val_1_display)}")
            # 0 -> +inf
            F_inf2 = F.subs(x, t2) - F.subs(x, 0)
            lim_val_2 = limit(F_inf2, t2, oo)
            lim_val_2_display = clean_divergence_result(lim_val_2)
            st.markdown(f"**Parte 2 (0 -> +inf):** {latex(lim_val_2_display)}")

            is_div_1 = (lim_val_1_display is oo or lim_val_1_display is -oo or lim_val_1_display is sp.nan)
            is_div_2 = (lim_val_2_display is oo or lim_val_2_display is -oo or lim_val_2_display is sp.nan)

            if is_div_1 or is_div_2:
                if is_div_1:
                    final_res_step_by_step = lim_val_1
                if is_div_2:
                    final_res_step_by_step = lim_val_2
            else:
                final_res_step_by_step = lim_val_1 + lim_val_2

        if final_res_step_by_step is None:
            final_res_step_by_step = lim_val

        final_res_clean = clean_divergence_result(final_res_step_by_step)

        # Paso 5: validar con integrate
        res_full = integrate(f, (x, a, b))
        st.write("**Paso 5: Análisis de Convergencia (Conclusión Final)**")

        is_finite = False
        try:
            is_finite = res_full.is_finite
        except Exception:
            try:
                numeric = sp.N(res_full)
                is_finite = numeric.is_real
            except Exception:
                is_finite = False

        if is_finite:
            if mode == "internal_singular" and (lim_val_1_display is oo or lim_val_1_display is -oo or lim_val_2_display is oo or lim_val_2_display is -oo):
                st.error("❌ **La integral DIVERGE** (límite lateral infinito).")
                st.write(f"Parte 1: {latex(lim_val_1_display)}, Parte 2: {latex(lim_val_2_display)}")
            else:
                st.success(f"✅ **La integral CONVERGE** a: ${latex(res_full)}", icon="🎯")
                st.write(f"**Resultado paso a paso (limites/partes):** ${latex(final_res_clean)}")
        else:
            st.error("❌ **La integral DIVERGE** (no converge).")
            st.write(f"Resultado (límite o sumas): ${latex(final_res_clean)}")

    except Exception as e:
        st.error(f"❌ Error en el cálculo: {str(e)}. Tips: usa x, x**2, sqrt(x), oo, log(), exp().")

# -------------------------
# Sidebar (Ayuda)
# -------------------------
with st.sidebar:
    st.markdown("<h2 style='color:#1E90FF; margin-bottom:0.2rem;'>⚙️ Config y Ayuda</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#1E90FF; margin-top:0.5rem;'>📝 Guía de Sintaxis</h3>", unsafe_allow_html=True)
    st.write("- Usa `x` como variable (ej: `1/x**2`).")
    st.write("- Para infinito: `oo` ó `-oo`.")
    st.write("- `sqrt(x)` o `x**(1/3)` para raíces.")
    st.markdown("""
        <div style='background-color:#eef2ff; color:#1e3a8a; padding:10px; border-radius:8px; font-weight:600;'>
        💡 Nota del Dev:
        <br>1. Identificamos tipo (propia/impropia).
        <br>2. Calculamos F(x).
        <br>3. Aplicamos límites (t o epsilon).
        <br>4. Convergencia validada con SymPy.
        </div>
    """, unsafe_allow_html=True)

modo = st.selectbox("✨ Opciones de Gráfica", ["Estándar", "Avanzado (con Gráfica Auto)"], index=0, key="select_modo")
if modo == "Avanzado (con Gráfica Auto)":
    # checkbox con key explícito para evitar duplicados
    auto_graf = st.checkbox("Activar gráfica automática al resolver", value=True, key="auto_graf_checkbox")
else:
    auto_graf = False

tab1, tab2 = st.tabs(["🚀 Resolver Manual", "🧪 Ejemplos Rápidos"])

with tab1:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        f_expr = st.text_input("🔢 f(x):", value="1/x**(5/3)", help="Ej: x**(1/3) | Escribe libremente", key="input_fx")
    with col2:
        a_lim = st.text_input("📏 a (inferior):", value="-1", help="Ej: 0, -1, oo", key="input_a")
    with col3:
        b_lim = st.text_input("📏 b (superior):", value="1", help="Ej: oo, 1", key="input_b")

    progress_bar = st.progress(0, key="progress_main")

    # Botón con key explícito único
    if st.button("🔍 Resolver con Detalle Completo", type="primary", key="resolver_detalle_btn"):
        for i in range(100):
            progress_bar.progress(i + 1)
        st.session_state.saved_f = f_expr
        st.session_state.saved_a = a_lim
        st.session_state.saved_b = b_lim
        resolver_integral(f_expr, a_lim, b_lim)
        if modo == "Avanzado (con Gráfica Auto)":
            st.session_state.show_graph = True

    # Checkbox persistente para la gráfica (con key)
    st.session_state.show_graph = st.checkbox(
        "📈 Mostrar Gráfica de f(x) (Área Bajo la Curva Visualizada)",
        value=st.session_state.show_graph,
        key="graph_checkbox"
    )

    # Bloque de gráfica si marcado
    if st.session_state.show_graph and st.session_state.saved_f != "":
        try:
            x_sym = Symbol('x')
            f_str_graph = st.session_state.saved_f.replace('E', 'exp(1)').replace('sqrt(', 'sp.sqrt(')
            f = sp.sympify(f_str_graph)
            a = sp.sympify(st.session_state.saved_a)
            b = sp.sympify(st.session_state.saved_b)
            fig, ax = plt.subplots(figsize=(10, 6))

            try:
                start = -10.0 if a == -oo else (float(a) if a.is_number and a != 0 else -1.0)
            except Exception:
                start = -1.0
            try:
                end = 10.0 if b == oo else (float(b) if b.is_number and b != 0 else 1.0)
            except Exception:
                end = 1.0

            if start < 0 and end > 0:
                x_vals_neg = np.linspace(start, -0.01, 200)
                x_vals_pos = np.linspace(0.01, end, 200)
                x_vals = np.concatenate((x_vals_neg, [np.nan], x_vals_pos))
            else:
                if start == 0: start = 0.01
                if end == 0: end = -0.01
                if start >= end:
                    end = start + 5.0
                x_vals = np.linspace(start, end, 400)

            try:
                f_np = lambdify(x_sym, f, 'numpy')
                y_vals_raw = f_np(x_vals)
                y_vals = np.real(y_vals_raw)
                y_vals[~np.isfinite(y_vals)] = np.nan
                max_y_limit = 1e5
                y_vals = np.clip(y_vals, -max_y_limit, max_y_limit)
            except Exception as e:
                st.error(f"❌ Error de Dominio en Gráfica: {e}. Mostrando eje solamente.")
                y_vals = np.zeros_like(x_vals)

            # Plot sin pasar objetos SymPy como key
            ax.plot(x_vals, y_vals, label=f"f(x) = {st.session_state.saved_f}", linewidth=2)
            # Fill between (manejo de NaNs)
            finite_mask = np.isfinite(y_vals)
            if finite_mask.any():
                ax.fill_between(x_vals[finite_mask], 0, y_vals[finite_mask], alpha=0.25)

            ax.axhline(0, linewidth=0.8)
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.set_title("Gráfica de f(x)")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.4)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Error generando gráfica: {e}")

with tab2:
    st.markdown("### Ejemplos rápidos")
    ejemplos = [
        ("1/x", "1", "oo"),
        ("1/x**2", "1", "oo"),
        ("1/sqrt(x)", "0", "1"),
        ("1/x**(5/3)", "-1", "1"),
    ]
    for i, (f_ex, a_ex, b_ex) in enumerate(ejemplos):
        # botón de ejemplo con key único por índice
        if st.button(f"Ejecutar ejemplo: ∫ {f_ex} de {a_ex} a {b_ex}", key=f"ejemplo_btn_{i}"):
            st.session_state.saved_f = f_ex
            st.session_state.saved_a = a_ex
            st.session_state.saved_b = b_ex
            resolver_integral(f_ex, a_ex, b_ex)
            # si estaba activada auto-graf, mostrar gráfica
            if modo == "Avanzado (con Gráfica Auto)":
                st.session_state.show_graph = True

# fin del script
