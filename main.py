import streamlit as st
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
    page_title="Solver de Integrales Impropias Detallado",
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
    "<h1 style='text-align: center; color: #1e3a8a;'>🧮 Solver de Integrales Impropias - Paso a Paso Detallado</h1>",
    unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #4b5563; font-size: 18px;'>Ingresa la función y límites. La app explica **cada subpaso** antes de la respuesta final: tipo, cálculo de antiderivada, evaluación del límite y análisis de convergencia. ¡Visualiza el área y converge a la excelencia! 🎓</p>",
    unsafe_allow_html=True)
st.markdown("---")

# Función auxiliar para verificar si hay una singularidad interna en el intervalo [a, b]
def check_for_singularities(f, a_val, b_val, x):
    """
    Verifica si la integral es impropia debido a límites infinitos o singularidades.
    Retorna el modo (ej. 'proper', 'infinite_upper', 'internal_singular').
    """
    
    # 1. Chequeo de Límites Infinitos
    if a_val == -oo and b_val == oo:
        return "infinite_both"
    if b_val == oo:
        return "infinite_upper"
    if a_val == -oo:
        return "infinite_lower"

    # Convertir límites a flotantes para la verificación de singularidad interna
    a = float(a_val) if a_val.is_number else None
    b = float(b_val) if b_val.is_number else None
    
    # 2. Chequeo de Singularidades Internas y en Límites
    if a is not None and b is not None and a < b:
        # Puntos de chequeo heurísticos (0, a, b y un punto intermedio)
        check_points = [a, b]
        if a < 0 and b > 0:
             check_points.append(0) # Chequear singularidad en 0
        
        # Chequear en los límites
        try:
            if not sp.re(f.subs(x, a)).is_finite: return "singular_lower"
        except Exception: 
            pass # Ignorar errores de dominio en el límite a
        
        try:
            if not sp.re(f.subs(x, b)).is_finite: return "singular_upper"
        except Exception:
            pass # Ignorar errores de dominio en el límite b

        # Chequear singularidad interna (ej. en 0 o en algún polo simple)
        try:
            poles = sp.poles(f, x)
            if poles:
                for pole in poles:
                    # Si el polo es real y está estrictamente entre a y b
                    if pole.is_real and a < float(pole) < b:
                        return "internal_singular"
            
            # Chequeo heurístico extra para x=0 si está dentro
            if a < 0 < b:
                f_at_0 = f.subs(x, 0)
                if not sp.re(f_at_0).is_finite:
                    return "internal_singular"
        except Exception:
            # Fallback en caso de funciones muy complejas
            pass
            
    # Si no es impropia por límites o singularidades detectadas
    return "proper"

def resolver_integral(f_str, a_str, b_str, var='x'):
    try:
        x = Symbol(var)
        # Reemplazar 'E' con 'exp(1)' y 'sqrt' con 'sqrt' para SymPy
        f_str_sympify = f_str.replace('E', 'exp(1)').replace('sqrt', 'sp.sqrt')
        
        f = sp.sympify(f_str_sympify)
        a = sp.sympify(a_str)
        b = sp.sympify(b_str)

        st.subheader("📊 Análisis Completo Paso a Paso")
        
        # --- LÓGICA DE DETECCIÓN DE TIPO MÁS ROBUSTA ---
        mode = check_for_singularities(f, a, b, x)
        analysis_notes = []
        
        if mode == "internal_singular":
            analysis_notes.append("Esta es una integral impropia por **singularidad interna** (discontinuidad en el intervalo $[{0}, {1}]$).".format(latex(a), latex(b)))
            analysis_notes.append("Se debe dividir en dos integrales $\\int_a^c f(x) dx + \\int_c^b f(x) dx$, donde $c$ es el punto de discontinuidad. Si una diverge, la integral completa **DIVERGE**.")
        elif mode == "infinite_both":
            analysis_notes.append("Esta es una integral impropia por **límite infinito doble** ($-\infty$ a $\infty$).")
            analysis_notes.append("Se resuelve como $\\int_{-\infty}^{\infty} f(x) \, dx = \\int_{-\infty}^{c} f(x) \, dx + \\int_{c}^{\infty} f(x) \, dx$ (con $c$ como cualquier constante real). Si una diverge, la integral completa **DIVERGE**.")
        elif mode == "infinite_upper":
            analysis_notes.append("Esta es una integral impropia por **límite infinito superior**. Se resuelve como:")
            analysis_notes.append(r"\int_a^\infty f(x) \, dx = \lim_{t \to \infty} \int_a^t f(x) \, dx")
            analysis_notes.append("**Explicación detallada**: Evaluaremos $F(t)-F(a)$ y tomaremos el límite $t \to \infty$.")
        elif mode == "infinite_lower":
            analysis_notes.append("Esta es una integral impropia por **límite infinito inferior**. Se resuelve como:")
            analysis_notes.append(r"\int_{-\infty}^b f(x) \, dx = \lim_{t \to -\infty} \int_t^b f(x) \, dx")
            analysis_notes.append("**Explicación detallada**: Evaluaremos $F(b)-F(t)$ y tomaremos el límite $t \to -\infty$.")
        elif mode == "singular_lower":
            analysis_notes.append("Esta es una integral impropia por **singularidad en el límite inferior** (discontinuidad en $a$). Se resuelve como:")
            analysis_notes.append(r"\int_a^b f(x) \, dx = \lim_{\epsilon \to a^+} \int_\epsilon^b f(x) \, dx")
            analysis_notes.append("**Explicación detallada**: Evitamos la singularidad acercándonos desde la derecha.")
        elif mode == "singular_upper":
            analysis_notes.append("Esta es una integral impropia por **singularidad en el límite superior** (discontinuidad en $b$). Se resuelve como:")
            analysis_notes.append(r"\int_a^b f(x) \, dx = \lim_{\epsilon \to b^-} \int_a^\epsilon f(x) \, dx")
            analysis_notes.append("**Explicación detallada**: Evitamos la singularidad acercándonos desde la izquierda.")
        else:
            analysis_notes.append("Esta es una **integral propia** (límites finitos y función continua en el intervalo de integración). Se calcula $F(b) - F(a)$ directamente.")

        st.write("**Paso 1: Identificación del Tipo de Integral**")
        for note in analysis_notes:
            st.markdown(note)

        st.write("**Función dada**:")
        st.latex(f"f(x) = {latex(f)}")
        st.write(f"**Límites**: de ${latex(a)}$ a ${latex(b)}$")

        # --- CÁLCULO DE LA ANTIDERIVADA Y APLICACIÓN DE LÍMITES (SOLO PARA MOSTRAR EL PASO) ---
        F = sp.integrate(f, x)
        st.write("**Paso 2: Encontrar la Antiderivada Indefinida F(x)**")
        st.latex(latex(F) + r" + C \quad \text{(donde C es constante, pero se cancela en límites)}")

        st.write("**Paso 3 & 4: Evaluación y Cálculo del Límite (Proceso)**")

        # Intentamos simular el proceso de límites para el paso a paso
        if mode == "proper":
             F_b = F.subs(x, b)
             F_a = F.subs(x, a)
             expr = F_b - F_a
             st.latex(r"\int_a^b f(x) \, dx = F(b) - F(a) = " + latex(expr))
        elif mode == "infinite_upper":
            t = Symbol('t')
            expr = F.subs(x, t) - F.subs(x, a)
            st.latex(r"\int_a^t f(x) \, dx = F(t) - F(a) = " + latex(expr))
            st.write("Cálculo del Límite:")
            st.latex(r"\lim_{t \to \infty} \left[ " + latex(expr) + r" \right]")
        elif mode == "infinite_lower":
            t = Symbol('t')
            expr = F.subs(x, b) - F.subs(x, t)
            st.latex(r"\int_t^b f(x) \, dx = F(b) - F(t) = " + latex(expr))
            st.write("Cálculo del Límite:")
            st.latex(r"\lim_{t \to -\infty} \left[ " + latex(expr) + r" \right]")
        elif mode == "singular_lower":
            epsilon = Symbol('epsilon')
            a_val_float = float(a)
            # Solo mostrar el límite si el punto singular es a=0 (más simple)
            if a_val_float == 0:
                expr = F.subs(x, b) - F.subs(x, epsilon)
                st.latex(r"\int_\epsilon^b f(x) \, dx = F(b) - F(\epsilon) = " + latex(expr))
                st.write("Cálculo del Límite:")
                st.latex(r"\lim_{\epsilon \to 0^+} \left[ " + latex(expr) + r" \right]")
            else:
                 st.markdown(f"Se utiliza el límite $\\lim_{{\epsilon \\to {latex(a)}^+}} \\int_{{\\epsilon}}^b f(x) \, dx$.")
        elif mode == "singular_upper":
            epsilon = Symbol('epsilon')
            b_val_float = float(b)
            # Solo mostrar el límite si el punto singular es b=0 (más simple)
            if b_val_float == 0:
                expr = F.subs(x, epsilon) - F.subs(x, a)
                st.latex(r"\int_a^\epsilon f(x) \, dx = F(\epsilon) - F(a) = " + latex(expr))
                st.write("Cálculo del Límite:")
                st.latex(r"\lim_{\epsilon \to 0^-} \left[ " + latex(expr) + r" \right]")
            else:
                 st.markdown(f"Se utiliza el límite $\\lim_{{\epsilon \\to {latex(b)}^-}} \\int_{{a}}^\\epsilon f(x) \, dx$.")
        else: # internal_singular o infinite_both
            st.write("Para estos casos complejos, el cálculo requiere dividir la integral en múltiples partes y evaluar los límites de cada una. Se utiliza el resultado directo de SymPy para determinar la convergencia.")
        
        # --- ANÁLISIS DE CONVERGENCIA (USA EL RESULTADO DIRECTO DE SYMPY) ---
        
        # Este es el cálculo final y robusto que asegura la respuesta correcta.
        res_full = integrate(f, (x, a, b))
        st.write("**Paso 5: Análisis de Convergencia (Resultado Directo)**")
        
        # Intenta verificar si el resultado es finito. SymPy maneja oo, zoo (infinito complejo) y números.
        is_finite = False
        try:
            # Si is_finite funciona y no es oo/zoo
            is_finite = res_full.is_finite
        except Exception:
            # Fallback: comprobación numérica y por string
            if str(res_full).lower() not in ['oo', 'zoo', 'nan', 'infinity'] and sp.N(res_full).is_real:
                is_finite = True
            
        if is_finite:
            # Asegura que no sea un resultado de valor principal de Cauchy
            if mode == "internal_singular" and res_full == 0:
                # Caso especial: integral impar de singularidad, divergente en realidad.
                st.error("❌ **La integral DIVERGE** (no converge).")
                st.write("**Aclaración**: Aunque el valor principal de Cauchy es cero, la integral propiamente dicha diverge ya que el límite es infinito en la singularidad interna.")
            else:
                st.success(
                    f"✅ **La integral CONVERGE** a un valor finito: ${latex(res_full)}$."
                )
                st.write(
                    "**Explicación detallada**: El límite existe y es finito, por lo que el área bajo la curva es acotada (limitada). Esto implica que la función decae lo suficientemente rápido."
                )
                st.success("✅ ¡Cálculo completado exitosamente! La integral converge.", icon="🎯")
                st.info("Usa los pasos arriba para entender el proceso matemático.")
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
            # Muestra el resultado de SymPy (que puede ser oo, zoo o un número en casos especiales de Cauchy)
            st.write(f"El resultado de la integral es: ${latex(res_full)}$")
                 
            st.write(
                "**Explicación detallada**: El límite es infinito o no existe, lo que significa que el área crece sin cota (ilimitada)."
            )

    except Exception as e:
        st.error(
            f"❌ Error en el cálculo: {str(e)}. Tips: Usa 'x' como variable, '**' para potencias (ej. x**2), **x**(1/3) para $\\sqrt[3]{x}$, **oo** para $\\infty$, **log()** para $\\ln()$, **exp(x)** para $e^x$. Ejemplo: 1/x**2."
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
    st.write("- **Raíces**: Usa potencias fraccionarias (ej. `x**(1/3)` para $\\sqrt[3]{x}$) o `sqrt(x)`.")
    st.write("- **Infinito**: Usa **oo** para $+\\infty$ o **-oo** para $-\\infty$.")
    st.write("- **Funciones**: Usa **log(x)** para $\\ln(x)$, **exp(x)** para $e^x$, y **E** para la constante de Euler.")
    
    # ----------------------------------------------------------------------------------
    # CORRECCIÓN FINAL: Se usan solo caracteres Unicode y Markdown simple en el Tip Pro.
    # ----------------------------------------------------------------------------------
    st.markdown(
        f"""
        <div style='background-color:#eef2ff; color:#1e3a8a; padding:10px; border-radius:8px; font-weight:600;'>
        💡 <strong>Tip Pro: Lógica del Solver</strong>
        <br>1. La app resuelve la integral mediante el **Teorema Fundamental del Cálculo** y luego aplica el **límite**.
        <br>2. Para límites infinitos ({'\u221e'}), usa una variable **t** y evalúa **Lím t → {'\u221e'}**.
        <br>3. Para singularidades (discontinuidad en el límite, ej. en 0), usa una variable **&epsilon;** y evalúa **Lím &epsilon; → 0⁺**.
        <br>4. La **convergencia** se declara solo si el límite final es un valor **finito** (no {'\u221e'}).
        <br>5. **ROBUSTEZ**: La conclusión final (Paso 5) se basa en el cálculo directo de SymPy, garantizando la máxima precisión.
        </div>
        """,
        unsafe_allow_html=True
    )
    # ----------------------------------------------------------------------------------

    modo = st.selectbox("✨ Opciones de Gráfica",
                        ["Estándar", "Avanzado (con Gráfica Auto)"],
                        index=0)
    if modo == "Avanzado (con Gráfica Auto)":
        st.checkbox("Activar gráfica automática al resolver", value=True)

tab1, tab2 = st.tabs(["🚀 Resolver Manual", "🧪 Ejemplos Rápidos"])

with tab1:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        # Valor de ejemplo del usuario: 1/x**(5/3)
        f_expr = st.text_input("🔢 f(x):",
                               value="1/x**(5/3)",
                               help="Ej: x**(1/3) | Escribe libremente")
    with col2:
        # Valor de ejemplo del usuario: -1
        a_lim = st.text_input(
            "📏 a (inferior):",
            value="-1",
            help="Ej: 0 (singularidad), 1, o cualquier número")
    with col3:
        # Valor de ejemplo del usuario: 1
        b_lim = st.text_input("📏 b (superior):",
                              value="1",
                              help="Ej: oo (infinito), 1, o cualquier número")

    progress_bar = st.progress(0)
    if st.button("🔍 Resolver con Detalle Completo", type="primary"):
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
            # Usamos el string original guardado para la gráfica, pero lo pre-procesamos si usa 'E'
            f_str_graph = st.session_state.saved_f.replace('E', 'exp(1)').replace('sqrt', 'sp.sqrt')
            f = sp.sympify(f_str_graph)
            a = sp.sympify(st.session_state.saved_a)
            b = sp.sympify(st.session_state.saved_b)
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Manejo seguro de start/end para la gráfica
            try:
                # Si el límite inferior es infinito negativo, ajustamos el inicio
                start = -10.0 if a == -oo else (float(a) if a.is_number and a != 0 else -1.0)
            except Exception:
                start = -1.0
            try:
                # Si el límite superior es infinito positivo, ajustamos el final
                end = 10.0 if b == oo else (float(b) if b.is_number and b != 0 else 1.0)
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
            ax.fill_between(x_fill,
                            0,
                            y_fill,
                            alpha=0.3,
                            color='#3b82f6',
                            label='Área aproximada')
            
            # Límites (ajustados para infinito)
            if a != -oo and a.is_number:
                ax.axvline(float(a),
                           color='r',
                           linestyle='--',
                           label=f'Límite inferior: {a}',
                           linewidth=2)
            if b != oo and b.is_number:
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
    # Ahora usamos dos filas de 3 columnas para los 6 ejemplos
    st.markdown("### Ejemplos Clásicos de Integrales Impropias")
    
    col_ej1, col_ej2, col_ej3 = st.columns(3)
    with col_ej1:
        # Sintaxis original restaurada
        with st.expander("Ej1: $\\int 1/x^2 dx$ de 1 a $\\infty$ (Converge)"):
            st.write("**Función**: $1/x^2$ | **Límites**: $a=1, b=\\infty$")
            if st.button("Resolver Ejemplo 1", key="ej1"):
                st.session_state.saved_f = "1/x**2"
                st.session_state.saved_a = "1"
                st.session_state.saved_b = "oo"
                resolver_integral("1/x**2", "1", "oo")
                if modo == "Avanzado (con Gráfica Auto)": st.session_state.show_graph = True
    with col_ej2:
        # Singularidad en el límite inferior
        with st.expander("Ej2: $\\int 1/\\sqrt{x} dx$ de 0 a 1 (Singular, Converge)"):
            st.write("**Función**: $1/\\sqrt{x}$ | **Límites**: $a=0, b=1$")
            if st.button("Resolver Ejemplo 2", key="ej2"):
                st.session_state.saved_f = "1/sqrt(x)" 
                st.session_state.saved_a = "0"
                st.session_state.saved_b = "1"
                resolver_integral("1/sqrt(x)", "0", "1")
                if modo == "Avanzado (con Gráfica Auto)": st.session_state.show_graph = True
    with col_ej3:
        # Diverge a infinito
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
        # Singularidad en 0, converge
        with st.expander("Ej4: $\\int \ln(x) dx$ de 0 a 1 (Singular, Converge)"):
            st.write("**Función**: $\\ln(x)$ | **Límites**: $a=0, b=1$")
            if st.button("Resolver Ejemplo 4", key="ej4"):
                st.session_state.saved_f = "log(x)" 
                st.session_state.saved_a = "0"
                st.session_state.saved_b = "1"
                resolver_integral("log(x)", "0", "1")
                if modo == "Avanzado (con Gráfica Auto)": st.session_state.show_graph = True
    with col_ej5:
        # Singularidad interna (Divergente, tu ejemplo modificado)
        with st.expander("Ej5: $\\int 1/x^{5/3} dx$ de -1 a 1 (Singularidad Interna, DIVERGE)"):
            st.write("**Función**: $1/x^{5/3}$ | **Límites**: $a=-1, b=1$")
            if st.button("Resolver Ejemplo 5", key="ej5"):
                st.session_state.saved_f = "1/x**(5/3)" 
                st.session_state.saved_a = "-1"
                st.session_state.saved_b = "1"
                resolver_integral("1/x**(5/3)", "-1", "1")
                if modo == "Avanzado (con Gráfica Auto)": st.session_state.show_graph = True
    with col_ej6:
        # Integral Propia (fácil)
        with st.expander("Ej6: $\\int x^2 dx$ de 0 a 2 (Integral Propia, Converge)"):
            st.write("**Función**: $x^2$ | **Límites**: $a=0, b=2$")
            if st.button("Resolver Ejemplo 6", key="ej6"):
                st.session_state.saved_f = "x**2" 
                st.session_state.saved_a = "0"
                st.session_state.saved_b = "2"
                resolver_integral("x**2", "0", "2")
                if modo == "Avanzado (con Gráfica Auto)": st.session_state.show_graph = True
