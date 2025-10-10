import streamlit as st
import sympy as sp
from sympy import limit, oo, Symbol, integrate, latex, lambdify, exp, E
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

def resolver_integral(f_str, a_str, b_str, var='x'):
    try:
        x = Symbol(var)
        # Reemplazar 'E' con 'exp(1)' para garantizar que SymPy lo reconozca como constante de Euler en la entrada del usuario.
        f_str_sympify = f_str.replace('E', 'exp(1)')
        
        f = sp.sympify(f_str_sympify)
        a = sp.sympify(a_str)
        b = sp.sympify(b_str)

        st.subheader("📊 Análisis Completo Paso a Paso")

        st.write("**Paso 1: Identificación del Tipo de Integral**")
        mode = None
        if b == oo:
            st.markdown(
                "Esta es una integral impropia por **límite infinito superior**. Se resuelve como:"
            )
            st.latex(r"\int_a^\infty f(x) \, dx = \lim_{t \to \infty} \int_a^t f(x) \, dx")
            st.write(
                "**Explicación detallada**: Evaluaremos F(t)-F(a) y tomaremos el límite $t \to \infty$."
            )
            mode = "infinite_upper"
        elif a == 0:
            st.markdown(
                "Esta es una integral impropia por **singularidad en el límite inferior** (ej. discontinuidad en $x=0$). Se resuelve como:"
            )
            st.latex(r"\int_0^b f(x) \, dx = \lim_{\epsilon \to 0^+} \int_\epsilon^b f(x) \, dx")
            st.write(
                "**Explicación detallada**: Evitamos la singularidad acercándonos a 0 desde la derecha."
            )
            mode = "singular_lower"
        else:
            st.write(
                "Esta es una **integral propia** (límites finitos y función continua en el intervalo). Se calcula $F(b) - F(a)$."
            )
            mode = "proper"

        st.write("**Función dada**:")
        st.latex(f"f(x) = {latex(f)}")
        st.write(f"**Límites**: de ${latex(a)}$ a ${latex(b)}$")

        st.write("**Paso 2: Encontrar la Antiderivada Indefinida F(x)**")
        st.write(
            "**Explicación detallada**: La antiderivada F(x) satisface $F'(x) = f(x)$. SymPy la calcula automáticamente, pero veamos el proceso:"
        )
        st.write(
            "- Si $f(x) = 1/x^n$ ($n>1$), $\\int = -1/( (n-1) x^{n-1} )$ por regla de potencias."
        )
        st.write(
            "- Para otros, usa sustitución o integración por partes si es necesario."
        )

        F = sp.integrate(f, x)
        st.latex(latex(F) + r" + C \quad \text{(donde C es constante, pero se cancela en límites)}")
        st.write("**Paso 3: Evaluar la Integral Definida Usando el Teorema Fundamental**")
        if mode == "infinite_upper":
            st.write("Evaluamos: $[F(t) - F(a)]$ y tomamos el límite $t\to\infty$")
            t = Symbol('t')
            F_t = F.subs(x, t)
            F_a = F.subs(x, a)
            st.write("**Subpaso 3.1: Sustituir en la antiderivada para el límite superior variable (t)**")
            st.latex(latex(F_t))
            st.write("**Explicación**: Reemplazamos x = t en F(x) para obtener el valor en el límite superior.")
            st.write("**Subpaso 3.2: Sustituir en la antiderivada para el límite inferior fijo (a)**")
            st.latex(latex(F_a))
            st.write("**Explicación**: Reemplazamos x = a en F(x) para obtener el valor en el límite inferior.")
            st.write("**Subpaso 3.3: Aplicar el Teorema Fundamental - Restar para formar la expresión de la integral**")
            st.write("Por el Teorema Fundamental del Cálculo, la integral definida es la diferencia de la antiderivada en los límites.")
            expr = F_t - F_a
            st.latex(r"\int_a^t f(x) \, dx = F(t) - F(a) = " + latex(expr))
            res = limit(expr, t, oo)
        elif mode == "singular_lower":
            st.write("Evaluamos: $[F(b) - F(\epsilon)]$ y tomamos el límite $\\epsilon\to 0^+$")
            epsilon = Symbol('epsilon')
            F_b = F.subs(x, b)
            F_epsilon = F.subs(x, epsilon)
            st.write("**Subpaso 3.1: Sustituir en la antiderivada para el límite superior fijo (b)**")
            st.latex(latex(F_b))
            st.write("**Explicación**: Reemplazamos x = b en F(x).")
            st.write("**Subpaso 3.2: Sustituir en la antiderivada para el límite inferior variable (ε)**")
            st.latex(latex(F_epsilon))
            st.write("**Explicación**: Reemplazamos x = ε para evitar la singularidad en 0.")
            st.write("**Subpaso 3.3: Aplicar el Teorema Fundamental - Restar para formar la expresión**")
            st.write("La integral de ε a b es F(b) - F(ε).")
            expr = F_b - F_epsilon
            st.latex(r"\int_\epsilon^b f(x) \, dx = F(b) - F(\epsilon) = " + latex(expr))
            res = limit(expr, epsilon, 0, dir='+')
        else:
            st.write("Evaluamos: $F(b) - F(a)$ (integral propia, sin límites variables)")
            F_b = F.subs(x, b)
            F_a = F.subs(x, a)
            st.write("**Subpaso 3.1: Sustituir en la antiderivada para el límite superior (b)**")
            st.latex(latex(F_b))
            st.write("**Explicación**: Reemplazamos x = b en F(x).")
            st.write("**Subpaso 3.2: Sustituir en la antiderivada para el límite inferior (a)**")
            st.latex(latex(F_a))
            st.write("**Explicación**: Reemplazamos x = a en F(x).")
            st.write("**Subpaso 3.3: Aplicar el Teorema Fundamental - Restar para el valor exacto**")
            st.write("No se necesita límite (integral propia directa). El valor es la expresión simplificada.")
            expr = F_b - F_a
            st.latex(r"\int_a^b f(x) \, dx = F(b) - F(a) = " + latex(expr))
            res = sp.simplify(expr)

        st.write("**Paso 4: Calcular el Límite**")
        if mode == "infinite_upper":
            st.write("Tomamos el límite de la expresión cuando $t \to \infty$.")
            st.latex(r"\lim_{t \to \infty} \left[ " + latex(expr) + r" \right]")
            st.write("**Explicación detallada del cálculo**: Analizamos término por término. Los constantes quedan iguales, y los términos que crecen con t (o dependen de 1/t) tienden a 0 si la función decae rápido (ej. para $1/x^2$, $1/t \to 0$ cuando $t \to \infty$, así que queda el valor constante).")
        elif mode == "singular_lower":
            st.write("Tomamos el límite de la expresión cuando $\\epsilon \to 0^+$.")
            st.latex(r"\lim_{\epsilon \to 0^+} \left[ " + latex(expr) + r" \right]")
            st.write("**Explicación detallada del cálculo**: Verificamos el comportamiento cerca de $\\epsilon=0$. Si hay término como $1/\sqrt{\\epsilon}$, diverge a $\\infty$; si converge, el límite es finito.")
        else:
            st.write("No se necesita límite (integral propia directa). El valor es la expresión simplificada.")
            st.latex(latex(expr))
        st.latex(r"\text{Resultado del Límite: } " + latex(res))
        st.write("**Paso 5: Análisis de Convergencia**")
        # En algunos casos res puede ser sympy oo o nan; manejamos lo más robusto posible
        try:
            is_finite = res.is_finite
        except Exception:
            # Si res no tiene is_finite, lo convertimos a string y comprobamos heurísticamente
            is_finite = False
            try:
                if str(res).lower() not in ['oo', 'zoo', 'nan', 'infinity', 'nan']:
                    is_finite = True
            except Exception:
                is_finite = False

        if is_finite:
            st.success(
                f"✅ **La integral CONVERGE** a un valor finito: ${latex(res)}$."
            )
            st.write(
                "**Explicación detallada**: El límite existe y es finito, por lo que el área bajo la curva es acotada (limitada). Esto implica que la función decae lo suficientemente rápido (ej. como $1/x^2$ o mejor)."
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
            st.write(
                "**Explicación detallada**: El límite es infinito o no existe, lo que significa que el área crece sin cota (ej. función decae lento como $1/x$). Usa pruebas como comparación o p-test para confirmar."
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
    st.write("- **Raíces**: Usa potencias fraccionarias (ej. `x**(1/3)` para $\\sqrt[3]{x}$).")
    st.write("- **Infinito**: Usa **oo** para $+\\infty$.")
    st.write("- **Funciones**: Usa **log(x)** para $\\ln(x)$, **sqrt(x)** para $\\sqrt{x}$, **exp(x)** para $e^x$, y **E** para la constante de Euler.")
    
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
        f_expr = st.text_input("🔢 f(x):",
                               value="x**(1/3)",
                               help="Ej: x**(1/3) | Escribe libremente")
    with col2:
        a_lim = st.text_input(
            "📏 a (inferior):",
            value="0",
            help="Ej: 0 (singularidad), 1, o cualquier número")
    with col3:
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
            f_str_graph = st.session_state.saved_f.replace('E', 'exp(1)') 
            f = sp.sympify(f_str_graph)
            a = sp.sympify(st.session_state.saved_a)
            b = sp.sympify(st.session_state.saved_b)
            fig, ax = plt.subplots(figsize=(10, 6))
            # Manejo seguro de start/end para la gráfica
            try:
                # Si el límite inferior es infinito negativo, ajustamos el inicio
                start = -10.0 if a == -oo else (0.01 if a == 0 else float(a))
            except Exception:
                start = 0.01
            try:
                # Si el límite superior es infinito positivo, ajustamos el final
                end = 10.0 if b == oo else float(b)
            except Exception:
                end = 10.0

            # Aseguramos que 'start' sea menor que 'end'
            if start >= end:
                end = start + 5.0

            x_vals = np.linspace(start, end, 200)
            
            # Evaluar f(x) numéricamente con lambdify y manejo robusto de errores
            try:
                f_np = lambdify(x_sym, f, 'numpy')
                y_vals_raw = f_np(x_vals)

                # --- FIX: Manejo Robusto para Gráficas (Evita errores de Dominio/Números Complejos) ---
                # 1. Convertir a valores reales (esencial para funciones como sqrt o log)
                y_vals = np.real(y_vals_raw)

                # 2. Reemplazar valores no finitos (NaN/Inf) con NaN para que Matplotlib no dibuje líneas
                y_vals[~np.isfinite(y_vals)] = np.nan

                # 3. Limitar valores extremos para una gráfica legible (clipping)
                max_y_limit = 100.0
                y_vals = np.clip(y_vals, -max_y_limit, max_y_limit)
                # -----------------------------------------------------------------------------------

            except Exception as e:
                st.error(f"❌ Error de Dominio en Gráfica: {e}. Esto sucede cuando la función (ej. sqrt) se evalúa fuera de su dominio. Mostrando solo el eje.")
                y_vals = np.zeros_like(x_vals) # Fallback

            ax.plot(x_vals,
                    y_vals,
                    label=f"f(x) = {st.session_state.saved_f}",
                    color='#3b82f6',
                    linewidth=2)
            # Sombreado para área bajo la curva
            ax.fill_between(x_vals,
                            0,
                            y_vals,
                            alpha=0.3,
                            color='#3b82f6',
                            label='Área aproximada')
            ax.axvline(start,
                       color='r',
                       linestyle='--',
                       label=f'Límite inferior: {a}',
                       linewidth=2)
            if b != oo:
                ax.axvline(end,
                           color='g',
                           linestyle='--',
                           label=f'Límite superior: {b}',
                           linewidth=2)
            ax.set_title(
                "🔍 Gráfica Interactiva: Visualiza el Área de la Integral",
                fontsize=16,
                color='#1e3a8a')
            ax.set_xlabel("x", fontsize=12)
            ax.set_ylabel("f(x)", fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            # Ajustar el límite y para que el eje x se vea bien
            # Usamos nanmin/nanmax con guard rails para evitar errores si todo es nan
            try:
                y_min = np.nanmin(y_vals)
                y_max = np.nanmax(y_vals)
                if np.isnan(y_min) or np.isnan(y_max):
                    ax.set_ylim(-1, 1)
                else:
                    ax.set_ylim(min(0, y_min), max(0, y_max))
            except Exception:
                ax.set_ylim(-1, 1)

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
        # Sintaxis original restaurada
        with st.expander("Ej2: $\\int 1/\\sqrt{x} dx$ de 0 a 1 (Singular, Converge)"):
            st.write("**Función**: $1/\\sqrt{x}$ | **Límites**: $a=0, b=1$")
            if st.button("Resolver Ejemplo 2", key="ej2"):
                st.session_state.saved_f = "1/sqrt(x)" 
                st.session_state.saved_a = "0"
                st.session_state.saved_b = "1"
                resolver_integral("1/sqrt(x)", "0", "1")
                if modo == "Avanzado (con Gráfica Auto)": st.session_state.show_graph = True
    with col_ej3:
        # Sintaxis original restaurada
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
        # Logaritmo Natural
        with st.expander("Ej4: $\\int \ln(x) dx$ de 0 a 1 (Singular, Converge)"):
            st.write("**Función**: $\\ln(x)$ | **Límites**: $a=0, b=1$")
            if st.button("Resolver Ejemplo 4", key="ej4"):
                st.session_state.saved_f = "log(x)" 
                st.session_state.saved_a = "0"
                st.session_state.saved_b = "1"
                resolver_integral("log(x)", "0", "1")
                if modo == "Avanzado (con Gráfica Auto)": st.session_state.show_graph = True
    with col_ej5:
        # Raíz Cúbica (Potencia Fraccionaria)
        with st.expander("Ej5: $\\int 1/\\sqrt[3]{x} dx$ de 0 a 8 (Raíz Cúbica, Converge)"):
            # Raíz cúbica es x^(-1/3)
            st.write("**Función**: $x^{-1/3}$ | **Límites**: $a=0, b=8$")
            if st.button("Resolver Ejemplo 5", key="ej5"):
                st.session_state.saved_f = "x**(-1/3)" 
                st.session_state.saved_a = "0"
                st.session_state.saved_b = "8"
                resolver_integral("x**(-1/3)", "0", "8")
                if modo == "Avanzado (con Gráfica Auto)": st.session_state.show_graph = True
    with col_ej6:
        # Exponencial
        with st.expander("Ej6: $\\int e^{-x} dx$ de 1 a $\\infty$ (Exponencial, Converge)"):
            # Exponencial es exp(-x)
            st.write("**Función**: $e^{-x}$ | **Límites**: $a=1, b=\\infty$")
            if st.button("Resolver Ejemplo 6", key="ej6"):
                st.session_state.saved_f = "exp(-x)" 
                st.session_state.saved_a = "1"
                st.session_state.saved_b = "oo"
                resolver_integral("exp(-x)", "1", "oo")
                if modo == "Avanzado (con Gráfica Auto)": st.session_state.show_graph = True
