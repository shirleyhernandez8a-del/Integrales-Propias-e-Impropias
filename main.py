streamlit
sympy
matplotlib
numpy

import streamlit as st
import sympy as sp
from sympy import limit, oo, Symbol, integrate, latex
import matplotlib.pyplot as plt
import numpy as np

# *** ADICIÓN: Tema personalizado para diseño lindo (azul matemático) ***
st.set_page_config(
    page_title="Solver de Integrales Impropias Detallado",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"  # Sidebar abierta por default
)

# *** ADICIÓN: Configuración de tema (colores pro) ***
st.markdown("""
    <style>
    .main {background-color: #f0f8ff;}  /* Fondo azul claro */
    .stApp {background-color: #f0f8ff;}
    .sidebar .sidebar-content {background-color: #e6f3ff;}  /* Sidebar azul */
    h1 {color: #1e3a8a; text-align: center; font-family: 'Arial Black';}  /* Título bold */
    .stButton > button {background-color: #3b82f6; color: white; border-radius: 10px;}  /* Botones redondeados azules */
    </style>
""",
            unsafe_allow_html=True)

# *** ADICIÓN: Header creativo con emoji y descripción ***
st.markdown("---")
st.markdown(
    "<h1 style='text-align: center; color: #1e3a8a;'>🧮 Solver de Integrales Impropias - Paso a Paso Detallado</h1>",
    unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #4b5563; font-size: 18px;'>Ingresa la función y límites. La app explica **cada subpaso** antes de la respuesta final: tipo, cálculo de antiderivada, evaluación del límite y análisis de convergencia. ¡Visualiza el área y converge a la excelencia! 🎓</p>",
    unsafe_allow_html=True)
st.markdown("---")


# Tu código original (intacto)
def resolver_integral(f_str, a_str, b_str, var='x'):
    try:
        x = Symbol(var)
        f = sp.sympify(f_str)
        a = sp.sympify(a_str)
        b = sp.sympify(b_str)

        st.subheader("📊 Análisis Completo Paso a Paso")

        # Paso 1: Identificar tipo con explicación detallada
        st.write("**Paso 1: Identificación del Tipo de Integral**")
        mode = None
        if b == oo:
            # *** CORRECCIÓN APLICADA: Usar st.markdown para texto ***
            st.markdown(
                "Esta es una integral impropia por **límite infinito superior**. Se resuelve como:"
            )
            st.latex(
                r"$$\int_a^\infty f(x) \, dx = \lim_{t \to \infty} \int_a^t f(x) \, dx$$"
            )
            st.write(
                "**Explicación detallada**: Evaluaremos F(t)-F(a) y tomaremos el límite $t \to \infty$."
            )
            mode = "infinite_upper"
        elif a == 0:
            # *** CORRECCIÓN APLICADA: Usar st.markdown para texto ***
            st.markdown(
                "Esta es una integral impropia por **singularidad en el límite inferior** (ej. discontinuidad en $x=0$). Se resuelve como:"
            )
            st.latex(
                r"$$\int_0^b f(x) \, dx = \lim_{\epsilon \to 0^+} \int_\epsilon^b f(x) \, dx$$"
            )
            st.write(
                "**Explicación detallada**: Evitamos la singularidad acercándonos a 0 desde la derecha."
            )
            mode = "singular_lower"
        else:  # proper
            # Si no es ninguno de los anteriores, la tratamos como integral propia y la calculamos normalmente
            st.write(
                "Esta es una **integral propia** (límites finitos y función continua en el intervalo). Se calcula $F(b) - F(a)$."
            )
            mode = "proper"

        st.write("**Función dada**:")
        st.latex(f"f(x) = {latex(f)}")
        st.write(f"**Límites**: de ${latex(a)}$ a ${latex(b)}$")

        # Paso 2: Cálculo de antiderivada con subpasos
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
        # *** CORRECCIÓN APLICADA: Usar \text{} en LaTeX para el texto descriptivo y evitar que se peguen las palabras ***
        st.latex(
            f"F(x) = {latex(F)} + C \\quad \\text{{(donde C es constante, pero se cancela en límites)}}"
        )

        # Evaluar integral definida con límites
        st.write(
            "**Paso 3: Evaluar la Integral Definida Usando el Teorema Fundamental**"
        )
        if mode == "infinite_upper":
            st.write(
                "Evaluamos: $[F(t) - F(a)]$ y tomamos $\lim_{t\\to\\infty}$")
            t = Symbol('t')
            expr = F.subs(x, t) - F.subs(x, a)
            st.latex(f"Expresión: {latex(expr)}")
            res = limit(expr, t, oo)
        elif mode == "singular_lower":
            st.write(
                "Evaluamos: $[F(b) - F(\epsilon)]$ y tomamos $\lim_{\epsilon\\to 0^+}$"
            )
            epsilon = Symbol('epsilon')
            expr = F.subs(x, b) - F.subs(x, epsilon)
            st.latex(f"Expresión: {latex(expr)}")
            res = limit(expr, epsilon, 0, dir='+')
        else:  # proper
            st.write("Evaluamos: $F(b) - F(a)$ (integral propia)")
            expr = F.subs(x, b) - F.subs(x, a)
            st.latex(f"Expresión: {latex(expr)}")
            res = sp.simplify(expr)

        st.write("**Paso 4: Calcular el Límite**")
        st.latex(f"Resultado del límite: {latex(res)}")

        # Paso 5: Análisis de convergencia con detalle
        st.write("**Paso 5: Análisis de Convergencia**")
        if res.is_finite:
            st.success(
                f"✅ **La integral CONVERGE** a un valor finito: ${latex(res)}$."
            )
            st.write(
                "**Explicación detallada**: El límite existe y es finito, por lo que el área bajo la curva es acotada (limitada). Esto implica que la función decae lo suficientemente rápido (ej. como $1/x^2$ o mejor)."
            )
            # *** ADICIÓN: Efecto wow para éxito ***
            st.balloons()  # Confetti virtual al converger
        else:
            st.error("❌ **La integral DIVERGE** (no converge).")
            st.write(
                "**Explicación detallada**: El límite es infinito o no existe, lo que significa que el área crece sin cota (ej. función decae lento como $1/x$). Usa pruebas como comparación o p-test para confirmar."
            )

        # Gráfica opcional (mejorada con área shaded)
        if st.checkbox(
                "📈 Mostrar Gráfica de f(x) (Área Bajo la Curva Visualizada)"):
            fig, ax = plt.subplots(figsize=(10, 6))
            # Manejo seguro de start/end para la gráfica
            try:
                start = 0.01 if a == 0 else float(a)
            except Exception:
                start = 0.01
            try:
                end = 10.0 if b == oo else float(b)
            except Exception:
                end = 10.0

            x_vals = np.linspace(start, end, 200)
            y_vals = []
            for val in x_vals:
                try:
                    y_vals.append(float(f.subs(x, val)))
                except:
                    y_vals.append(0)  # Manejo de singularidades
            ax.plot(x_vals,
                    y_vals,
                    label=f"f(x) = {f_str}",
                    color='#3b82f6',
                    linewidth=2)
            # *** ADICIÓN: Sombreado para área bajo la curva (wow visual) ***
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
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    except Exception as e:
        st.error(
            f"❌ Error en el cálculo: {str(e)}. Tips: Usa 'x' como variable, '**' para potencias (ej. x**2), 'oo' para $\\infty$. Ejemplo: 1/x**2."
        )


# *** ADICIÓN: Sidebar mejorada (más creativa con selectbox y tips) ***
with st.sidebar:
    st.header("🎛️ Panel de Control Creativo")
    st.markdown("### 📖 Guía Rápida")
    st.write(
        "- **f(x)**: Expresa en términos de x (ej. 1/x**2, sin(x)/x, e**(-x))."
    )
    st.write("- **a**: Límite inferior (0 para singularidad).")
    st.write("- **b**: Límite superior (oo para infinito).")
    st.info(
        "**Tip Pro**: Escribe libremente en los campos (ej. 'oo' para ∞). ¡La gráfica shaded muestra el 'área' que converge!"
    )

    # *** ADICIÓN: Selector de 'modo' para más interactividad ***
    modo = st.selectbox("🌟 Modo de Visualización",
                        ["Estándar", "Avanzado (con Gráfica Auto)"],
                        index=0)
    if modo == "Avanzado (con Gráfica Auto)":
        st.checkbox("Activar gráfica automática al resolver", value=True)

# *** ADICIÓN: Tabs para organización creativa (Inputs | Ejemplos) ***
tab1, tab2 = st.tabs(["🚀 Resolver Manual", "🧪 Ejemplos Rápidos"])

with tab1:
    # *** CAMBIO: Inputs todos como text_input para flexibilidad total ***
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        f_expr = st.text_input("🔢 f(x):",
                               value="1/x**2",
                               help="Ej: 1/x**2 | Escribe libremente")
    with col2:
        a_lim = st.text_input(
            "📏 a (inferior):",
            value="1",
            help="Ej: 0 (singularidad), 1, o cualquier número")
    with col3:
        b_lim = st.text_input("📏 b (superior):",
                              value="oo",
                              help="Ej: oo (infinito), 1, o cualquier número")

    # *** ADICIÓN: Barra de progreso para simular cálculo (wow) ***
    progress_bar = st.progress(0)
    if st.button("🔍 Resolver con Detalle Completo", type="primary"):
        for i in range(100):
            progress_bar.progress(i + 1)
            # Simula carga
        resolver_integral(f_expr, a_lim, b_lim)
        # *** ADICIÓN: Auto gráfica si modo avanzado ***
        if modo == "Avanzado (con Gráfica Auto)":
            st.rerun()  # Refresca para mostrar checkbox checked

with tab2:
    st.subheader("🧪 Ejemplos Pre-cargados (Clic para Ver Pasos Detallados)")
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    with col_ex1:
        if st.button("Ej1: ∫ 1/x² dx de 1 a ∞", use_container_width=True):
            with st.expander("🔓 Revelar Pasos Detallados"
                             ):  # *** ADICIÓN: Expander para no saturar ***
                resolver_integral("1/x**2", "1", "oo")
    with col_ex2:
        if st.button("Ej2: ∫ 1/√x dx de 0 a 1", use_container_width=True):
            with st.expander("🔓 Revelar Pasos Detallados"):
                resolver_integral("1/sqrt(x)", "0", "1")
    with col_ex3:
        if st.button("Ej3: ∫ 1/x dx de 1 a ∞ (Diverge)",
                     use_container_width=True):
            with st.expander("🔓 Revelar Pasos Detallados"):
                resolver_integral("1/x", "1", "oo")

# *** ADICIÓN: Footer creativo ***
st.markdown("---")
col_footer1, col_footer2 = st.columns(2)
with col_footer1:
    st.caption(
        "👨‍💻 Desarrollado con ❤️ usando Streamlit y SymPy. ¡Proyecto para [tu nombre/clase]!"
    )
with col_footer2:
    st.caption(
        "📚 Para más info: [Khan Academy Integrales](https://www.khanacademy.org/math) | Versión 2.0 - Diseño Premium"
    )
st.markdown("---")
