import streamlit as st
import sympy as sp
from sympy import limit, oo, Symbol, integrate, latex, lambdify  # *** FIX: Agregué lambdify para gráfica robusta ***
import matplotlib
matplotlib.use('Agg')  # *** FIX: Backend para cloud – evita errores en gráfica ***
import matplotlib.pyplot as plt
import numpy as np

# *** FIX MÍNIMO: Inicializar session_state para gráfica persistente (recuerda checkbox y datos) ***
if "show_graph" not in st.session_state:
    st.session_state.show_graph = False
if "saved_f" not in st.session_state:
    st.session_state.saved_f = ""
if "saved_a" not in st.session_state:
    st.session_state.saved_a = ""
if "saved_b" not in st.session_state:
    st.session_state.saved_b = ""

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
    .main {background-color: #f0f8ff;} /* Fondo azul claro */
    .stApp {background-color: #f0f8ff;}
    .sidebar .sidebar-content {background-color: #e6f3ff;} /* Sidebar azul */
    h1 {color: #1e3a8a; text-align: center; font-family: 'Arial Black';} /* Título bold */
    .stButton > button {background-color: #3b82f6; color: white; border-radius: 10px;} /* Botones redondeados azules */
    </style>
""", unsafe_allow_html=True)

# *** ADICIÓN: Header creativo con emoji y descripción ***
st.markdown("---")
st.markdown(
    "<h1 style='text-align: center; color: #1e3a8a;'>🧮 Solver de Integrales Impropias - Paso a Paso Detallado</h1>",
    unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #4b5563; font-size: 18px;'>Ingresa la función y límites. La app explica **cada subpaso** antes de la respuesta final: tipo, cálculo de antiderivada, evaluación del límite y análisis de convergencia. ¡Visualiza el área y converge a la excelencia! 🎓</p>",
    unsafe_allow_html=True)
st.markdown("---")

# Tu código original (intacto, con fixes solo en Paso 3/4)
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

        # *** FIX: Paso 3 Mejorado con Subpasos Detallados + LaTeX Display (Renderiza Perfecto, Explica de Dónde Sale la Expr) ***
        st.write("**Paso 3: Evaluar la Integral Definida Usando el Teorema Fundamental**")
        if mode == "infinite_upper":
            st.write("Evaluamos: $[F(t) - F(a)]$ y tomamos $\lim_{t\\to\\infty}$")
            t = Symbol('t')
            # Subpasos detallados: De dónde sale cada parte
            F_t = F.subs(x, t)
            F_a = F.subs(x, a)
            st.write("**Subpaso 3.1: Sustituir en la antiderivada para el límite superior variable (t)**")
            st.latex(r"$$F(t) = " + latex(F_t) + r"$$")
            st.write("**Explicación**: Reemplazamos x = t en F(x) para obtener el valor en el límite superior.")
            st.write("**Subpaso 3.2: Sustituir en la antiderivada para el límite inferior fijo (a)**")
            st.latex(r"$$F(a) = " + latex(F_a) + r"$$")
            st.write("**Explicación**: Reemplazamos x = a en F(x) para obtener el valor en el límite inferior.")
            st.write("**Subpaso 3.3: Aplicar el Teorema Fundamental - Restar para formar la expresión de la integral**")
            st.write("Por el Teorema Fundamental del Cálculo, la integral definida es la diferencia de la antiderivada en los límites.")
            expr = F_t - F_a
            st.latex(r"$$\int_a^t f(x) \, dx = F(t) - F(a) = " + latex(expr) + r"$$")
            res = limit(expr, t, oo)
        elif mode == "singular_lower":
            st.write("Evaluamos: $[F(b) - F(\epsilon)]$ y tomamos $\lim_{\epsilon\\to 0^+}$")
            epsilon = Symbol('epsilon')
            # Subpasos detallados
            F_b = F.subs(x, b)
            F_epsilon = F.subs(x, epsilon)
            st.write("**Subpaso 3.1: Sustituir en la antiderivada para el límite superior fijo (b)**")
            st.latex(r"$$F(b) = " + latex(F_b) + r"$$")
            st.write("**Explicación**: Reemplazamos x = b en F(x).")
            st.write("**Subpaso 3.2: Sustituir en la antiderivada para el límite inferior variable (ε)**")
            st.latex(r"$$F(\epsilon) = " + latex(F_epsilon) + r"$$")
            st.write("**Explicación**: Reemplazamos x = ε para evitar la singularidad en 0.")
            st.write("**Subpaso 3.3: Aplicar el Teorema Fundamental - Restar para formar la expresión**")
            st.write("La integral de ε a b es F(b) - F(ε).")
            expr = F_b - F_epsilon
            st.latex(r"$$\int_\epsilon^b f(x) \, dx = F(b) - F(\epsilon) = " + latex(expr) + r"$$")
            res = limit(expr, epsilon, 0, dir='+')
        else:  # proper
            st.write("Evaluamos: $F(b) - F(a)$ (integral propia, sin límites variables)")
            # Subpasos detallados para proper
            F_b = F.subs(x, b)
            F_a = F.subs(x, a)
            st.write("**Subpaso 3.1: Sustituir en la antiderivada para el límite superior (b)**")
            st.latex(r"$$F(b) = " + latex(F_b) + r"$$")
            st.write("**Explicación**: Reemplazamos x = b en F(x).")
            st.write("**Subpaso 3.2: Sustituir en la antiderivada para el límite inferior (a)**")
            st.latex(r"$$F(a) = " + latex(F_a) + r"$$")
            st.write("**Explicación**: Reemplazamos x = a en F(x).")
            st.write("**Subpaso 3.3: Aplicar el Teorema Fundamental - Restar para el valor exacto**")
            st.write("No hay singularidad ni infinito, así que la resta da el resultado directo.")
            expr = F_b - F_a
            st.latex(r"$$\int_a^b f(x) \, dx = F(b) - F(a) = " + latex(expr) + r"$$")
            res = sp.simplify(expr)

        # *** FIX: Paso 4 Mejorado con Explicación Detallada del Cálculo + LaTeX Display (Renderiza Bonito, No Raw) ***
        st.write("**Paso 4: Calcular el Límite**")
        if mode == "infinite_upper":
            st.write("Tomamos el límite de la expresión cuando $t \\to \\infty$.")
            st.latex(r"$$\lim_{t \to \infty} \left[ " + latex(expr) + r" \right]$$")
            st.write("**Explicación detallada del cálculo**: Analizamos término por término. Los constantes quedan iguales, y los términos que crecen con t (o dependen de 1/t) tienden a 0 si la función decae rápido (ej. para 1/x², 1/t → 0 cuando t → ∞, así que queda el valor constante).")
        elif mode == "singular_lower":
            st.write("Tomamos el límite de la expresión cuando $\\epsilon \\to 0^+$.")
            st.latex(r"$$\lim_{\epsilon \to 0^+} \left[ " + latex(expr) + r" \right]$$")
            st.write("**Explicación detallada del cálculo**: Verificamos el comportamiento cerca de ε=0. Si hay término como 1/√ε, diverge a ∞; si converge, el límite es finito.")
        else:  # proper
            st.write("No se necesita límite (integral propia directa). El valor es la expresión simplificada.")
            st.latex(r"$$" + latex(expr) + r"$$")
        # Resultado final en display LaTeX (centrado, math real - no raw $\text{1}$)
        st.latex(r"$$\text{Resultado del Límite: } " + latex(res) + r"$$")

        # Paso 5: Análisis de convergencia con detalle
        st.write("**Paso 5: Análisis de Convergencia**")
        if res.is_finite:
            st.success(
                f"✅ **La integral CONVERGE** a un valor finito: ${latex(res)}$."
            )
            st.write(
                "**Explicación detallada**: El límite existe y es finito, por lo que el área bajo la curva es acotada (limitada). Esto implica que la función decae lo suficientemente rápido (ej. como $1/x^2$ o mejor)."
            )
            # *** FIX: Success profesional + confetti leve (20 copos, speed media – sutil, dura 2-3 seg) ***
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

    # ***
