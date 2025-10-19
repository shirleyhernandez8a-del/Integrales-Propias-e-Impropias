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
    page_icon=" 🧮 ",
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
    h1 {color: #1f3a5f; text-align: center; font-weight: 800;}
    h2 {color: #2c5890; font-weight: 600;}
    h3 {color: #3b71b8; border-bottom: 2px solid #3b71b8;}
    
    .stButton>button {
        background-color: #6a8dff;
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #4a6bff;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stExpander {
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #c0d9ff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        background-color: #dbe4ff;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b71b8 !important;
        color: white !important;
        border-bottom: 3px solid #1f3a5f !important;
    }
    </style>
""", unsafe_allow_html=True)


# --- Variables Globales ---
x = sp.Symbol('x')
R = sp.Reals # Conjunto de números reales (para referencia conceptual)

def parse_limit(s):
    """Convierte un string de límite a un objeto SymPy, manejando 'oo' y '-oo'."""
    if s.strip().lower() == 'oo':
        return sp.oo
    elif s.strip() == '-oo':
        return -sp.oo
    try:
        # Intenta convertir a número racional o entero
        return sp.S(s)
    except:
        raise ValueError(f"Límite no reconocido: {s}")

def analizar_y_resolver_impropia(f, a_sym, b_sym, a_str, b_str):
    """
    Función robusta que diagnostica el tipo de integral impropia,
    calcula la antiderivada, aplica la definición de límite
    y determina la convergencia, incluyendo la verificación de dominio en R.
    """
    st.subheader("🛠️ Análisis de Integral Impropiapaso a Paso")

    # 1. Encontrar la Antiderivada Indefinida F(x)
    try:
        F_x = sp.integrate(f, x)
        st.write("---")
        st.markdown(f"**Paso 1: Antiderivada Indefinida**")
        st.latex(f"F(x) = \\int {latex(f)} \\, dx = {latex(F_x)}")
    except Exception as e:
        st.error(f"Error al calcular la antiderivada: {e}")
        return

    # 2. Diagnóstico de Integral (Tipo I, Tipo II, o Mixta)
    st.write("---")
    st.markdown("**Paso 2: Diagnóstico y División**")
    
    is_type_i = a_sym == -sp.oo or b_sym == sp.oo
    singularities = []

    # Heurística para encontrar singularidades (raíces del denominador o puntos de discontinuidad)
    try:
        if f.is_rational_function():
            # Para funciones racionales, buscar raíces del denominador
            den = sp.denom(f)
            if den != 1:
                sols = sp.solveset(den, x, domain=sp.Reals)
                if sols != sp.EmptySet:
                    for sol in sols:
                        if sol.is_number and a_sym.is_number and b_sym.is_number:
                            # Convertir límites a flotantes para la comparación, si no son oo
                            a_val = float(a_sym)
                            b_val = float(b_sym)
                            sol_val = float(sol)
                            
                            if (a_val < sol_val < b_val) or sp.N(sol) == a_sym or sp.N(sol) == b_sym:
                                singularities.append(sol)
                        elif sol == a_sym or sol == b_sym:
                            singularities.append(sol)
        
        # También se puede buscar una singularidad de punto final de forma más simple:
        if a_sym.is_number and f.limit(x, a_sym, dir='+') in [sp.oo, -sp.oo]:
             if a_sym not in singularities: singularities.append(a_sym)
        if b_sym.is_number and f.limit(x, b_sym, dir='-') in [sp.oo, -sp.oo]:
             if b_sym not in singularities: singularities.append(b_sym)


        singularities = sorted(list(set(singularities)), key=lambda s: sp.N(s))

    except Exception:
        # En caso de error de cálculo de soluciones, confiar solo en el límite infinito
        st.warning("No se pudo analizar completamente las singularidades internas. Se confiará en la función 'integrate' de Sympy.")

    
    # 3. Definir Límites de Integración
    
    # Caso 1: Singularidad Interna (requiere división)
    if any(a_sym < s < b_sym for s in singularities):
        c = singularities[0] # Usar la primera singularidad interna
        st.markdown(f"**Tipo II (Singularidad Interna):** Discontinuidad en $c={latex(c)}$. La integral se divide en dos:")
        st.latex(f"\\int_{{{latex(a_sym)}}}^{{{latex(b_sym)}}} f(x) \\, dx = \\lim_{{t_1 \\to {latex(c)}^-}} \\int_{{{latex(a_sym)}}}^{{t_1}} f(x) \\, dx + \\lim_{{t_2 \\to {latex(c)}^+}} \\int_{{t_2}}^{{{latex(b_sym)}}} f(x) \\, dx")
        parts = [
            (a_sym, c, c, '-'),
            (c, b_sym, c, '+')
        ]
    # Caso 2: Límites Infinitos (Tipo I)
    elif is_type_i:
        st.markdown("**Tipo I (Límites Infinitos):** Aplicación de la definición de límite.")
        if a_sym == -sp.oo and b_sym == sp.oo:
            # Mixta, dividir en un punto c arbitrario (usar c=0)
            c = 0
            st.latex(f"\\int_{{- \\infty}}^{\\infty} f(x) \\, dx = \\lim_{{t_1 \\to - \\infty}} \\int_{{t_1}}^{{0}} f(x) \\, dx + \\lim_{{t_2 \\to \\infty}} \\int_{{0}}^{{t_2}} f(x) \\, dx")
            parts = [
                (-sp.oo, c, -sp.oo, 'oo'), # Usamos 'oo' para indicar que el límite tiende al infinito
                (c, sp.oo, sp.oo, 'oo')
            ]
        elif a_sym == -sp.oo:
            st.latex(f"\\int_{{- \\infty}}^{{{latex(b_sym)}}} f(x) \\, dx = \\lim_{{t \\to - \\infty}} \\int_{{t}}^{{{latex(b_sym)}}} f(x) \\, dx")
            parts = [(a_sym, b_sym, -sp.oo, 'oo')]
        elif b_sym == sp.oo:
            st.latex(f"\\int_{{{latex(a_sym)}}}^{\\infty} f(x) \\, dx = \\lim_{{t \\to \\infty}} \\int_{{{latex(a_sym)}}}^{{t}} f(x) \\, dx")
            parts = [(a_sym, b_sym, sp.oo, 'oo')]
    # Caso 3: Singularidad en un Extremo (Tipo II)
    elif a_sym in singularities or b_sym in singularities:
        s = a_sym if a_sym in singularities else b_sym
        dir_char = '+' if s == a_sym else '-'
        st.markdown(f"**Tipo II (Singularidad en Extremo):** Discontinuidad en $x={latex(s)}$.")
        if s == a_sym:
            st.latex(f"\\int_{{{latex(a_sym)}}}^{{{latex(b_sym)}}} f(x) \\, dx = \\lim_{{t \\to {latex(a_sym)}{dir_char}}} \\int_{{t}}^{{{latex(b_sym)}}} f(x) \\, dx")
            parts = [(a_sym, b_sym, a_sym, '+')]
        else: # s == b_sym
            st.latex(f"\\int_{{{latex(a_sym)}}}^{{{latex(b_sym)}}} f(x) \\, dx = \\lim_{{t \\to {latex(b_sym)}{dir_char}}} \\int_{{{latex(a_sym)}}}^{{t}} f(x) \\, dx")
            parts = [(a_sym, b_sym, b_sym, '-')]
    else:
        # Fallback si Sympy la marcó como impropia pero la heurística falló
        st.warning("Advertencia: El diagnóstico de singularidad no fue concluyente, se usará la integración directa de Sympy.")
        parts = [(a_sym, b_sym, None, None)]


    # 4. Cálculo de los Límites
    st.write("---")
    st.markdown("**Paso 3: Cálculo de la Convergencia**")
    
    final_result = 0
    all_converge = True
    
    for i, (lower, upper, s_point, s_dir) in enumerate(parts):
        # f(t) - f(lim_fijo) o f(lim_fijo) - f(t)
        
        # Definir la integral definida F(b) - F(a)
        
        # Si s_point es infinito, el límite es en t (la variable)
        is_inf_limit = s_point == sp.oo or s_point == -sp.oo

        if not is_inf_limit and s_point is not None:
            # Es singularidad de Tipo II
            if s_dir == '-': # Límite superior se acerca a s_point por la izquierda
                t_limit = s_point
                limit_dir = '-'
                limite_expr = F_x.subs(x, sp.Symbol('t')) - F_x.subs(x, lower)
                st.markdown(f"**Parte {i+1}: $\\lim_{{t \\to {latex(s_point)}^-}} [F(t) - F({latex(lower)})]$**")
            elif s_dir == '+': # Límite inferior se acerca a s_point por la derecha
                t_limit = s_point
                limit_dir = '+'
                limite_expr = F_x.subs(x, upper) - F_x.subs(x, sp.Symbol('t'))
                st.markdown(f"**Parte {i+1}: $\\lim_{{t \\to {latex(s_point)}^+}} [F({latex(upper)}) - F(t)]**")
            else:
                 # Singularidad en un extremo que no es ni '+' ni '-' (debería ser cubierta antes)
                 st.error(f"Error interno al definir la dirección del límite para la singularidad en {s_point}.")
                 continue
                 
            try:
                result_part = limit(limite_expr, sp.Symbol('t'), t_limit, dir=limit_dir)
                st.latex(f"\\lim = {latex(limite_expr)} = {latex(result_part)}")
            except Exception as e:
                st.error(f"Error al calcular el límite de la Parte {i+1}: {e}")
                result_part = sp.nan

        elif is_inf_limit:
            # Es límite infinito de Tipo I
            if s_point == sp.oo: # Límite superior es infinito
                t_limit = sp.oo
                limite_expr = F_x.subs(x, sp.Symbol('t')) - F_x.subs(x, lower)
                st.markdown(f"**Parte {i+1}: $\\lim_{{t \\to \\infty}} [F(t) - F({latex(lower)})]$**")
            else: # Límite inferior es infinito
                t_limit = -sp.oo
                limite_expr = F_x.subs(x, upper) - F_x.subs(x, sp.Symbol('t'))
                st.markdown(f"**Parte {i+1}: $\\lim_{{t \\to - \\infty}} [F({latex(upper)}) - F(t)]**")

            try:
                result_part = limit(limite_expr, sp.Symbol('t'), t_limit)
                st.latex(f"\\lim = {latex(limite_expr)} = {latex(result_part)}")
            except Exception as e:
                st.error(f"Error al calcular el límite de la Parte {i+1}: {e}")
                result_part = sp.nan

        else:
            # Si no hubo partes definidas, simplemente usamos la integral directa (fallo en heurística)
            try:
                result_part = sp.integrate(f, (x, lower, upper))
            except Exception as e:
                st.error(f"Error al calcular la integral directa: {e}")
                result_part = sp.nan


        # Conclusión de la parte
        if result_part.is_number and result_part != sp.oo and result_part != -sp.oo and result_part != sp.nan:
            st.success(f"La Parte {i+1} CONVERGE a ${latex(result_part)}$")
            final_result += result_part
        else:
            st.error(f"La Parte {i+1} DIVERGE a ${latex(result_part)}$ o el resultado es indefinido.")
            all_converge = False
            break # Si una parte diverge, la integral total diverge

    # 5. Conclusión Final
    st.write("---")
    st.markdown("**Paso 4: Conclusión Final**")

    if all_converge:
        st.success(f"✅ La integral completa CONVERGE a la suma de sus partes.")
        st.latex(f"\\int_{{{latex(a_sym)}}}^{{{latex(b_sym)}}} {latex(f)} \\, dx = {latex(final_result)}")
    else:
        st.error("🚫 La integral completa DIVERGE porque al menos una de sus partes diverge.")
        st.latex(f"\\int_{{{latex(a_sym)}}}^{{{latex(b_sym)}}} {latex(f)} \\, dx \\text{ DIVERGE}")


def resolver_integral(f_str, a_str, b_str):
    """
    Función principal para parsear, diagnosticar y resolver integrales
    propias e impropias, manejando errores de dominio.
    """
    if not f_str or not a_str or not b_str:
        st.error("Por favor, introduce la función y los límites de integración.")
        return

    try:
        f = sp.sympify(f_str, evaluate=False) # No evaluar inmediatamente
        a_sym = parse_limit(a_str)
        b_sym = parse_limit(b_str)
    except Exception as e:
        st.error(f"Error al parsear la expresión o los límites: {e}")
        return
    
    st.markdown(f"## 📝 Resolución Detallada de: $\\int_{{{latex(a_sym)}}}^{{{latex(b_sym)}}} {latex(f)} \\, dx$")
    st.write("---")

    # --- 1. Calcular el Resultado Global (SymPy) para diagnóstico ---
    try:
        resultado_simbolico = sp.integrate(f, (x, a_sym, b_sym))
    except NotImplementedError:
        st.error("Sympy no puede resolver esta integral de forma simbólica.")
        return
    except Exception as e:
        # En caso de otros errores de integración (p. ej., función mal definida)
        st.error(f"Ocurrió un error al intentar integrar: {e}")
        return

    # --- 2. VERIFICACIÓN CRÍTICA: Dominio Real y Convergencia ---
    if resultado_simbolico.is_complex and not resultado_simbolico.is_real:
        st.error("🚫 ¡Advertencia Crítica de Dominio! 🚫")
        st.markdown("La función $\\frac{1}{x^{1/2}}$ (o similar) no está definida en el dominio real para una porción del intervalo de integración (e.g., raíces pares de números negativos).")
        st.markdown("En el contexto del cálculo real (cálculo de áreas), esta integral **DIVERGE** o **NO ESTÁ DEFINIDA**.")
        st.write("El resultado simbólico que involucra números complejos es:")
        st.latex(f"\\text{{Resultado Simbólico: }} {latex(resultado_simbolico)}")
        return
    
    # --- 3. Diagnóstico de Tipo (Propia vs. Impropia) ---
    
    is_improper = False
    
    # Criterio 1: Límites infinitos (Tipo I)
    if a_sym == -sp.oo or b_sym == sp.oo:
        is_improper = True
        
    # Criterio 2: Resultado de Sympy indica divergencia (Heurística)
    elif resultado_simbolico in [sp.oo, -sp.oo, sp.nan]:
        is_improper = True
        
    # Criterio 3: Singularidad en los extremos (Tipo II)
    elif a_sym.is_number and b_sym.is_number:
        try:
            lim_a = f.limit(x, a_sym, dir='+-')
            lim_b = f.limit(x, b_sym, dir='-+')
            if lim_a in [sp.oo, -sp.oo] or lim_b in [sp.oo, -sp.oo]:
                 is_improper = True
        except:
             # Si el límite falla, no podemos asegurar que sea impropia por este criterio
             pass

    
    # --- 4. Ejecución del Proceso ---
    
    if is_improper:
        analizar_y_resolver_impropia(f, a_sym, b_sym, a_str, b_str)
    else:
        st.subheader("✅ Integral Propia (Definida) Detectada")
        
        # Procedimiento de Integral Propia (Teorema Fundamental del Cálculo)
        try:
            F_x = sp.integrate(f, x)
            st.markdown(f"**Paso 1: Antiderivada**")
            st.latex(f"F(x) = \\int {latex(f)} \\, dx = {latex(F_x)}")

            st.markdown(f"**Paso 2: Aplicar Teorema Fundamental del Cálculo (TFC)**")
            st.latex(f"\\int_{{{latex(a_sym)}}}^{{{latex(b_sym)}}} f(x) \\, dx = F({latex(b_sym)}) - F({latex(a_sym)})")

            F_b = F_x.subs(x, b_sym)
            F_a = F_x.subs(x, a_sym)
            
            st.latex(f"F({latex(b_sym)}) - F({latex(a_sym)}) = ({latex(F_b)}) - ({latex(F_a)}) = {latex(resultado_simbolico)}")

            st.success(f"Resultado Final: La integral CONVERGE a ${latex(resultado_simbolico)}$")

        except Exception as e:
            st.error(f"Error en el procedimiento de la integral propia: {e}")
            st.info(f"Resultado obtenido directamente de SymPy: ${latex(resultado_simbolico)}$")


def dibujar_grafica(f_str, a_str, b_str):
    """Dibuja la gráfica de la función y sombrea el área de la integral."""
    try:
        f = sp.sympify(f_str)
        a = parse_limit(a_str)
        b = parse_limit(b_str)
        
        # Convertir a función lambda para numpy
        f_num = lambdify(x, f, "numpy")
        
        # Definir el rango de x
        if a == -sp.oo:
            min_x = -10
        elif a.is_number:
            min_x = float(a) - 1
        
        if b == sp.oo:
            max_x = 10
        elif b.is_number:
            max_x = float(b) + 1
        
        # Ajuste para singularidades: Evitar calcular f_num en el punto exacto de singularidad/límite
        
        # Crear rango de x
        x_range = np.linspace(min_x, max_x, 500)
        y_range = f_num(x_range)
        
        # Manejar límites infinitos para el sombreado
        if a == -sp.oo: a_shade = min(x_range)
        elif a.is_number: a_shade = float(a)
        
        if b == sp.oo: b_shade = max(x_range)
        elif b.is_number: b_shade = float(b)
            
        # Para el sombreado de la singularidad, ajustamos el límite inferior/superior si es un punto singular
        # Simplemente creamos un nuevo rango para el área sombreada
        
        # Si a o b son números finitos, el rango de sombreado es [a, b]
        x_shade = np.linspace(a_shade, b_shade, 500)
        y_shade = f_num(x_shade)

        # Reemplazar valores no finitos (NaN, Inf) en y_shade con NaN para evitar errores de plot
        y_shade[~np.isfinite(y_shade)] = np.nan
        
        # Reemplazar valores no finitos (NaN, Inf) en y_range con NaN para evitar errores de plot
        y_range[~np.isfinite(y_range)] = np.nan
        
        # Manejar el rango Y para que no sea ridículamente grande cerca de singularidades
        y_min = np.nanmin(y_range[np.isfinite(y_range)]) if np.any(np.isfinite(y_range)) else -5
        y_max = np.nanmax(y_range[np.isfinite(y_range)]) if np.any(np.isfinite(y_range)) else 5
        
        # Limitar el rango de Y para la visualización
        Y_LIMIT = 10
        y_plot_min = max(y_min, -Y_LIMIT)
        y_plot_max = min(y_max, Y_LIMIT)
        
        if y_plot_max - y_plot_min < 0.5:
             y_plot_max = y_plot_min + 1

        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Trazar la función principal
        ax.plot(x_range, y_range, label=f"f(x) = ${latex(f)}$", color='#4a6bff', linewidth=2)
        
        # Sombreado del área
        ax.fill_between(x_shade, y_shade, where=np.isfinite(y_shade), color='#90b1ff', alpha=0.5, label="Área Aproximada")
        
        # Límites de integración
        if a.is_number:
            ax.axvline(x=float(a), color='red', linestyle='--', label=f"Límite Inferior: {a_str}")
        if b.is_number:
            ax.axvline(x=float(b), color='green', linestyle='--', label=f"Límite Superior: {b_str}")
            
        # Ejes y Leyendas
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_ylim(y_plot_min, y_plot_max)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='best')
        ax.set_title(f"Gráfica de la Integral $\\int_{{{a_str}}}^{{{b_str}}} {f_str} \\, dx$", fontsize=14)
        
        st.pyplot(fig)
        
    except Exception as e:
        # st.error(f"No se pudo generar la gráfica debido a: {e}")
        st.warning("No se pudo generar la gráfica, posiblemente por singularidades o valores no reales. Intente simplificar la función.")


# --- TÍTULO PRINCIPAL ---
st.title(" 🚀 Solver Detallado de Integrales Definitas (Propia/Impropia)")
st.caption("Herramienta avanzada de cálculo que diagnostica integrales impropias (Tipo I y II) y proporciona el procedimiento de límites correcto.")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("Configuración y Datos")
    f_input = st.text_input("Función f(x) (ej: 1/x**2, exp(-x), 1/sqrt(x))", value=st.session_state.saved_f, key="f_input_main")
    a_input = st.text_input("Límite Inferior 'a' (ej: 0, -oo, -1)", value=st.session_state.saved_a, key="a_input_main")
    b_input = st.text_input("Límite Superior 'b' (ej: 1, oo, 5)", value=st.session_state.saved_b, key="b_input_main")
    
    modo = st.radio("Modo de Visualización de Gráfica", 
                   ["Simple (Cálculo sin Gráfica)", "Avanzado (con Gráfica Auto)"],
                   index=1,
                   key="modo_visual")

    if st.button("Resolver Integral", key="solve_button"):
        st.session_state.saved_f = f_input 
        st.session_state.saved_a = a_input
        st.session_state.saved_b = b_input
        
        if modo == "Avanzado (con Gráfica Auto)":
             st.session_state.show_graph = True
        else:
             st.session_state.show_graph = False
        
        resolver_integral(f_input, a_input, b_input)
    
    # Botón para mostrar la gráfica por separado
    if modo == "Simple (Cálculo sin Gráfica)" and st.session_state.saved_f:
        if st.button("Mostrar Gráfica del Último Cálculo", key="show_graph_manual"):
            st.session_state.show_graph = True
            
# --- CONTENIDO PRINCIPAL ---
if st.session_state.saved_f and modo == "Avanzado (con Gráfica Auto)":
    # La función resolver_integral ya fue llamada al presionar 'Resolver Integral'
    pass

# Si el modo es avanzado, o si se pidió mostrar la gráfica manualmente
if st.session_state.show_graph and st.session_state.saved_f:
    st.header("Gráfica Interactiva: Visualiza el Área de la Integral")
    dibujar_grafica(st.session_state.saved_f, st.session_state.saved_a, st.session_state.saved_b)
    st.write("La gráfica muestra la función y el área de integración (aproximada, especialmente en singularidades).")
    st.write("---")
    
if st.session_state.saved_f and st.session_state.show_graph == False:
    # Asegurarse de que el cálculo se ejecute en modo simple si no se hace en el botón
    resolver_integral(st.session_state.saved_f, st.session_state.saved_a, st.session_state.saved_b)

# --- EJEMPLOS RÁPIDOS ---
st.header("Ejemplos Rápidos")
col_ej1, col_ej2, col_ej3 = st.columns(3)
col_ej4, col_ej5, col_ej6 = st.columns(3)

with col_ej1:
    with st.expander("Ej1: $\\int 1/x^2 dx$ de 1 a $\\infty$ (Tipo I, CONVERGE)"):
        st.write("**Función**: $1/x^2$ | **Límites**: $a=1, b=\\text{oo}$")
        if st.button("Resolver Ejemplo 1", key="ej1"):
            st.session_state.saved_f = "1/x**2" 
            st.session_state.saved_a = "1"
            st.session_state.saved_b = "oo"
            resolver_integral("1/x**2", "1", "oo")
            if modo == "Avanzado (con Gráfica Auto)": st.session_state.show_graph = True
            
with col_ej2:
    with st.expander("Ej2: $\\int e^{-x} dx$ de 0 a $\\infty$ (Tipo I, CONVERGE)"):
        st.write("**Función**: $exp(-x)$ | **Límites**: $a=0, b=\\text{oo}$")
        if st.button("Resolver Ejemplo 2", key="ej2"):
            st.session_state.saved_f = "exp(-x)" 
            st.session_state.saved_a = "0"
            st.session_state.saved_b = "oo"
            resolver_integral("exp(-x)", "0", "oo")
            if modo == "Avanzado (con Gráfica Auto)": st.session_state.show_graph = True
            
with col_ej3:
    with st.expander("Ej3: $\\int 1/x^3 dx$ de -1 a 1 (Singularidad Interna, DIVERGE)"):
        st.write("**Función**: $1/x**3$ | **Límites**: $a=-1, b=1$")
        if st.button("Resolver Ejemplo 3", key="ej3"):
            st.session_state.saved_f = "1/x**3" 
            st.session_state.saved_a = "-1"
            st.session_state.saved_b = "1"
            resolver_integral("1/x**3", "-1", "1")
            if modo == "Avanzado (con Gráfica Auto)": st.session_state.show_graph = True

with col_ej4:
    with st.expander("Ej4: $\\int \\ln(x) dx$ de 0 a 1 (Singularidad en Extremo, CONVERGE)"):
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
    with st.expander("Ej6: $\\int x^2 dx$ de 0 a 2 (Integral Propia, CONVERGE)"):
        st.write("**Función**: $x^2$ | **Límites**: $a=0, b=2$")
        if st.button("Resolver Ejemplo 6", key="ej6"):
            st.session_state.saved_f = "x**2" 
            st.session_state.saved_a = "0"
            st.session_state.saved_b = "2"
            resolver_integral("x**2", "0", "2")
            if modo == "Avanzado (con Gráfica Auto)": st.session_state.show_graph = True
        
st.write("---")
st.markdown("##### 🔑 NOTAS DE LA FUNCIÓN:")
st.markdown("* **Raíz Cuadrada:** Usa `sqrt(x)` o `x**(1/2)`. El código verifica si la función es real en el intervalo.")
st.markdown("* **Exponencial:** Usa `exp(x)`.")
st.markdown("* **Infinito:** Usa `oo` (dos letras 'o' minúsculas) para $\\infty$.")
st.markdown("* **Logaritmo Natural:** Usa `log(x)`.")
