import streamlit as st
import sympy as sp
from sympy import limit, oo, Symbol, integrate, latex, lambdify
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

    /* 2. BARRA LATERAL - FIX CRÍTICO: Forzar el color a azul oscuro (#1e3a8a) 
       para asegurar visibilidad y consistencia con el cuerpo principal. */
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
        /* Forzamos color azul oscuro, igual que el cuerpo principal */
        color: #1e3a8a !important; 
    }
    /* Aseguramos también el contraste de las alertas en el sidebar */
    .sidebar .sidebar-content .stAlert p, 
    .sidebar .sidebar-content .stAlert h3, 
    .sidebar .sidebar-content .stAlert *
    {
        color: #1e3a8a !important; 
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
        f = sp.sympify(f_str)
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
                "**Explicación detallada**: Evaluaremos F(t)-F(a) y tomaremos el límite $t \\to \\infty$."
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

        st.write("**Paso 2: Encontrar la Antiderivada Indefinida F(x**
