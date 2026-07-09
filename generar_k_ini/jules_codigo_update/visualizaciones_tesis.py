# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:43:04 2026

@author: matta
"""

# -*- coding: utf-8 -*-
"""
MÓDULO DE VISUALIZACIÓN CIENTÍFICA Y DIAGNÓSTICO DE CONVERGENCIA
Desarrollado para el análisis de Generación de Columnas Acelerado (Dantzig-Wolfe)
en problemas de clasificación lineal a gran escala (Soft-Margin SVM).

Autores: Matías Muñoz Flores & Colaborador
Año: 2026
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:43:04 2026

@author: matta
"""

# -*- coding: utf-8 -*-
"""
MÓDULO DE VISUALIZACIÓN CIENTÍFICA Y DIAGNÓSTICO DE CONVERGENCIA
Desarrollado para el análisis de Generación de Columnas Acelerado (Dantzig-Wolfe)
en problemas de clasificación lineal a gran escala (Soft-Margin SVM).

Autores: Matías Muñoz Flores & Colaborador
Año: 2026
"""

import os
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_convergence_paper(history, conic_opt=None, title="Convergencia Primal-Dual", filename=None, log_scale=False):
    """
    Grafica la evolución conjunta del Upper Bound (Master) y Lower Bound (Pricing).
    Diseñado bajo estándares de publicación (Estilo Paper).

    Parámetros:
    -----------
    history : dict
        Diccionario con las claves 'iter', 'master_value' y 'pricing_value'.
    conic_opt : float, opcional
        Valor óptimo del problema cónico completo (Mosek) como línea de referencia.
    title : str
        Título principal del gráfico.
    filename : str, opcional
        Ruta o nombre del archivo para guardar la imagen en alta resolución (300 DPI).
    log_scale : bool
        Si es True, aplica escala 'symlog' para manejar valores duales muy negativos.
    """
    iterations = history['iter']
    ub = history['master_value']
    lb = history['pricing_value']

    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Trazado de las curvas primarias de convergencia
    ax.plot(iterations, ub, color='red', linewidth=2, label='Upper Bound (Master)')
    ax.plot(iterations, lb, color='blue', linewidth=2, label='Dual Bound (Pricing)')

    # 2. Inyección de la línea de referencia del óptimo teórico global
    if conic_opt is not None:
        ax.axhline(y=conic_opt, color='green', linestyle='--', alpha=0.7,
                   label=f'Óptimo Cónico ({conic_opt:.4f})')

    # 3. Configuración de etiquetas y formato de texto
    ax.set_xlabel('Iteraciones (Naturales)', fontsize=12)
    ax.set_ylabel('Valor Objetivo / Energía', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)

    # 4. Control del encuadre y rejillas según la escala seleccionada
    if log_scale:
        # 'symlog' evita errores numéricos con valores negativos o cero provenientes del dual
        ax.set_yscale('symlog', linthresh=1e-5)
        ax.grid(True, which="both", linestyle=':', alpha=0.4)
        print("-> Nota: Usando escala 'symlog' (Symmetric Log) para soportar valores negativos si existen.")
    else:
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()

    # 5. Guardado físico del artefacto visual
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()


def plot_convergence_metrics(history, conic_opt, dataset_size, use_filter_iters=None):
    """
    Gráfico de panel doble para evaluar el comportamiento absoluto y la velocidad
    de convergencia mediante el Gap logarítmico.

    Parámetros:
    -----------
    history : dict
        Diccionario con datos de iteraciones y valores de función objetivo.
    conic_opt : float
        Valor óptimo teórico utilizado para calcular la distancia al óptimo (Gap).
    dataset_size : str
        Descripción de las dimensiones del dataset para el título del panel.
    use_filter_iters : list, opcional
        Lista de índices de iteraciones donde el filtro de columnas estuvo activo.
    """
    iterations = np.array(history['iter'])
    # Cálculo de la distancia absoluta entre el Master restringido y el óptimo real
    gaps = np.abs(np.array(history['master_value']) - conic_opt)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Panel Izquierdo: Convergencia en escala lineal con marcas de filtros
    ax[0].plot(iterations, history['master_value'], label='Master Obj', color='red', linewidth=1.5)
    ax[0].axhline(y=conic_opt, color='k', linestyle='--', label='Óptimo')

    # Mapeo visual de los eventos de filtrado/purga de columnas
    if use_filter_iters:
        ax[0].scatter(use_filter_iters, [history['master_value'][i] for i in use_filter_iters],
                      color='magenta', marker='x', s=100, label='Filtro Activado', zorder=5)
    ax[0].set_title(f"Convergencia Normal - Dataset {dataset_size}")
    ax[0].set_xlabel('Iteraciones')
    ax[0].set_ylabel('Valor Master')
    ax[0].legend()
    ax[0].grid(True, linestyle=':', alpha=0.5)

    # Panel Derecho: Trayectoria del Gap en escala logarítmica (mide tasa de convergencia)
    ax[1].plot(iterations, gaps, color='purple', linewidth=1.5)
    ax[1].set_yscale('log')
    ax[1].set_xlabel('Iteraciones')
    ax[1].set_ylabel('Gap vs Óptimo (Log)')
    ax[1].set_title("Velocidad de Convergencia (Log Scale)")
    ax[1].grid(True, which="both", ls="-", alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_convergence_split(history, conic_opt=None, title="Convergencia Primal-Dual (Separada)", filename=None, use_log_scale=False):
    """
    Genera dos subplots verticales independientes compartiendo el eje X.
    Fuerza notación científica estricta en el eje Y para detectar variaciones marginales.

    Parámetros:
    -----------
    history : dict
        Diccionario indexado con los valores del Master y del Pricing.
    conic_opt : float, opcional
        Óptimo de referencia.
    title : str
        Título de la composición.
    filename : str, opcional
        Nombre del archivo de salida.
    use_log_scale : bool
        Aplica symlog al panel inferior si el pricing es altamente inestable.
    """
    iterations = history['iter']
    ub = history['master_value']
    lb = history['pricing_value']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # --- SUBPLOT 1: COMPORTAMIENTO DEL MASTER (PRIMAL / UPPER BOUND) ---
    ax1.plot(iterations, ub, color='#D32F2F', linewidth=2, label='Master Obj (Upper Bound)')

    if conic_opt is not None:
        ax1.axhline(y=conic_opt, color='green', linestyle='--', alpha=0.8, linewidth=1.5)
        # Etiqueta flotante alineada al último punto del eje X
        ax1.text(
            x=iterations[-1],
            y=conic_opt,
            s=f' {conic_opt:.5f} ',
            color='green', fontsize=9, ha='right', va='top',
            bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.7)
        )

    ax1.set_ylabel('Valor Objetivo', fontsize=11)
    ax1.set_title(f"{title} - Vista Detallada", fontsize=13)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Configuración estricta de notación científica para el Master
    formatter1 = ticker.ScalarFormatter(useMathText=True)
    formatter1.set_powerlimits((0, 0)) # (0,0) fuerza el exponente x10^k obligatoriamente
    ax1.yaxis.set_major_formatter(formatter1)

    # Ventana de zoom adaptativo si el cambio es infinitesimal
    if np.max(ub) - np.min(ub) < 1e-4 and conic_opt:
        ax1.set_ylim(conic_opt - 2e-3, conic_opt + 1e-3)

    # --- SUBPLOT 2: COMPORTAMIENTO DEL PRICING (DUAL / LOWER BOUND) ---
    ax2.plot(iterations, lb, color='#1976D2', linewidth=2, label='Dual Bound (Pricing)')

    if conic_opt is not None:
        ax2.axhline(y=conic_opt, color='green', linestyle='--', alpha=0.6, linewidth=1)
        ax2.text(
            x=iterations[-1], y=conic_opt, s=f' {conic_opt:.5e} ',
            color='green', fontsize=9, ha='right', va='top',
            bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.7)
        )

    ax2.set_ylabel('Energía Dual', fontsize=11)
    ax2.set_xlabel('Iteraciones', fontsize=12)
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Control del formateo del eje Y para el Pricing
    if use_log_scale:
        ax2.set_yscale('symlog', linthresh=1.0)
        print("-> Escala Logarítmica activada para el eje Dual.")
    else:
        ax2.set_yscale('linear')
        formatter2 = ticker.ScalarFormatter(useMathText=True)
        formatter2.set_powerlimits((0, 0))
        ax2.yaxis.set_major_formatter(formatter2)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()


def plot_heatmap_memoria_robusto(historial_dict, tipo="Theta", log_scale=False):
    """
    Genera un mapa de calor que muestra la evolución de los pesos de las columnas.
    Resuelve el error de strings al procesar diccionarios dispersos alineando
    correctamente los identificadores numéricos.

    Parámetros:
    -----------
    historial_dict : dict of lists
        Diccionario donde cada llave es un identificador de columna (ej: 'p_0')
        y el valor es una lista de pesos a lo largo de las iteraciones.
    tipo : str
        Etiqueta para identificar el tipo de columna analizada ('Theta' o 'Mu').
    log_scale : bool
        Si es True, aplica escala logarítmica (log10) a los valores (útil para Mu).
    """
    if not historial_dict or not isinstance(historial_dict, dict):
        print(f"[FLAG - PLOT] Historial de {tipo} vacío o con formato incorrecto.")
        return

    # 1. Extracción y ordenamiento natural de todas las llaves que existieron en el ciclo
    # Convierte 'p_10' en entero (10) para evitar que el ordenamiento alfabético ponga 'p_10' antes de 'p_2'
    try:
        todas_columnas = sorted(list(historial_dict.keys()), key=lambda x: int(x.split('_')[1]))
    except Exception as e:
        print(f"[FLAG - PLOT ERROR] Error al parsear índices de las columnas: {e}")
        return

    # 2. Inicialización de la estructura matricial densa (Iteraciones x Columnas Únicas)
    if len(todas_columnas) == 0:
        print(f"[FLAG - PLOT] No hay columnas para graficar en {tipo}.")
        return

    n_iters = max(len(historial_dict[k]) for k in todas_columnas)
    n_cols = len(todas_columnas)

    matriz = np.full((n_iters, n_cols), np.nan)

    # 3. Transferencia indexada del diccionario a la matriz numérica
    for j, col in enumerate(todas_columnas):
        vals = historial_dict[col]
        matriz[:len(vals), j] = vals

    # Manejar log_scale
    if log_scale:
        matriz_plot = np.log10(np.clip(np.abs(matriz), 1e-12, None))
        cbar_label = f'Log10 |Magnitud del Peso| ({tipo})'
        cmap_plot = "magma"
    else:
        matriz_plot = matriz
        cbar_label = f'Magnitud del Peso ({tipo})'
        cmap_plot = "viridis"

    matriz_plot = matriz_plot.T

    # 4. Configuración y despliegue del Heatmap
    plt.figure(figsize=(14, 6))
    ax = sns.heatmap(matriz_plot, cmap=cmap_plot, cbar_kws={'label': cbar_label},
                     xticklabels=max(1, n_iters//10), yticklabels=False)

    # 5. Rastreador forense de columnas dadas de baja (marcas 'X' rojas)
    for j in range(n_cols):
        for i in range(1, len(historial_dict[todas_columnas[j]])):
            val_prev = matriz[i-1, j]
            val_curr = matriz[i, j]
            if not np.isnan(val_prev) and val_prev > 1e-6:
                if np.isnan(val_curr) or val_curr <= 1e-6:
                    ax.text(i + 0.5, j + 0.5, 'X', color='red',
                            ha='center', va='center', fontsize=9, fontweight='bold')
                    break

    plt.title(f"Evolución Dinámica de Memoria de Columnas: {tipo}", fontsize=13)
    plt.xlabel("Número de Iteración del Pipeline", fontsize=11)
    plt.ylabel("Identificador Único de Columna", fontsize=11)

    step_y = max(1, n_cols // 15)
    ax.set_yticks(np.arange(0, n_cols, step_y) + 0.5)
    ax.set_yticklabels([todas_columnas[i] for i in range(0, n_cols, step_y)], rotation=0, fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_computation_times(time_master, time_pricing, filename=None):
    """
    Gráfica del tiempo de ejecución por iteración separando el esfuerzo
    computacional del Master (Mosek) y del Pricing (Gurobi).

    Parámetros:
    -----------
    time_master : list o np.array
        Historial de tiempos que tomó el maestro en cada iteración.
    time_pricing : list o np.array
        Historial de tiempos que tomó el subproblema en cada iteración.
    filename : str, opcional
        Nombre del archivo para guardar el gráfico.
    """
    if not time_master or not time_pricing:
        print("[FLAG - PLOT] Datos de tiempo incompletos.")
        return

    iteraciones = np.arange(1, len(time_master) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(iteraciones, time_master, marker='s', color='darkorange',
            linestyle='--', linewidth=1.5, label='Tiempo Master (Mosek)')
    ax.plot(iteraciones, time_pricing, marker='^', color='forestgreen',
            linestyle='--', linewidth=1.5, label='Tiempo Pricing (Gurobi/Screening)')

    ax.set_title("Perfil de Tiempo de Ejecución por Componente", fontsize=14, fontweight='bold')
    ax.set_xlabel("Iteraciones", fontsize=12)
    ax.set_ylabel("Tiempo de Ejecución (segundos)", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=11)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()


def ejecutar_pipeline_graficacion_master(gen_col, n_samples, n_features, opt_teorico_con=None, opt_skl_val=None):
    """
    Función envolvente que extrae, sanea, sincroniza los datos de la instancia
    del objeto de generación de columnas e invoca el set de gráficos.

    Parámetros:
    -----------
    gen_col : objeto class
        Instancia ejecutada que contiene los historiales del algoritmo.
    n_samples : int
        Número de muestras del dataset procesado.
    n_features : int
        Número de características (dimensionalidad).
    opt_teorico_con : float, opcional
        Óptimo cónico global.
    opt_skl_val : float, opcional
        Óptimo de Scikit-Learn de referencia.
    """
    print("\n[FLAG - DIAGNÓSTICO VISUAL] Iniciando extracción de historiales...")

    # 1. Extracción de vectores objetivos primarios
    ub_history = np.array([v if v is not None else np.nan for v in gen_col.opt_val_fin])
    raw_lb = np.array([v if v is not None else np.nan for v in gen_col.lb_fin])

    # SANITIZACIÓN: Eliminar valores de -inf o banderas de fallback (-6742069)
    # Reemplazándolos por el último valor válido observado (o un valor muy bajo al inicio)
    lb_history = np.copy(raw_lb)
    last_valid = -100.0 # Valor por defecto si todo comienza mal
    for i in range(len(lb_history)):
        val = lb_history[i]
        if np.isnan(val) or val == -np.inf or val == -6742069 or val <= -1e6:
            lb_history[i] = last_valid
        else:
            # Forzamos monotonía ascendente (el lower bound real siempre es el máximo histórico)
            last_valid = max(last_valid, val)
            lb_history[i] = last_valid

    # 2. Sincronización estricta de longitudes para evitar descalces en el gráfico
    min_len = min(len(ub_history), len(lb_history))
    if min_len == 0:
        print("[FLAG - PLOT ERROR] Historiales vacíos. No es posible graficar convergencia.")
        return

    ub_history = ub_history[:min_len]
    lb_history = lb_history[:min_len]
    iteraciones = np.arange(1, min_len + 1)

    # 3. Consolidación de la estructura intermedia 'history_data'
    history_data = {
        'iter': iteraciones,
        'master_value': ub_history,
        'pricing_value': lb_history
    }

    timestamp = datetime.datetime.now().strftime("%H%M%S")

    # 4. Despliegue secuencial de las visualizaciones científicas
    print("-> Desplegando Gráfica Estilo Paper...")
    plot_convergence_paper(
        history=history_data,
        conic_opt=opt_teorico_con,
        title=f"Convergencia Primal-Dual (Dataset {n_samples}x{n_features})",
        filename=f"grafico_paper_{timestamp}.png",
        log_scale=True # Activado para proteger la escala Y frente a LB masivamente negativos
    )

    print("-> Desplegando Gráfica de Métricas Dinámicas y Gaps...")
    plot_convergence_metrics(
        history=history_data,
        conic_opt=opt_teorico_con if opt_teorico_con is not None else ub_history[-1],
        dataset_size=f"{n_samples}x{n_features}",
        use_filter_iters=getattr(gen_col, 'iters_filtro', None)
    )

    print("-> Desplegando Composición Separada con Notación Científica...")
    plot_convergence_split(
        history=history_data,
        conic_opt=opt_teorico_con,
        title=f"Convergencia P-D ({n_samples}x{n_features})",
        filename=f"grafico_split_{timestamp}.png",
        use_log_scale=False # Forzado lineal con ScalarFormatter dinámico
    )

    # 5. Despliegue de los mapas de calor de utilización de memoria
    # Se extraen de los atributos 'memoria_theta' y 'memoria_mu'
    if hasattr(gen_col, 'memoria_theta'):
        print("-> Analizando pesos esparcidos de Puntos (Theta)...")
        plot_heatmap_memoria_robusto(gen_col.memoria_theta, tipo="Puntos (Theta)")

    if hasattr(gen_col, 'memoria_mu') and len(gen_col.memoria_mu) > 0:
        print("-> Analizando pesos esparcidos de Rayos (Mu)...")
        plot_heatmap_memoria_robusto(gen_col.memoria_mu, tipo="Rayos (Mu)", log_scale=True)

    # 6. Despliegue del perfil de tiempos de cómputo
    if hasattr(gen_col, 'time_master') and hasattr(gen_col, 'time_pricing'):
        if len(gen_col.time_master) > 0 and len(gen_col.time_pricing) > 0:
            print("-> Desplegando Perfil de Tiempos de Cómputo...")
            plot_computation_times(
                time_master=gen_col.time_master,
                time_pricing=gen_col.time_pricing,
                filename=f"grafico_tiempos_{timestamp}.png"
            )


# =============================================================================
# EJEMPLO SINTÉTICO DE USO RECOMENDADO (Pauta de Integración)
# =============================================================================
if __name__ == "__main__":
    """
    Este bloque simula la integración estructural al final de tu script principal
    tras finalizar la ejecución de tu objeto gen_col de la clase generacion_DW_2.
    """
    print("\n--- Modo demostración de visualizaciones_tesis.py ---")
    print("Para producción, importa este archivo e invoca 'ejecutar_pipeline_graficacion_master'")

    # Variables de control que ya tienes en tus celdas de Spyder
    samples_demo = 1000
    features_demo = 100000
    opt_teorico_demo = 0.6452

    # Simulación mock del comportamiento de la clase gen_col para validar tipos
    class MockGenCol:
        def __init__(self):
            # Simula una escalera descendente
            self.opt_val_fin = [0.655, 0.652, 0.650, 0.649, 0.648, 0.647, 0.646]
            # Simula un dual que sube desde muy abajo
            self.lb_fin = [-100.0, -10.0, 0.1, 0.5, 0.62, 0.64, 0.643]
            # Marcador de filtros
            self.iters_filtro = [2, 5]
            # Diccionarios de pesos con llaves de tipo string (actualizado al nuevo formato)
            self.memoria_theta = {
                'p_0': [1.0, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0],
                'p_1': [0.0, 0.5, 0.8, 1.0, 0.7, 0.4, 0.1],
                'p_2': [np.nan, np.nan, np.nan, np.nan, 0.3, 0.6, 0.9]
            }

    instance_mock = MockGenCol()

    # Invocación de prueba de la rutina consolidada
    ejecutar_pipeline_graficacion_master(
        gen_col=instance_mock,
        n_samples=samples_demo,
        n_features=features_demo,
        opt_teorico_con=opt_teorico_demo
    )

# from visualizaciones_tesis import ejecutar_pipeline_graficacion_master
# Invocar pasando tu instancia activa 'gen_col'
# ejecutar_pipeline_graficacion_master(
#     gen_col=gen_col,
#     n_samples=X_train.shape[0],
#     n_features=X_train.shape[1],
#     opt_teorico_con=opt_teorico_con # Tu variable con el óptimo del problema cónico completo
# )
