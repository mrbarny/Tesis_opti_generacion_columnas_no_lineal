# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:52:09 2026

@author: matta
"""
# In[]:
    
import joblib
from pathlib import Path
import numpy as np
import gc
import cvxpy as cp
import mosek
import scipy.sparse as sp
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import LinearSVC
import time
import gurobipy as gp
from gurobipy import GRB
# In[2]: Inicializar las funciones master pricing. 
def solve_svm_conic(X, y, C=1.0, time_limit_sec=3600*60, solo_w_b_xi=True):
    n_samples, n_features = X.shape
    y_neg = np.where(y <= 0, -1, 1)
    w = cp.Variable(n_features)
    b = cp.Variable()
    xi = cp.Variable(n_samples, nonneg=True)
    tau = cp.Variable()
    constraints = [
        cp.multiply(y_neg, X @ w + b) >= 1 - xi,
        cp.norm(w, 2) <= tau
    ]
    objective = cp.Minimize(tau + C * cp.sum(xi))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, warm_start=True) #le quitaron el max time para que sea acertado. 
    if solo_w_b_xi == True:
        return (w.value, b.value, xi.value)
    else:
        return [(w.value, b.value, xi.value),prob.value]
def pricing_gurobi(X, y, grad_w_flat, K_rays_dict, C,gurobi_config=None, *args, **kwargs):
    """
    Estrategia de Pricing Cónico Nativo mediante Gurobi Matricial.
    Acepta *args y **kwargs para capturar parámetros extra (Mosek, warm_start, etc.) 
    y evitar errores de firma al ser llamada por la clase.
    """
    
    n_samples, n_features = X.shape
    # 1. Recuperamos M_box del llamado posicional de la clase (es el 7° argumento, cae en *args[0])
    M_box = None
    if len(args) > 0:
        M_box = args[0]
        
    verbose_flag = kwargs.get('verbose', True)
    reportar_inf = kwargs.get('reportar_rayo_como_infinito', True)
    # -----------------------------------------------------------------
    # ESTRATEGIA B: OPTIMIZACIÓN CÓNICA (Rayo Extremo de Gurobi)

    # --- 2. CREACIÓN DEL MODELO NATIVO EN GUROBI ---
    # Creamos un entorno y un modelo limpio
    env = gp.Env()
    env.setParam("OutputFlag", 1 if verbose_flag else 0) # el wea
    model = gp.Model("Subproblema_Tesis_Nativo", env=env)

    model.setParam("InfUnbdInfo", 1)
    model.setParam("Presolve", 0)
    if isinstance(gurobi_config, dict):
        for param_name, param_val in gurobi_config.items():
            if not str(param_name).startswith("MSK_"):
                try:
                    model.setParam(param_name, param_val)
                except Exception:
                    pass

    # --- 3. CREACIÓN DE VARIABLES CON NOMBRE ---
    # --- 3. REGLA DE ORO: USAR addMVar() EN LUGAR DE addVars() ---
    # Creamos las variables como vectores nativos de Gurobi
    # lb=-GRB.INFINITY es vital para w y b ya que por defecto Gurobi las asume >= 0


    # addMVar nos devuelve vectores matemáticos compatibles con el operador @
    w_vars = model.addMVar(shape=n_features, lb=-GRB.INFINITY, name="pesos_w")
    b_var = model.addMVar(shape=1, lb=-GRB.INFINITY, name="sesgo_b") # jules lo cambio. Se mantiene consistencia del codigo asi. 
    xi_vars = model.addMVar(shape=n_samples, lb=0.0, name="holguras_xi")


    # --- 2. OBJETIVO MATRICIAL CON CAMBIO DE DIMENSIÓN ---
    #vector_unos = np.ones((1,n_samples))  
    vector_unos = np.ones(n_samples) * C

    
    # --- 5. RESTRICCIONES NATIVAS ---
    # Agregamos las 100,000 restricciones de margen funcional
    # Para agilizar el proceso en matrices grandes, calculamos el producto punto de cada fila. Jules cambio de 510 a 515. 
    y_neg = np.where(y <= 0, -1, 1)
    y_neg_diag = sp.diags(y_neg, dtype=np.float64)
    X_y = y_neg_diag @ X
    y_neg_col = y_neg.reshape(-1, 1).astype(np.float64)
    
    model.addConstr(X_y @ w_vars + y_neg_col @ b_var + xi_vars >= 1.0, name="restriccion_margen")
    
    if M_box is not None:
        # Si activas la caja para estabilizar, Gurobi acotará los pesos w
        model.addConstr(w_vars <= M_box, name="caja_superior")
        model.addConstr(w_vars >= -M_box, name="caja_inferior")

    # Ahora el operador @ funcionará perfectamente con los MVars de Gurobi
    # Al hacer la multiplicación, Gurobi reduce la expresión a un escalar lineal nativo
    model.setObjective(vector_unos @ xi_vars - grad_w_flat @ w_vars, GRB.MINIMIZE)
    # --- 6. OPTIMIZACIÓN ---
    print("Optimizando con Gurobi matricial...")
    model.optimize()
        
    if model.Status == GRB.UNBOUNDED:
        print("\n==================================================")
        print("   EXTRACCIÓN NOMINAL DEL RAYO DIRECTA DESDE GUROBI NATIVO  ")
        print("==================================================")
        
        # En Gurobi nativo, como no hubo dualización de CVXPY,
        # el rayo de las variables originales está directamente en el atributo .UnbdRay
        
        # EXTRACCIÓN INSTANTÁNEA: Sin bucles for de Python, directo desde C++
        delta_w_nativo = w_vars.UnbdRay
        print('rayo',delta_w_nativo)
        delta_b_nativo = b_var.UnbdRay[0]
        
        # Buscamos cada una de las 100,000 variables de pesos por su nombre exacto
        norm = np.linalg.norm(delta_w_nativo)
        
        if norm > 1e-9 :
            scale = 1.0   
            w_sparse = sp.csc_matrix(delta_w_nativo.reshape(-1, 1), dtype=np.float32)
            # Retornamos -inf para indicar explícitamente que es un rayo
            if reportar_inf:
                lb_retorno = -np.inf
            else:
                # Retorna el valor de la tasa de descenso real de la arista (un escalar negativo real)
                lb_retorno = -np.dot(grad_w_flat, delta_w_nativo * scale)
            return (w_sparse * scale, delta_b_nativo * scale, None, lb_retorno) #aca no consideré que fuese necesario entregar XI. 
        else:
            print("[pricing] Advertencia: El rayo recuperado es nulo (norma ~ 0).")
            return None
                
    # -----------------------------------------------------------------
    # MANEJO DEL CASO OPTIMAL (Solución Básica Factible / Punto)
    # -----------------------------------------------------------------
    elif model.Status == GRB.OPTIMAL:
        print("[Pricing] Solución acotada encontrada (PUNTO).")
        # Retornamos en el formato estándar (w_val, b_val, xi_val, obj_val)
        return (w_vars.X, b_var.X[0], xi_vars.X, model.ObjVal)
    else:
        print(f"[Pricing] Gurobi finalizó con un estatus inesperado: {model.Status}")
        return None
def generar_canonico_sparse(n_features, coordenada, n_samples, dtype=np.float32):
    xi_placeholder = sp.csc_matrix((n_samples, 1), dtype=dtype) if n_samples is not None else None
    if coordenada == -1:
        w_sparse = sp.csc_matrix((n_features, 1), dtype=dtype)
        return (w_sparse, 0.0, xi_placeholder)
    else:
        w_sparse = sp.csc_matrix(
            (np.array([1.0], dtype=dtype), (np.array([int(coordenada)]), np.array([0]))),
            shape=(n_features, 1), dtype=dtype)
        return (w_sparse, 0.0, xi_placeholder)

def generar_K_canonico_sparse(n_features, n_samples, tamaño=0.1):
    np.random.seed(420)
    K_generado = [generar_canonico_sparse(n_features, coordenada=-1, n_samples=n_samples)] 
    num_canonicos = int(n_features * tamaño)
    if num_canonicos >= n_features: columnas = np.arange(n_features)
    else: columnas = np.random.choice(n_features, num_canonicos, replace=False)
    print(f"Generando {len(columnas)} vectores canónicos sparse...")
    for i in columnas:
        K_generado.append(generar_canonico_sparse(n_features, coordenada=i, n_samples=n_samples))
    print("Set K canónico sparse generado.")
    return K_generado

def _precompute_sparse_matrix(K_list, n_features):
    """
    Toma una lista K (de puntos o rayos) y la apila en una matriz CSC.
    """
    if not K_list:
        # Devuelve una matriz esparsa vacía con el shape correcto
        return sp.csc_matrix((n_features, 0), dtype=np.float32)

    sparse_Ws_list = []
    for k_idx, col in enumerate(K_list):
        w = col[0] # Asumimos que K es una lista de tuplas (w, b, xi)
        
        if sp.issparse(w):
            if w.shape == (n_features, 1):
                sparse_Ws_list.append(w.astype(np.float32))
            elif w.shape == (1, n_features):
                sparse_Ws_list.append(w.T.astype(np.float32))
            else:
                raise ValueError(f"Columna esparsa {k_idx} tiene shape {w.shape} (esperado: ({n_features}, 1) o (1, {n_features}))")
        
        elif isinstance(w, np.ndarray):
            w_col = np.asarray(w, dtype=np.float32).reshape(-1, 1)
            if w_col.shape[0] != n_features:
                raise ValueError(f"Columna densa {k_idx} tiene {w_col.shape[0]} filas (esperado: {n_features})")
            sparse_Ws_list.append(sp.csc_matrix(w_col))
        
        else:
            raise TypeError(f"Elemento w en K (índice {k_idx}) es tipo {type(w)}")
            
    return sp.hstack(sparse_Ws_list, format='csc')
def solve_master_primal_v3(X, y, K, tipo, 
                           K_rayos=None, # <-- MODIFICACIÓN DW: Nuevo input
                           C=1.0, mosek_params={}, 
                           M_box=1e4,warm_start=True, verbose=True,tijonov=False):
    
    y_neg = np.where(y <= 0, -1, 1)
    
    # Manejo de listas de entrada
    if K is None: K = []
    if K_rayos is None: K_rayos = [] # <-- MODIFICACIÓN DW
        
    K_len = len(K)
    R_len = len(K_rayos) # <-- MODIFICACIÓN DW
    
    n_samples = len(y)
    n_features = X.shape[1] 

    # --- Variables ---
    # Variables para Puntos (K)
    if K_len > 0:
        if tipo in ["afin", "libre"]:
            theta = cp.Variable((K_len, 1))
        elif tipo in ["convexo", "conico", "mayor_uno"]:
            theta = cp.Variable((K_len, 1), nonneg=True)
        else:
            print(f"Error: tipo '{tipo}' no es válido para K.")
            return None # Devolver tupla de Nones
    else:
        theta = None # No hay puntos

    # MODIFICACIÓN DW: Variables para Rayos (K_rayos)
    if R_len > 0:
        # Los rayos SIEMPRE siguen una combinación cónica (nonneg=True)
        mu = cp.Variable((R_len, 1), nonneg=True)
    else:
        mu = None
    
    eta = cp.Variable()
    b = cp.Variable()
    xi = cp.Variable((n_samples, 1), nonneg=True) 

    # --- Precomputar Ws_sparse (Usando la función auxiliar) ---
    Ws_puntos = _precompute_sparse_matrix(K, n_features)
    Ws_rayos = _precompute_sparse_matrix(K_rayos, n_features) # <-- MODIFICACIÓN DW

    # --- MODIFICACIÓN DW: Construcción de w_combo ---
    w_combo_terms = []
    if theta is not None and K_len > 0:
        w_combo_terms.append(Ws_puntos @ theta)
    if mu is not None and R_len > 0:
        w_combo_terms.append(Ws_rayos @ mu)

    # Si ambas listas están vacías, w_combo es un vector cero constante
    if not w_combo_terms:
        w_combo = cp.Constant(np.zeros((n_features, 1), dtype=np.float32))
    else:
        w_combo = cp.sum(w_combo_terms)
    
    # --- Restricciones ---
    y_neg_col = y_neg.reshape(-1, 1)
    if M_box is not None: 
        constraints_dict = {
            "soc_norm": cp.SOC(eta, w_combo),
            "classification": cp.multiply(y_neg_col, X @ w_combo + b) >= 1 - xi, 
            "master_box_pos": w_combo <= M_box, 
            "master_box_neg": w_combo >= -M_box
        }
    else:
        constraints_dict = {
            "soc_norm": cp.SOC(eta, w_combo),
            "classification": cp.multiply(y_neg_col, X @ w_combo + b) >= 1 - xi
        }
    # Restricciones para Puntos (theta) - NADA CAMBIA AQUÍ
    if theta is not None:
        if tipo in ["convexo", "afin"]: constraints_dict["theta_sum"] = (cp.sum(theta) == 1)
        elif tipo == "mayor_uno": constraints_dict["theta_sum"] = (cp.sum(theta) >= 1)
        elif tipo == "conico": constraints_dict["theta_sum"] = (cp.sum(theta) >= 0)
    
    # NOTA: 'mu' (rayos) no tiene restricción de suma, solo 'nonneg=True'
    # lo cual ya se definió en la variable.
    
    if tijonov==True: 
        epsilon = 1e-6 #1/features
        reg_term = 0
        if theta is not None: reg_term += epsilon * cp.sum_squares(theta)
        if mu is not None:    reg_term += epsilon * cp.sum_squares(mu)
        objective = cp.Minimize(eta + C * (cp.sum(xi)) + reg_term)
    else: 
        
        objective = cp.Minimize(eta + C * (cp.sum(xi)))
        
    prob = cp.Problem(objective, list(constraints_dict.values()))
    try:
        prob.solve(solver=cp.MOSEK, warm_start=warm_start, verbose=verbose) #el warm start a veces cagonea cuando la solucion numericamente cambia poco en el lategame. ayuda harto en el early eso si.  
    except cp.error.SolverError:
        print("⚠️ Master CRASH con params estrictos/warm_start. Reintentando relajado...")
        try:
            # Intento 2: Sin warm_start y tolerancias relajadas
            params_relaxed = mosek_params.copy()
            params_relaxed["MSK_DPAR_INTPNT_TOL_REL_GAP"] = 1e-5
            params_relaxed["MSK_DPAR_INTPNT_TOL_PFEAS"] = 1e-5
            params_relaxed["MSK_DPAR_INTPNT_TOL_DFEAS"] = 1e-5
            
            prob.solve(solver=cp.MOSEK, warm_start=False, verbose=True)
            print("✅ Master recuperado.")
        except Exception as e:
            print(f"🔥 Master falló definitivamente: {e}")
            return (None, None, None, None, None, None, None, None, None)
        
    # --- CÁLCULO DE GAP Y GRADIENTE ---

    grad_w_correcto = np.zeros(n_features, dtype=np.float32) 
    alpha = None
    
    if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        try:
            alpha = constraints_dict["soc_norm"].dual_value[1]
            pi_dual = constraints_dict["classification"].dual_value
            if alpha is not None and pi_dual is not None:
                sum_yxpi = X.T @ (y_neg * pi_dual.flatten())
                alpha_flat = alpha.flatten()
                sum_yxpi_flat = sum_yxpi.flatten()
                grad_w_correcto = alpha_flat + sum_yxpi_flat
                if tijonov==True: 
                    print(prob.value)
                    primal_value_UB = prob.value-reg_term.value
                else:
                    primal_value_UB = prob.value
            else: print("ADVERTENCIA: No se pudieron obtener las variables duales (valores None).")
        except Exception as e:
            print(f"ADVERTENCIA: Error al calcular el gradiente: {e}"); alpha = None
    else: print(f"ADVERTENCIA: Master no resolvió óptimamente (status: {prob.status}).")

    print(f"Norma del gradiente correcto para w: {np.linalg.norm(grad_w_correcto):.4f}")

    # --- MODIFICACIÓN DW: ORDEN DE RETORNO (9 VALORES) ---
    # Se añade mu_val. ¡Esto cambia el orden de los resultados!
    
    theta_val = theta.value.flatten() if theta is not None and theta.value is not None else np.array([])
    mu_val = mu.value.flatten() if mu is not None and mu.value is not None else np.array([]) # <-- NUEVO
    
    w_combo_val = w_combo.value.flatten() if w_combo.value is not None else None
    b_val = b.value.item() if b.value is not None else None
    xi_val = xi.value.flatten() if xi.value is not None else None
    alpha_val = alpha.flatten() if alpha is not None else None

    # El nuevo orden de retorno ahora tiene 9 elementos
    return (
        theta_val,      # [0] Coeficientes de Puntos (K)
        eta.value,      # [1]
        alpha_val,      # [2]
        w_combo_val,    # [3]
        primal_value_UB,# [4] <-- CUIDADO: índice cambiado (antes 4)
        b_val,          # [5] <-- CUIDADO: índice cambiado (antes 5)
        xi_val,         # [6] <-- CUIDADO: índice cambiado (antes 6)
        grad_w_correcto, # [7] <-- CUIDADO: índice cambiado (antes 7)
        mu_val,         # [8] Coeficientes de Rayos (K_rayos) <-- NUEVO
        sum_yxpi_flat , pi_dual #[9] y [10]
        
    )   
    


# In[]:

if __name__ == "__main__":
    # n_samples = 5
# n_features = 3
# X = sp.csr_matrix(np.random.rand(n_samples, n_features))
# y = np.array([-1, 1, -1, 1, -1])

    X=np.array([[2,1],[1,2],[3,3]])

    X = sp.csr_matrix(X)
    y= np.array([1, 1, -1])
    n_samples, n_features = X.shape

    K_ini=[[np.array([[-0.8,0]]),0.0,None],[np.zeros(2),0.0,None]] #la que se obtiene con solo 50% de las features
    K_zero= [[np.zeros(2),0.0,None]]
    K_ray=[[np.array([[-0.242, -0.97014253]]),0.0,None]]
    #K_ray=[[np.array([[-0.242, -0.97014253]]),0.0,None],[np.array([[0.44, -0.894]]),0.0,None]]

    res_con= solve_svm_conic(X, y, C=1.0, time_limit_sec=3600*60, solo_w_b_xi=False)

    res_master =solve_master_primal_v3(X, y, K_ini, 'convexo', 
                               K_rayos=None, # <-- MODIFICACIÓN DW: Nuevo input
                               C=1.0, mosek_params={}, 
                               M_box=None,warm_start=False, verbose=True,tijonov=False)


    grad= res_master[2]+res_master[9]
    grad_raro= res_master[2]-res_master[9]
    # In[]:
    res_pricing_pos=pricing_gurobi(X,y,grad,None,1)
    res_pricing_neg=pricing_gurobi(X,y,grad*-1,None,1)
    res_pricing_pos_raro=pricing_gurobi(X,y,grad_raro,None,1)
    res_pricing_neg_raro=pricing_gurobi(X,y,grad_raro*-1,None,1)

    K_ray_neg=[[res_pricing_neg[0],0.0,None]]
    K_ray_pos=[[res_pricing_pos[0],0.0,None]]
    K_ray_neg_raro=[[res_pricing_neg_raro[0],0.0,None]]
    K_ray_pos_raro=[[res_pricing_pos_raro[0],0.0,None]]

    res_master_pos =solve_master_primal_v3(X, y, K_ini, 'convexo', 
                               K_rayos=K_ray_pos, # <-- MODIFICACIÓN DW: Nuevo input
                               C=1.0, mosek_params={}, 
                               M_box=None,warm_start=False, verbose=True,tijonov=False)
    res_master_neg =solve_master_primal_v3(X, y, K_ini, 'convexo', 
                               K_rayos=K_ray_neg, # <-- MODIFICACIÓN DW: Nuevo input
                               C=1.0, mosek_params={}, 
                               M_box=None,warm_start=False, verbose=True,tijonov=False)
    # In[]:
    res_master_pos_raro =solve_master_primal_v3(X, y, K_ini, 'convexo', 
                               K_rayos=K_ray_pos_raro, # <-- MODIFICACIÓN DW: Nuevo input
                               C=1.0, mosek_params={}, 
                               M_box=None,warm_start=False, verbose=True,tijonov=False)
    res_master_neg_raro =solve_master_primal_v3(X, y, K_ini, 'convexo', 
                               K_rayos=K_ray_neg_raro, # <-- MODIFICACIÓN DW: Nuevo input
                               C=1.0, mosek_params={}, 
                               M_box=None,warm_start=False, verbose=True,tijonov=False)


    #res_master_pos y res_master_neg_raro llegan al optimo. Tambien usar el K_ray_neg_raro como son llega al optimo. 
    #La diferencia es el patron de alfa +pi x y =0 en algunos casos, y en otros no. Eso es interesante. 

    # In[]:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC


def graficar_svm_generacion_columnas(X, y, w_opt, b_opt, w_inputs=None, w_ray=None, title="SVM - Generación de Columnas"):
    # Si X es una matriz dispersa (sparse), la convertimos a densa para poder graficar

        
    plt.figure(figsize=(10, 8))
    
    # 1. Graficar los puntos de datos
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', marker='o', s=80, label='Clase +1', edgecolors='k', zorder=3)
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', marker='s', s=80, label='Clase -1', edgecolors='k', zorder=3)
    
    # Calcular límites iniciales basados en los datos
    ax = plt.gca()
    xlim = list(ax.get_xlim())
    ylim = list(ax.get_ylim())
    
    # --- CORRECCIÓN CLAVE ---
    # Asegurarnos de que el origen (0,0) siempre esté dentro del gráfico para ver los vectores
    xlim[0] = min(xlim[0], -0.5)
    xlim[1] = max(xlim[1], 0.5)
    ylim[0] = min(ylim[0], -0.5)
    ylim[1] = max(ylim[1], 0.5)
    
    # 2. Graficar el hiperplano óptimo y sus márgenes
    xx = np.linspace(xlim[0], xlim[1], 50)
    if w_opt[1] != 0:
        yy_opt = -(w_opt[0] * xx + b_opt) / w_opt[1]
        yy_margin_up = -(w_opt[0] * xx + b_opt - 1) / w_opt[1]
        yy_margin_down = -(w_opt[0] * xx + b_opt + 1) / w_opt[1]
        
        plt.plot(xx, yy_opt, 'k-', linewidth=2, label='Hiperplano Óptimo', zorder=1)
        plt.plot(xx, yy_margin_up, 'k--', linewidth=1, alpha=0.6, zorder=1)
        plt.plot(xx, yy_margin_down, 'k--', linewidth=1, alpha=0.6, zorder=1)

    # Función auxiliar mejorada para graficar vectores
    def plot_vector(vec, color, label, linestyle='-'):
        # Normalizar y escalar el vector al 20% del tamaño del gráfico para que siempre sea visible
        scale = max(xlim[1]-xlim[0], ylim[1]-ylim[0]) * 0.2 / (np.linalg.norm(vec) + 1e-8)
        v = vec * scale
        
        # Actualizar límites si el vector es muy grande
        xlim[0], xlim[1] = min(xlim[0], v[0]*1.1), max(xlim[1], v[0]*1.1)
        ylim[0], ylim[1] = min(ylim[0], v[1]*1.1), max(ylim[1], v[1]*1.1)
        
        # Dibujar la línea y la flecha explícitamente
        plt.plot([0, v[0]], [0, v[1]], color=color, linestyle=linestyle, linewidth=2.5, label=label, zorder=4)
        plt.annotate('', xy=(v[0], v[1]), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color=color, lw=2.5), zorder=4)

    # 3. Graficar el vector w óptimo
    plot_vector(w_opt, 'black', 'w óptimo')

    # 4. Graficar los vectores input
    if w_inputs is not None:
        for i, w_in in enumerate(w_inputs):
            label = 'w inputs' if i == 0 else "" 
            plot_vector(w_in, 'green', label)

    # 5. Graficar el vector rayo extremo
    if w_ray is not None:
        plot_vector(w_ray, 'magenta', 'Rayo Extremo (Gurobi)', linestyle='--')

    # Ajustar estética final
    plt.axhline(0, color='grey', linewidth=1, linestyle='-', zorder=0)
    plt.axvline(0, color='grey', linewidth=1, linestyle='-', zorder=0)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.show()

# ==========================================
# SECCIÓN DE PRUEBA (DUMMY DATA)
# ==========================================
if __name__ == "__main__":
    # Generar dataset de prueba 2D separable
    # X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.5)
    # y = np.where(y == 0, -1, 1) # Convertir a {-1, 1}
    
    # Entrenar un SVM lineal simple para obtener el w_opt y b_opt real

    if sp.issparse(X):
        X = X.toarray()
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(X, y)
    w_opt_real = clf.coef_[0]
    b_opt_real = clf.intercept_[0]
    
    # Simular vectores inputs (las columnas que le pasaste a tu maestro)
    # Por ejemplo, vectores aleatorios iniciales o de un K-Forest
    
    w_input_1 = K_ini[0][0][0]
    #w_input_2 = np.array([0, 1])
    #lista_w_inputs = [w_input_1, w_input_2]
    lista_w_inputs=[w_input_1,w_input_1]
    
    # Simular un rayo extremo entregado por Gurobi (status Unbounded)
    # Suele ser una dirección que apunta hacia donde el costo reducido decrece infinitamente
    w_rayo_gurobi = K_ray[0][0][0]  
    
    # Llamar a la función
    graficar_svm_generacion_columnas(X, y, w_opt_real, b_opt_real, lista_w_inputs, w_rayo_gurobi)
    
    
    