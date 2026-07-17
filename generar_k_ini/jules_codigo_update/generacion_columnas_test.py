# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 19:41:51 2026

@author: matta
"""

# In[0]: 
# --- CÓMO CARGARLAS ---

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

absolute_path = Path("K_ini.py").resolve() #o el nombre de este .py
dir_path = absolute_path.parent  


K_ini_loaded = joblib.load(dir_path /'K_ini_end_2.pkl')
K_ini_canonico_loaded= joblib.load(dir_path /'K_ini_canonico_list.pkl')
print(f"Cargadas {len(K_ini_loaded)} columnas.")


load_path = dir_path / "data_MaxAbsScaler.joblib"

# 2. Cargar el diccionario completo
data_MaxAbsScaler = joblib.load(load_path)

# 3. Extraer cada variable del diccionario
X_train = data_MaxAbsScaler["X_train"]
X_test = data_MaxAbsScaler["X_test"]
y_train = data_MaxAbsScaler["y_train"]
y_test = data_MaxAbsScaler["y_test"]
tipo_escalador = data_MaxAbsScaler["tipo"]

print(f"¡Data cargada con éxito! Tipo de escalador: {tipo_escalador}")
# In[1]: Test base



# --- FUNCIONES DE SVM (Sin cambios) ---
def skl_svm(X, y, X_test, C=1.0, loss="hinge", max_iter=100000, solo_w_b_xi=False):
    y = np.where(y <= 0, -1, 1)
    clf = OneVsRestClassifier(LinearSVC(C=C, loss=loss, max_iter=max_iter))
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    est = clf.estimators_[0]
    w = est.coef_.flatten()
    b = est.intercept_[0]
    margins = y * (X @ w + b)
    xi = np.maximum(0, 1 - margins)
    obj_val = np.linalg.norm(w) + C * np.sum(xi)
    # ... (prints omitidos por brevedad) ...
    if solo_w_b_xi == True:
        return (w, b, xi)
    else:
        return [(w, b, xi), obj_val, y_pred]

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

#el resultado del OG. 
skl_svm_res = [(None,None,None), 0]
#el resultado del conico
con_svm_res = [(None,None,None), 0]
opt_teorico_con=con_svm_res[1] 

# In[2]: Funciones auxiliares 
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

def pares_en_K_sparse(K):
    ya = set()
    it = K.values() if isinstance(K, dict) else K
    for tpl in it:
        w_sparse = tpl[0]
        if w_sparse.nnz == 1:
            j = int(w_sparse.indices[0])
            sg = int(np.sign(w_sparse.data[0])) or 1
            ya.add((j, sg))
    return ya

def generar_canonico_sparse(n_features, coordenada, n_samples, dtype=np.float32, signo=+1.0):
    xi_placeholder = sp.csc_matrix((n_samples, 1), dtype=dtype) if n_samples is not None else None
    if coordenada == -1:
        w_sparse = sp.csc_matrix((n_features, 1), dtype=dtype)
        return (w_sparse, 0.0, xi_placeholder)
    else:
        val = float(np.sign(signo))
        if val == 0.0: val = 1.0
        w_sparse = sp.csc_matrix(
            (np.array([val], dtype=dtype), (np.array([int(coordenada)]), np.array([0]))),
            shape=(n_features, 1), dtype=dtype)
        return (w_sparse, 0.0, xi_placeholder)
    
def generar_canonico_con_signo_sparse(n_features, n_samples, j, signo=+1.0):
    return generar_canonico_sparse(
        n_features=n_features, coordenada=int(j), n_samples=n_samples, dtype=np.float32, signo=signo)


def _pack_solution(w_val, b_val, xi_val, obj_val):
    """Helper para empaquetar en sparse y retornar"""
    if w_val is None: return (None, None, None, 0.0)
    
    w_sparse = sp.csc_matrix(w_val.reshape(-1, 1), dtype=np.float32)
    xi_sparse = sp.csc_matrix(xi_val.reshape(-1, 1), dtype=np.float32)
    
    return (w_sparse, b_val, xi_sparse, obj_val)

def mosek_params_from_tol(tol, *, threads=None, presolve_level=1, optimizer_code=0):
    tol = float(tol)
    p = {
        "MSK_DPAR_OPTIMIZER_MAX_TIME": 7200.0,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": tol, # Relative gap termination tolerance used by the interior-point optimizer for conic problems. #default = 1e-8 # MSK_DPAR_INTPNT_QO_TOL_REL_GAP for cuadratic
        "MSK_DPAR_INTPNT_TOL_PFEAS":   tol , #Primal feasibility tolerance used by the interior-point optimizer for conic problems. #default = 1e-8
        "MSK_DPAR_INTPNT_TOL_DFEAS":   tol , #factibilidad dual   # Dual feasibility tolerance used by the interior-point optimizer for quadratic problems.   #default = 1e-8
        "MSK_DPAR_BASIS_TOL_X":     max(tol, 1e-9), #Maximum absolute dual bound violation in an optimal basic solution.  #default = 1.0e-6
        "MSK_DPAR_BASIS_TOL_S":     max(tol, 1e-9), # Maximum relative dual bound violation allowed in an optimal basic solution. #defaul = 1.0e-12
        "MSK_IPAR_OPTIMIZER": int(optimizer_code), # 0 interior, 1 simplex, 2 dual simplex.
        "MSK_IPAR_PRESOLVE_USE": int(presolve_level),  # THE DEVIL INCARNATE 
    }
    if presolve_level == 0:
        p["MSK_IPAR_PRESOLVE_ELIMINATOR_MAX_NUM_TRIES"] = 0
        p["MSK_IPAR_PRESOLVE_LINDEP_USE"] = 0
    if threads is not None:
        p["MSK_IPAR_NUM_THREADS"] = int(threads)
    return p

def convertir_dict_a_K(K_dict):
    """Convierte un dict de columnas en una lista de tuplas (w,b,xi) y una lista de claves"""
    return list(K_dict.values()), list(K_dict.keys())
# In[2]: Inicializar las funciones master pricing. 

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
    )




# In[3]: funciones
def screen_from_dual_inequalities(X, alpha, Y, *, eps=1e-8, k_max=50, already=None):
    """
    Implementa las 4 desigualdade (todas deben ser >= 0): el punto es xi=1 el resto 0. 
      v1 = -alpha - X^T Y #costo reducido para W positivo
      v2 =  alpha + X^T Y #costo reducido para W negativos
      v3 =  1^T Y #costo reducido para b positivo
      v4 = -1^T Y #costo reducido para b negativo    

    - Si alguna es < 0 (por debajo de -eps), hay violación ⇒ esa columna mejora.
    - Devuelve top-K columnas violadas, su signo recomendado (+e_j o -e_j) y severidad.

    Parámetros:
      X: matriz (n_samples x n_features). Sirve densa o scipy.sparse CSR/CSC (usa .dot).
      alpha: vector (n_features,).
      Y: vector (n_samples,) que aparece en X^T Y y en 1^T Y.
      eps: tolerancia numérica para decidir violación.
      k_max: máximo de columnas a proponer en este screening.
      already: colección opcional de índices ya presentes en K (para evitar duplicados).

    Retorna:
      idx  : índices de features seleccionados (top-K por severidad)
      sign : +1 si conviene +e_j, -1 si conviene -e_j (según qué desigualdad viola)
      sev  : severidad (magnitud positiva de la violación)
      meta : diccionario con sumY, sugerencia de sesgo 'b' y conteos de violaciones
    """
    
    # (Tu código para screen_from_dual_inequalities va aquí, sin cambios)
    a  = np.asarray(alpha, float).ravel()
    Yv = np.asarray(Y,     float).ravel()
    s = X.T.dot(Yv) if hasattr(X, "dot") else X.T @ Yv
    s = np.asarray(s, float).ravel()
    v1 = -a - s; v2 =  a + s; sumY = float(np.sum(Yv)); v3 = -sumY; v4 = sumY
    viol1 = v1 < 0-eps; viol2 = v2 < 0-eps
    if already:
        mask1 = np.ones_like(a, dtype=bool); mask2 = np.ones_like(a, dtype=bool)
        if any(isinstance(t, tuple) and len(t) == 2 for t in already):
            ban_pos = {j for (j,sg) in already if int(sg) == +1}
            ban_neg = {j for (j,sg) in already if int(sg) == -1}
            if ban_pos: mask1[np.fromiter(ban_pos, dtype=int)] = False
            if ban_neg: mask2[np.fromiter(ban_neg, dtype=int)] = False
        else:
            ban = np.fromiter(already, dtype=int); mask1[ban] = False; mask2[ban] = False
        viol1 &= mask1; viol2 &= mask2
    sev1 = -(v1[viol1]); sev2 = -(v2[viol2])
    idx1 = np.flatnonzero(viol1); idx2 = np.flatnonzero(viol2)
    idx_all  = np.concatenate([idx1, idx2]); sev_all  = np.concatenate([sev1, sev2])
    sign_all = np.concatenate([-np.ones_like(idx1, float), +np.ones_like(idx2, float)])
    if sev_all.size == 0:
        return (np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float),
                {"sumY": sumY, "b_suggest": 0, "n_viol1": 0, "n_viol2": 0})
    k = int(min(k_max, sev_all.size))
    top_idx_local = np.argpartition(sev_all, -k)[-k:]
    order = np.argsort(sev_all[top_idx_local])[::-1]
    sel = top_idx_local[order]
    idx  = idx_all[sel]; sign = sign_all[sel]; sev  = sev_all[sel]
    b_suggest = 0
    if v3 < -eps:   b_suggest = +1
    elif v4 < -eps: b_suggest = -1
    meta = {"sumY": sumY, "b_suggest": b_suggest, "n_viol1": int(viol1.sum()), "n_viol2": int(viol2.sum())}
    return idx, sign, sev, meta
def generar_set_columnas_costos_reducidos_sparse(
    X, y, alpha, K, n_features, n_samples, eps=1e-8, k_max=25, debug=False):
    """ Versión sparse del fallback. Usa helpers esparsos. """
    a = np.asarray(alpha, float).ravel()
    yv = np.asarray(y, float).ravel()
    already_pairs = pares_en_K_sparse(K) 
    idx, sign, sev, meta = screen_from_dual_inequalities(
        X, a, yv, eps=eps, k_max=k_max, already=already_pairs)
    if idx.size == 0:
        info = {"agregadas": 0, "idx": idx, "sign": sign, "sev": sev, "meta": meta}
        return K, info

    unique_idx, unique_sign, unique_sev = [], [], []
    seen_batch = set()
    for j, s, v in zip(idx, sign, sev):
        key = (int(j), int(np.sign(s)) or 1)
        if key in already_pairs or key in seen_batch: continue
        seen_batch.add(key)
        unique_idx.append(int(j))
        unique_sign.append(int(np.sign(s)) or 1)
        unique_sev.append(float(v))
    if not unique_idx:
        info = {"agregadas": 0, "idx": [], "sign": [], "sev": [], "meta": meta}
        return K, info

    nuevas = [generar_canonico_con_signo_sparse(n_features, n_samples, j, s)
              for j, s in zip(unique_idx, unique_sign)]
    agregadas = 0
    if isinstance(K, dict):
        try:
            next_idx = max(int(str(k).split('_')[1]) for k in K.keys() if '_' in str(k)) + 1
        except ValueError:
            next_idx = len(K)
        for col in nuevas:
            K[f'k_{next_idx}'] = col; next_idx += 1; agregadas += 1
        K_actualizado = K
    else:
        K.extend(nuevas); agregadas = len(nuevas); K_actualizado = K
    info = {"agregadas": agregadas, "idx": np.asarray(unique_idx, dtype=int),
            "sign": np.asarray(unique_sign, dtype=float), "sev": np.asarray(unique_sev, dtype=float), "meta": meta}
    return K_actualizado, info

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

    if isinstance(gurobi_config, dict):
        for param_name, param_val in gurobi_config.items():
            model.setParam(param_name, param_val)
    else:
        model.setParam("InfUnbdInfo", 1)
        model.setParam("Presolve", 0)

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
        delta_b_nativo = b_var.UnbdRay[0]
        
        # Buscamos cada una de las 100,000 variables de pesos por su nombre exacto
        norm = np.linalg.norm(delta_w_nativo)
        
        if norm > 1e-9 :
            scale = 1.0 / norm  
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
            



# In[4]:


import numpy as np
import scipy.sparse as sp
import time

class GeneracionColumnasDW_2:
    def __init__(self, tol, tol_master, M_box=None, threads=None):
        """
        Constructor Optimizado para Alta Dimensionalidad (100,000 features).
        Elimina parámetros obsoletos de solvers cruzados.
        """
        self.tol = tol
        self.M_box = M_box            # Caja del Master (si se requiere regularizar)
        self.M_box_pricing = None     # Por defecto el Pricing NO requiere caja (Gurobi maneja Unbounded)
        self.umbral_theta = 1e-6
        
        # --- Configuración Inicial de Gurobi Pricing ---
        self.pricing_gurobi_config = {
            "InfUnbdInfo": 1,         # Habilita el cálculo del rayo de escape nativo
            "Presolve": 0             # Desactivado para que el Simplex mapee aristas puras
        }
        
        # --- Parámetros Base del Master (Mosek) ---
        self.master_mosek_params = mosek_params_from_tol(tol_master, threads=threads)
        
        # --- Historiales Globales ---
        self.opt_val_fin = [] 
        self.lb_fin = []      
        self.alpha_set_fin = [] 
        self.time_master = []
        self.time_pricing = []
        
        # --- Estado del Algoritmo ---
        self.grad_w_actual = None 
        self.gradient_strategy = "full_gradient"
        self.terminamos = False
        self.status = "init"
        self.i = 0
        self.verbose = True
        self.warm_start = True
        
        # --- Almacenes de Columnas ---
        self.K_points_dict = {}       # Almacén de Puntos (Soluciones Acotadas)
        self.K_rays_dict = {}         # Almacén de Rayos (Direcciones de Fuga)
        self.memoria_theta = {}       # Historial de coeficientes del Master para Puntos
        self.memoria_mu = {}          # Historial de coeficientes del Master para Rayos
        
        # Contadores de IDs unívocos
        self.cnt_points = 0
        self.cnt_rays = 0
        self.X = None
        self.y = None
        
        # --- NUEVA BANDERA: Factor de Suavizado del Gradiente ---
        self.usar_suavizado_gradiente = False  # Por defecto desactivado
        self.smoothing_factor = 0.4            # Peso del gradiente histórico (0.0 a 1.0)
        
        # --- NUEVA BANDERA: Reportar Rayos de Gurobi como -inf o como ObjVal ---
        self.reportar_rayo_como_infinito = True # Por defecto True (Tu Caso A inteligente)

    def ingresar_data(self, X_train, y_train):
        """ Inyección de matrices de entrenamiento """
        self.X = X_train
        self.y = y_train
        self.n_samples, self.n_features = X_train.shape
    
    def ingresar_parametros(self, C, M, K_ini_points, K_ini_rays=None, 
                            tipo="convexo", 
                            pricing=pricing_gurobi, 
                            gradient_strategy="full_gradient", 
                            pricing_acceleration=False,
                            tijonov=True,usar_suavizado_gradiente=False,      # <--- NUEVO
                            smoothing_factor=0.4,                 # <--- NUEVO
                            reportar_rayo_como_infinito=True):    # <--- NUEVO
                            
        """
        Inicialización del flujo de generación de columnas.
        Remueve variables redundantes inactivas de aceleración de cotas.
        """
        self.C = C        
        self.tipo = tipo
        self.M = M
        self.pricing = pricing
        self.gradient_strategy = gradient_strategy
        self.pricing_acceleration = pricing_acceleration
        self.tijonov = tijonov
        
        # Asignación de las nuevas estrategias de control paramétrico
        self.usar_suavizado_gradiente = usar_suavizado_gradiente
        self.smoothing_factor = smoothing_factor
        self.reportar_rayo_como_infinito = reportar_rayo_como_infinito
        
        # Banderas por defecto para encender/apagar estrategias dentro del pricing de forma segura
        self.usar_aceleracion_magnitud = True
        self.usar_aceleracion_screening = True
        
        # Inicializar Puntos Primales
        if isinstance(K_ini_points, list):
            for col in K_ini_points:
                name = f'p_{self.cnt_points}' 
                self.K_points_dict[name] = col
                self.memoria_theta[name] = []
                self.cnt_points += 1
        elif isinstance(K_ini_points, dict):
            self.K_points_dict = K_ini_points.copy()
            self.cnt_points = len(self.K_points_dict)
            for k in self.K_points_dict: 
                self.memoria_theta[k] = []

        # Inicializar Rayos Primales
        if K_ini_rays:
            if isinstance(K_ini_rays, list):
                for col in K_ini_rays:
                    name = f'r_{self.cnt_rays}'
                    self.K_rays_dict[name] = col
                    self.memoria_mu[name] = []
                    self.cnt_rays += 1
            elif isinstance(K_ini_rays, dict):
                self.K_rays_dict = K_ini_rays.copy()
                self.cnt_rays = len(self.K_rays_dict)
                for k in self.K_rays_dict: 
                    self.memoria_mu[k] = []
                
    def limpiar_columnas(self, n_periodos):
        """ Remueve del problema maestro aquellas columnas inactivas bajo el umbral """
        # 1. Limpiar Puntos
        eliminar_p = []
        for nombre, hist in self.memoria_theta.items():
            if len(hist) >= n_periodos:
                recientes = hist[-n_periodos:]
                if all((val is not None and not np.isnan(val) and abs(val) < self.umbral_theta) for val in recientes):
                    eliminar_p.append(nombre)
        for nombre in eliminar_p:
            del self.K_points_dict[nombre]
            del self.memoria_theta[nombre]
            
        # 2. Limpiar Rayos
        eliminar_r = []
        for nombre, hist in self.memoria_mu.items():
            if len(hist) >= n_periodos:
                recientes = hist[-n_periodos:]
                if all((val is not None and not np.isnan(val) and abs(val) < self.umbral_theta) for val in recientes):
                    eliminar_r.append(nombre)
        for nombre in eliminar_r:
            del self.K_rays_dict[nombre]
            del self.memoria_mu[nombre]
            
        if eliminar_p or eliminar_r:
            print(f"🧹 Limpieza iter {self.i}: Eliminados {len(eliminar_p)} Puntos y {len(eliminar_r)} Rayos.")
            
    def iteracion_master(self):
        """ Ejecuta y desempaqueta el problema maestro cónico """
        list_points, keys_points = convertir_dict_a_K(self.K_points_dict)
        list_rays, keys_rays = convertir_dict_a_K(self.K_rays_dict)
        
        prev_obj = self.opt_val_fin[-1] if self.opt_val_fin else float('inf')
        
        # Ejecución del Master v3
        resultados = solve_master_primal_v3(
            self.X, self.y, 
            K=list_points, 
            tipo=self.tipo, 
            K_rayos=list_rays, 
            C=self.C,  
            mosek_params=self.master_mosek_params, 
            M_box=self.M_box,
            verbose=self.verbose,
            tijonov=self.tijonov
        )
        
        (theta_vals, eta, alpha, w_combo, obj_val, b, xi, grad_w, mu_vals) = resultados
        
        if obj_val is None:
            print("🛑 Error Crítico: El Master falló. Abortando algoritmo.")
            self.terminamos = True
            self.status = "MASTER_FAILURE"
            return self
        
        print(f"  >>> Master Obj: {obj_val:.7f} | Puntos: {len(list_points)} | Rayos: {len(list_rays)}")
        self.opt_val_fin.append(obj_val)
        self.alpha_set_fin.append(alpha)
        
        # ==============================================================
        # CRITERIO DE ACTUALIZACIÓN CON CONTROL DE SUAVIZADO PROXIMAL
        # ==============================================================
        # 1. Identificamos cuál es la dirección cruda que arrojó esta iteración
        nuevo_grad_w = alpha if self.gradient_strategy == "alpha_only" else grad_w

        # 2. Aplicamos la mezcla o la asignación directa según tus banderas de entrada
        if self.grad_w_actual is None:
            self.grad_w_actual = nuevo_grad_w
        else:
            if getattr(self, 'usar_suavizado_gradiente', False):
                # Estabilizador Proximal: Evita oscilaciones salvajes amortiguando la dirección
                self.grad_w_actual = self.smoothing_factor * self.grad_w_actual + (1.0 - self.smoothing_factor) * nuevo_grad_w
            else:
                # Flujo tradicional directo sin memoria histórica
                self.grad_w_actual = nuevo_grad_w

        # --- Guardar Historial Coeficientes de Puntos ---
        for k in keys_points: self.memoria_theta[k].append(0.0)
        if theta_vals is not None and len(theta_vals) > 0:
            for j, val in enumerate(theta_vals):
                if j < len(keys_points):
                    self.memoria_theta[keys_points[j]][-1] = val
        
        # --- Guardar Historial Coeficientes de Rayos ---
        for k in keys_rays: self.memoria_mu[k].append(0.0)
        if mu_vals is not None and len(mu_vals) > 0:
            for j, val in enumerate(mu_vals):
                if j < len(keys_rays):
                    self.memoria_mu[keys_rays[j]][-1] = val
                    
        if obj_val > prev_obj + self.tol:
            print(f"🚨 ALERTA: Master Obj subió de {prev_obj:.7f} a {obj_val:.7f} (Inestabilidad o WS fallido)")
            
        print("OBJ =", obj_val)
        print("||w|| =", np.linalg.norm(w_combo))
        print("sum_xi =", np.sum(xi))
        print("n_points =", len(list_points))
        print("n_rays =", len(list_rays))
        print("max_mu =", np.max(mu_vals) if len(mu_vals) else 0)


        
    def _generar_rayos_aceleracion(self, w_sparse, max_rayos=50):
        """
        ESTRATEGIA DE MAGNITUD RELATIVA:
        Genera rayos para las coordenadas con mayor valor absoluto en w,
        filtrando ruido y priorizando los 'drivers' principales del vector.
        """
        nuevos_rayos = []
        
        # 1. Chequeos de Seguridad
        if w_sparse is None: return []
        if self.X is None: return [] # Evita el AttributeError
        
        # Obtener dimensiones locales frescas
        n_samples_local, n_features_local = self.X.shape
        
        # Acceso eficiente a datos sparse
        indices = w_sparse.indices
        data = w_sparse.data
        print(1)
        if len(data) == 0: return []
        print(2)
        # 2. Determinar el Pico Máximo (Magnitud)
        max_abs_val = np.max(np.abs(data))
        
        # Si el vector es puro ruido (muy pequeño), no aceleramos
        if max_abs_val < 1e-7: #debe ser segun tol.  
            print("no se agrega NADA de rayos en esta aceleracion")
            return []

        # 3. Definir Umbral de Corte
        # Estrategia: Tomar cualquier coordenada que sea al menos el 50% del pico máximo.
        # Esto captura todos los coeficientes importantes, no solo el #1.
        umbral_relativo = 0.5 * max_abs_val 
        
        # 4. Filtrar Candidatos
        candidatos = []
        for idx, val in zip(indices, data):
            if abs(val) >= umbral_relativo:
                candidatos.append((abs(val), val, idx))
            
        # 5. Ordenar por Magnitud (De mayor a menor)
        # Priorizamos los coeficientes más grandes para el límite de max_rayos
        candidatos.sort(key=lambda x: x[0], reverse=True)
        
        # 6. Seleccionar Top-K y Generar
        seleccionados = candidatos[:max_rayos]
        
        for _, val, idx in seleccionados:
            # Generar rayo positivo o negativo según el signo de w
            if val > 0:
                rayo = generar_canonico_sparse(n_features_local, idx, n_samples_local)
            else:
                # Usamos el helper que permite signo negativo
                rayo = generar_canonico_con_signo_sparse(n_features_local, n_samples_local, idx, signo=-1.0)
            
            nuevos_rayos.append(rayo)
            
        return nuevos_rayos
    def _generar_rayos_costos_reducidos(self, max_rayos=50):
        """
        ESTRATEGIA B: SCREENING DUAL (Violación de Costos Reducidos)
        Retorna una LISTA de objetos sparse aislados para evitar corromper K_rays_dict.
        """
        if self.X is None or self.grad_w_actual is None: return []

        # Clonamos el diccionario actual para aislar la ejecución de la función base
        K_temp = self.K_rays_dict.copy()
        
        # Ejecutamos el screening en el contenedor temporal
        # Nota: Usamos self.grad_w_actual como el vector dual 'alpha' / gradiente según tu firma
        _, info = generar_set_columnas_costos_reducidos_sparse(
            self.X, self.y, self.grad_w_actual, K_temp, 
            self.n_features, self.n_samples, k_max=max_rayos
        )
        
        unique_idx = info.get('idx', [])
        unique_sign = info.get('sign', [])
        print('a')
        if len(unique_idx) == 0: return []
        print('b')
        # Generamos la lista de objetos sparse de forma pura
        nuevos_rayos_screening = [
            generar_canonico_con_signo_sparse(self.n_features, self.n_samples, int(j), float(s))
            for j, s in zip(unique_idx, unique_sign)
        ]
        
        print(f"  🚀 Screening Dual: {len(nuevos_rayos_screening)} candidatos identificados por violación KKT Y COSTOS REDUCIDOS.")
        return nuevos_rayos_screening
                
    def iteracion_pricing(self, max_rayos):
        # 1. Validación inicial
        if self.grad_w_actual is None:
            print("⚠️ Error: Gradiente nulo.")
            self.terminamos = True; return

        # 2. Llamada al Pricing (Pasando K_rays para el fallback)
        grad_w_scaled = self.grad_w_actual
        if np.linalg.norm(self.grad_w_actual) > 10:
            grad_w_scaled = self.grad_w_actual / np.linalg.norm(self.grad_w_actual)
        
        M_box_pricing = self.M_box_pricing
    
        res = self.pricing(
            self.X, 
            self.y, 
            grad_w_scaled, 
            self.K_rays_dict, 
            self.C, 
            self.pricing_gurobi_config,  # <--- CAMBIO: Mandamos el diccionario de Gurobi en vez de Mosek
            M_box_pricing,
            add_stabilization_cut=False, 
            warm_start=self.warm_start, 
            verbose=self.verbose
        )
        
        # ---------------------------------------------------------
        # CASO A: Pricing Exitoso (Procesa de forma inteligente Puntos vs Rayos)
        # ---------------------------------------------------------
        if isinstance(res, tuple) and len(res) == 4:
            w_sp, b_val, xi_sp, lb_val = res
            self.lb_fin.append(lb_val)
            
            if w_sp is None: 
                print("  [Pricing] No se generó columna."); return
            
            # ==============================================================
            # SUB-CASO A1: REALMENTE ES UN RAYO EXTREMO DE GUROBI (-inf)
            # ==============================================================
            if lb_val == -np.inf:
                # Lo registramos de forma correcta en el almacén de RAYOS
                name = f'r_{self.cnt_rays}'
                self.cnt_rays += 1
                self.memoria_mu[name] = [0.0] * (self.i + 1)
                
                # Estructura del rayo para tu maestro: (w_sparse, b_slope, xi_slope)
                self.K_rays_dict[name] = (w_sp, b_val, None)
                print(f"  🔥 [Rayo Real] Guardado correctamente como RAYO: {name}")
                
                # Dejamos w_original listo en formato denso/limpio para los cohetes de magnitud
                # Si w_sp es sparse de scipy, lo convertimos a array plano para que no falle .data
                if sp.issparse(w_sp):
                    w_para_acelerar = np.array(w_sp.toarray()).flatten()
                else:
                    w_para_acelerar = w_sp.flatten()

            # ==============================================================
            # SUB-CASO A2: ES UN PUNTO ACOBADO FACTIBLE (Solución Óptima)
            # ==============================================================
            else:
                # Normalización Vectorial (Solo aplica si el punto es Cónico)
                if self.tipo == "conico":
                    norm_w = np.linalg.norm(w_sp)
                    if norm_w > 1e-9: w_sp = w_sp / norm_w
                
                name = f'p_{self.cnt_points}'
                self.cnt_points += 1
                self.memoria_theta[name] = [0.0] * (self.i + 1) 
                self.K_points_dict[name] = (w_sp, b_val, xi_sp)
                print(f"  -> [Punto Real] Nuevo PUNTO acotado agregado: {name}")
                
                w_para_acelerar = w_sp.flatten()

            # ==============================================================
            # --- COHETES DE ACELERACIÓN POR MAGNITUD (Unificado y Seguro) ---
            # ==============================================================
            if self.pricing_acceleration:
                rayos_totales_aceleracion = []
                
                if getattr(self, 'usar_aceleracion_magnitud', True):
                    # Convertimos el vector limpio en el formato que '_generar_rayos_aceleracion' espera
                    w_sparse_input = sp.csc_matrix(w_para_acelerar.reshape(-1, 1), dtype=np.float32)
                    
                    rayos_magnitud = self._generar_rayos_aceleracion(w_sparse_input, max_rayos=max_rayos)
                    if rayos_magnitud:
                        rayos_totales_aceleracion.extend(rayos_magnitud)
                        print(f"  🚀 Cohetes activados: {len(rayos_magnitud)} rayos extraídos de la arista.")
                
                # La aceleración por Screening solo se dispara si el problema maestro está acotado
                if getattr(self, 'usar_aceleracion_screening', True): #and lb_val != -np.inf:
                    rayos_screening = self._generar_rayos_costos_reducidos(max_rayos=max_rayos)
                    if rayos_screening:
                        rayos_totales_aceleracion.extend(rayos_screening)
                        print(f"  🔎 Cohetes KKT activados: {len(rayos_screening)} aristas por Screening.")
                
                # Inyección unificada en la memoria cónica de la clase
                if rayos_totales_aceleracion:
                    for ray in rayos_totales_aceleracion:
                        r_name = f'r_{self.cnt_rays}'
                        self.cnt_rays += 1
                        self.K_rays_dict[r_name] = ray
                        self.memoria_mu[r_name] = [0.0] * (self.i + 1)
                    print(f"  🔥 BATERÍA MULTI-RAYO INYECTADA: {len(rayos_totales_aceleracion)} nuevas columnas al Maestro.")
                       
        # ---------------------------------------------------------
        # CASO B: Pricing de Gurobi no devolvió Solución (Fallo o Flag)
        # -> Activamos Aceleración/Fallback Formal por Screening Dual
        # ---------------------------------------------------------
        elif res is None or (isinstance(res, tuple) and len(res) == 2):
            print("  [Pricing] Gurobi no retornó una solución válida. Activando Screening de Emergencia...")
            
            # 1. Llamamos a tu función de screening de la forma limpia y desacoplada
            # Recuerda que esta función ahora solo genera los objetos esparsos sin mutar el diccionario global
            rayos_screening = self._generar_rayos_costos_reducidos(max_rayos=max_rayos)
            
            # Sub-caso B1: El Screening Dual encontró violaciones KKT válidas
            if rayos_screening:
                for rayo_sparse in rayos_screening:
                    r_name = f'r_{self.cnt_rays}'
                    self.cnt_rays += 1
                    self.K_rays_dict[r_name] = rayo_sparse
                    self.memoria_mu[r_name] = [0.0] * (self.i + 1)
                
                print(f"  🚀 Fallback Exitoso: {len(rayos_screening)} nuevos RAYOS agregados por Screening Dual.")
                self.lb_fin.append(-6742069) # Mantener GAP abierto

            # Sub-caso B2: El Screening Dual también falló (Ya se agotaron todas las características posibles)
            else:
                # ==============================================================
                # ESTRATEGIA DE EMERGENCIA FINAL: RAYO DE GRADIENTE 
                # ==============================================================
                print("  ⚠️ Fallback por Screening Dual AGOTADO (Sin columnas nuevas).")
                
                # 1. Obtener el gradiente actual
                gradiente = self.grad_w_actual.flatten()
                n_features_total = gradiente.shape[0]
                
                # 2. Configuración de Velocidad DINÁMICA
                n_top = min(20000, n_features_total) 
                
                # 3. Identificar índices más importantes
                if n_top == n_features_total:
                    idx_top = np.arange(n_features_total)
                else:
                    idx_top = np.argpartition(np.abs(gradiente), -n_top)[-n_top:]
                
                # 4. Extraer valores (Signo negativo para descenso)
                vals_top = -gradiente[idx_top]
                
                # 5. Construcción SPARSE Consistente en formato columna (n_features, 1)
                data = vals_top
                indices = idx_top
                indptr = np.array([0, len(data)]) 
                
                rayo_grad_sparse = sp.csc_matrix(
                    (data, indices, indptr), 
                    shape=(n_features_total, 1), 
                    dtype=np.float32
                )
                
                # 6. Inserción en la memoria del problema maestro
                r_name = f'r_grad_{self.i}' 
                self.cnt_rays += 1
                
                # Para mantener la homogeneidad con tu maestro, si almacena tuplas de 3 o matrices puras:
                # Guardamos el rayo esparso con sesgo 0.0 igual que en tu emergencia original
                self.K_rays_dict[r_name] = (rayo_grad_sparse, 0.0, None)
                self.memoria_mu[r_name] = [0.0] * (self.i + 1)
                
                print(f"  🔥 ACTIVANDO EMERGENCIA ABSOLUTA: Rayo de Gradiente agregado ({n_top} nnz).")
                self.lb_fin.append(-6742069)


        else:
            print("  [Pricing] Formato de retorno desconocido.")
            self.lb_fin.append(-6742069)
            
    def actualizar_parametros_solver(self):
        """
        Ajusta dinámicamente las tolerancias y configuraciones de Mosek (Master) 
        y Gurobi (Pricing) basándose en la mejora de la función objetivo del Master.
        Objetivo: Ir de 'Rápido y Laxo' a 'Lento y Preciso'.
        """
        # Necesitamos al menos 2 iteraciones para comparar
        if len(self.opt_val_fin) < 2:
            return

        # Calcular mejora absoluta (descenso)
        curr_obj = self.opt_val_fin[-1]
        prev_obj = self.opt_val_fin[-2]
        diff = abs(prev_obj - curr_obj)
        
        # Inicializamos el diccionario base para el Pricing de Gurobi
        self.pricing_gurobi_config = {
            "InfUnbdInfo": 1,
            "Presolve": 0
        }
        
        # Estado 1: Cambios grandes -> Priorizar Velocidad
        if diff > 1e-1:
            new_tol = 1e-5
            mode_name = "VELOCIDAD (Coarse)"
            VER = False
            WS = True
            self.n_periodos = getattr(self, 'n_periodos_base', self.n_periodos) #jules hizo un fix a la recursion. 
            self.pricing_gurobi_config["FeasibilityTol"] = 1e-4
            
        # Estado 2: Cambios medios -> Precisión Estándar
        elif diff > 1e-3:
            new_tol = 1e-6
            mode_name = "ESTÁNDAR (Medium)"
            VER = True
            WS = True
            self.n_periodos = int(getattr(self, 'n_periodos_base', self.n_periodos) * 1.1) #jules hizo un fix a la recursion. 
            self.pricing_gurobi_config["FeasibilityTol"] = 1e-6

        # Estado 3: Cambios finos -> Precisión Máxima
        else:
            new_tol = self.master_mosek_params.get("MSK_DPAR_INTPNT_TOL_REL_GAP", 1e-8) 
            if new_tol > 1e-6: 
                new_tol = 1e-7 
            
            mode_name = "PRECISIÓN (Fine-Tuning)"
            self.M_box = None
            
            # CORRECCIÓN AQUÍ: Deja esto en None. Gurobi matricial NO necesita caja
            # para el pricing porque ya sabe extraer el UnbdRay de forma limpia.
            self.M_box_pricing = None 
            
            VER = True
            WS = False
            self.n_periodos = int(getattr(self, 'n_periodos_base', self.n_periodos) * 1.5) #jules hizo un fix a la recursion. 
            self.pricing_gurobi_config["FeasibilityTol"] = 1e-9
            self.pricing_gurobi_config["NumericFocus"] = 3
            
        print("*"*10, mode_name, "*"*10)
            
        # --- APLICACIÓN COMPLETA DE PARÁMETROS ---
        
        # 1. Actualizamos el Master de Mosek con la tolerancia calculada
        self.master_mosek_params = mosek_params_from_tol(new_tol)
        
        # 2. Seteamos las variables de control de la iteración de la clase
        self.verbose = VER
        self.warm_start = WS


    def run(self, max_iter, n_periodos=5, frecuencia_check=5, max_rayos=50):
        self.i = 0
        self.terminamos = False
        time_ini = time.time()
        self.n_periodos_base = n_periodos #jules hizo un fix a la recursion. 
        self.n_periodos = n_periodos
        
        while not self.terminamos and self.i < max_iter:
            self.actualizar_parametros_solver() # Ajusta Master (Mosek) y Pricing (Gurobi)
            print(f"\n=== Iteración {self.i} ===")
            
            # 1. Limpieza Periódica de Columnas Obsoletas
            if self.i > 0 and self.i % frecuencia_check == 0:
                n_periodos_actual = self.n_periodos
                self.limpiar_columnas(n_periodos_actual)
                
            # 2. Resolver Problema Master (Mosek)
            t_start = time.time()
            self.iteracion_master()
            t_master = time.time() - t_start
            self.time_master.append(t_master)
            
            # 3. Resolver Pricing (Genera Puntos o Rayos mediante Gurobi Matricial / Screening)
            t_start_p = time.time()
            self.iteracion_pricing(max_rayos)
            t_pricing = time.time() - t_start_p
            self.time_pricing.append(t_pricing)
            
            # 4. Chequeo de Convergencia Final de Iteración (Manejo seguro de -inf)
            if len(self.opt_val_fin) > 0 and len(self.lb_fin) > 0:
                ub_actual = self.opt_val_fin[-1]
                lb_actual = self.lb_fin[-1]
                
                # CORRECCIÓN DE FRONTERA: Si el pricing devolvió un rayo (-inf), el GAP matemático
                # es infinito. No calculamos la resta directa para evitar desbordamientos.
                if lb_actual == -np.inf or lb_actual == -6742069:
                    print("  [Loop] El subproblema arrojó un RAYO extremo. El GAP se mantiene abierto para pivotar.")
                else:
                    gap = ub_actual - lb_actual
                    print(f"  [Loop] GAP MASTER-PRICING: {abs(gap):.4f}")
                    
                    # Criterio estricto de parada por convergencia del Gap
                    if abs(gap) < self.tol:
                        print(f"✅ ¡Convergencia alcanzada con éxito por GAP!: {gap:.2e}")
                        self.status = "optimo_gap"
                        self.terminamos = True
            
            # 5. Chequeo de Estancamiento Inteligente (Sólo si el Pricing está acotado)
            if self.i > 5 and not self.terminamos:
                mejora = abs(self.opt_val_fin[-2] - self.opt_val_fin[-1])
                lb_actual = self.lb_fin[-1]
                
                # REGLA DE ORO: No declares estancamiento si el último pricing fue un rayo (-inf),
                # ya que el problema maestro está obligado a cambiar de base en la siguiente iteración.
                if mejora < self.tol and lb_actual != -np.inf and lb_actual != -6742069:
                    print(f"⚠️ Convergencia por estancamiento del Upper Bound (Diff: {mejora:.2e})")
                    self.status = "estancamiento"
                    self.terminamos = True

            self.i += 1
        
        if not self.terminamos:
            self.status = "max_iter"
            print("🛑 Máximo de iteraciones alcanzado sin cerrar el GAP.")

        print(f"⏱ Tiempo Total del Proceso: {(time.time() - time_ini)/60:.2f} min")
        
        # Retornar estructura consolidada para análisis posterior en la tesis
        return {
            "K_points": self.K_points_dict,
            "K_rays": self.K_rays_dict,
            "opt_vals": self.opt_val_fin,
            "status": self.status
        }


# In[5]:
print("\n[1/4] Configurando GeneracionColumnasDW...")

# =============================================================================
# SCRIPT DE EJECUCIÓN: Configuración de la Clase DW
# =============================================================================

n_samples, n_features = X_train.shape

# 1. Configuración de Hiperparámetros
C_param = 1
M_box = None
tol = 1e-6
tol_master = 1e-6
max_iter_gc = 450        # Máximo de iteraciones del algoritmo GC

n_periodos = 20
frecuencia_check = 40
umbral_theta = 1e-6     # Se configura internamente en la clase, o puedes redefinirlo tras instanciar
max_rayos = 500          # Cantidad de rayos a extraer en la aceleración por lote

# 2. Resolución de Control Inicial (Paso Cero para validar K_ini)
results_ini = solve_master_primal_v3(
    X_train, y_train, K_ini_loaded, 'convexo', #sospecho que deberia ser este convexo. 
    K_rayos=K_ini_canonico_loaded, 
    C=C_param, mosek_params={}, 
    M_box=M_box, warm_start=True, verbose=True, tijonov=False
)

print(f"Iniciando proceso para prueba: {n_samples} muestras x {n_features} features")
print('El valor óptimo actual del problema maestro con combinación afín y k_ini es: ', np.linalg.norm(results_ini[4]))

# 3. Instanciar la Nueva Clase Optimizada
# El constructor ahora es limpio y enfocado exclusivamente en las tolerancias y la regularización del Master
gen_col = GeneracionColumnasDW_2(
    tol=tol,
    tol_master=tol_master,
    M_box=M_box,
    threads=None  # Puedes pasar un entero si deseas limitar los cores de Mosek
)

# Ajustar el umbral si difiere del valor por defecto (1e-6)
gen_col.umbral_theta = umbral_theta

# 4. Ingresar Datos de Entrenamiento
gen_col.ingresar_data(X_train, y_train)

# 5. Ingresar Parámetros de Operación (Asignación del nuevo Pricing Gurobi y Aceleradores)
gen_col.ingresar_parametros(
    C=C_param,
    M=M_box,
    K_ini_points=K_ini_loaded,           # Input de PUNTOS (se etiquetarán p_X)
    K_ini_rays=K_ini_canonico_loaded,     # Input de RAYOS (se etiquetarán r_X)
    tipo="convexo",                       # Condición de convexidad para los puntos theta
    pricing=pricing_gurobi,               # Tu nuevo motor Core API matricial
    gradient_strategy="full_gradient",    # Estrategia basada en el gradiente funcional completo #"alpha_only"
    pricing_acceleration=True,           # Cambiar a True si deseas disparar Magnitud + Screening juntos
    tijonov=False,                         # Regularización de Tikhonov desactivada para paso base

    # --- NUEVOS INPUTS EXPERIMENTALES ---
    usar_suavizado_gradiente=True,    # True = Estabiliza el Pricing / False = Clásico
    smoothing_factor=0.4,             # Peso del gradiente histórico en la mezcla
    reportar_rayo_como_infinito=True  # True = Fuerza -inf en el loop / False = Usa costo real de arista
)

# Configurar el estado de las estrategias de aceleración independientes si activas pricing_acceleration=True
gen_col.usar_aceleracion_magnitud = True
gen_col.usar_aceleracion_screening = True

# =============================================================================
# SCRIPT DE EJECUCIÓN FINAL Y GENERACIÓN DE GRÁFICAS
# =============================================================================
print("\n[4/4] Lanzando algoritmo de Generación de Columnas DW...")

# 1. Ejecutar el bucle principal del algoritmo
# Configuramos el número máximo de iteraciones y los parámetros del lote de aceleración
historico_resultados = gen_col.run(
    max_iter=max_iter_gc,
    n_periodos=n_periodos,
    frecuencia_check=frecuencia_check,
    max_rayos=max_rayos
)

print("\n==================================================")
print("         RESUMEN DE CONVERGENCIA DE LA TESIS       ")
print("==================================================")
print(f"-> Estado de finalización: {historico_resultados['status']}")
print(f"-> Iteraciones totales ejecutadas: {gen_col.i}")
print(f"-> Total de Puntos acumulados en K: {len(historico_resultados['K_points'])}")
print(f"-> Total de Rayos acumulados en K:  {len(historico_resultados['K_rays'])}")

# In[6]:
#opt_teorico_skl=skl_svm_res[1]
# 2. Extracción de Métricas para Graficar mediante Módulo Consolidado
from visualizaciones_tesis import ejecutar_pipeline_graficacion_master

# Invocar pasando tu instancia activa 'gen_col'
ejecutar_pipeline_graficacion_master(
    gen_col=gen_col,
    n_samples=n_samples,
    n_features=n_features,
    opt_teorico_con=opt_teorico_con) # Tu variable con el óptimo de referencia