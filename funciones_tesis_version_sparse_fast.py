# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 13:14:55 2025

@author: matta
"""

# -*- coding: utf-8 -*-
"""
VERSIÓN FINAL CONSOLIDADA (v13+)
- Manejo de SPARSAS (K_ini, K_gen, Master)
- Cálculo de GAP (Master retorna UB/LB)
- Normalización de Columnas (Pricing)
- Chequeo de Hull
- Paso de Parámetros MOSEK
- Estrategia de Gradiente
"""
import gc
import cvxpy as cp
import numpy as np
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
    prob.solve(solver=cp.MOSEK, mosek_params={
               "MSK_DPAR_OPTIMIZER_MAX_TIME": float(time_limit_sec)}, warm_start=True)
    # ... (prints omitidos por brevedad) ...
    #print(prob.value)
    if solo_w_b_xi == True:
        return (w.value, b.value, xi.value)
    else:
        return [(w.value, b.value, xi.value),prob.value]
    
def predict_t(b, w, x_test):
    # Asegurarse que w sea 1D denso para .dot
    if sp.issparse(w):
        w_dense = w.toarray().flatten()
    else:
        w_dense = np.asarray(w).flatten()
    
    # x_test es sparse, w_dense es 1D. .dot es correcto.
    estimate = x_test.dot(w_dense) + b
    prediction = np.sign(estimate)
    return prediction

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# --- FUNCIONES DE MANEJO DE K (SPARSE) ---

def K_ini_version_dict(K_lista):
    """Convierte una lista de tuplas (w,b,xi) en un dict {'k_i': (w,b,xi)}"""
    return {f'k_{i}': col for i, col in enumerate(K_lista)}

def convertir_dict_a_K(K_dict):
    """Convierte un dict de columnas en una lista de tuplas (w,b,xi) y una lista de claves"""
    return list(K_dict.values()), list(K_dict.keys())

def compactar_K_a_sparse(K, keep_xi=True, dtype=np.float32):
    """Toma una lista K de tuplas (w_dense, b, xi_dense) y la convierte a (w_sparse, b, xi_sparse)"""
    K_sparse = []
    if not K: return K_sparse
    n_features, n_samples = 0, 0
    
    # Determinar n_features
    for col in K:
        if col[0] is not None: n_features = len(col[0]); break
    if n_features == 0: raise ValueError("No se pudo determinar n_features desde el conjunto K.")
    
    # Determinar n_samples
    if keep_xi:
        for col in K:
            if col[2] is not None: n_samples = len(col[2]); break
        if n_samples == 0: print("Advertencia: No se pudo determinar n_samples para xi.")

    for (w, b, xi) in K:
        # Convertir w
        if w is None:
            w_sparse = sp.csc_matrix((n_features, 1), dtype=dtype)
        else:
            w_dense_col = np.asarray(w, dtype=dtype).reshape(-1, 1)
            if w_dense_col.shape[0] != n_features: raise ValueError(f"Inconsistencia de shape w")
            w_sparse = sp.csc_matrix(w_dense_col)
            w_sparse.eliminate_zeros()
            
        b_float = float(b) if b is not None else 0.0
        
        # Convertir xi
        if keep_xi:
            if xi is None:
                xi_sparse = sp.csc_matrix((n_samples, 1), dtype=dtype) if n_samples > 0 else None
            else:
                xi_dense_col = np.asarray(xi, dtype=dtype).reshape(-1, 1)
                if n_samples == 0: n_samples = xi_dense_col.shape[0]
                if xi_dense_col.shape[0] != n_samples: raise ValueError(f"Inconsistencia de shape xi")
                xi_sparse = sp.csc_matrix(xi_dense_col)
                xi_sparse.eliminate_zeros()
        else:
            xi_sparse = None
            
        K_sparse.append((w_sparse, b_float, xi_sparse))
    return K_sparse

def crear_sub_problema(X_train, y_train, partes=101, time_max=60, tol=1e-06, keep_xi=True):
    K = [] # Contendrá tuplas con w densos
    column_sets = []  
    log_info = []
    n_cols = X_train.shape[1]
    
    for i in range(1, partes):
        gc.collect()
        frac = i / (partes - 1)
        sub_frac = (i - 1) / (partes - 1)
        start = int(n_cols * sub_frac); end = int(n_cols * frac)    
        if start >= end: continue # Evitar slices vacíos
        #selected_indices = np.arange(start, end)
        XX = X_train[:, start:end] # Cortar X esparso
        
        try:
            w_sub, b, xi = solve_svm_conic(XX, y_train, C=1.0, time_limit_sec=time_max)
            w_full = np.zeros(n_cols, dtype=np.float32) 
            w_sub_limp = w_sub.astype(np.float32, copy=False)
            w_full[start:end] = w_sub_limp
            K.append((w_full, float(b), xi)) 
            # ... (código para log_info) ...
        except Exception as e:
            print(f"[i={i}] Error al resolver SVM en columnas {start}:{end} -> {e}")
            # ... (código para log_info de error) ...
            
    df_log = pd.DataFrame(log_info)
    
    print(f"Compactando {len(K)} columnas de subproblemas a formato sparse...")
    K_sparse_list = compactar_K_a_sparse(K, keep_xi=keep_xi)
    print("Compactación finalizada.")
    
    return K_sparse_list, column_sets, df_log

def crear_sub_problema_random(X_train, y_train, partes=101, time_max=60, 
                              tol=1e-06, keep_xi=True, random_state=None,solapar=False):
    """
    Resuelve SVM en sub-problemas, usando subconjuntos de 
    características aleatorias (sin solapamiento).
    """
    
    K = [] # Contendrá tuplas con w densos
    column_sets = [] # Almacenará los índices de columnas usados por cada 'parte'
    log_info = []
    n_cols = X_train.shape[1]

    # --- INICIO DE LA MODIFICACIÓN ---
    
    # 1. Crear y barajar los índices de las columnas
    
    
    print(f"Barajando {n_cols} índices de columnas (Random State={random_state})...")
    
    indices = np.arange(n_cols)
    # Usar un Generador de Números Aleatorios (RNG) para reproducibilidad
    rng = np.random.default_rng(random_state) 
    rng.shuffle(indices) # Baraja 'indices' in-place
    
    # --- FIN DE LA MODIFICACIÓN ---

    print(f"Iniciando procesamiento de {partes-1} partes...")

    
    for i in range(1, partes):

        if solapar==True:
            chunk_size = int(n_cols / (partes - 1))
            if chunk_size == 0: continue
            gc.collect()
            selected_indices = rng.choice(n_cols, size=chunk_size, replace=False)
            selected_indices_sorted = np.sort(selected_indices)
            column_sets.append(selected_indices_sorted)
            XX = X_train[:, selected_indices_sorted]
            
            
        else:
            gc.collect()
            frac = i / (partes - 1)
            sub_frac = (i - 1) / (partes - 1)
            
            # Mismo slicing, pero sobre el array de índices barajados
            start = int(n_cols * sub_frac)
            end = int(n_cols * frac)
            
            if start >= end: continue
    
            # --- MODIFICACIÓN CLAVE ---
            
            # 2. Seleccionar el subconjunto de índices aleatorios
            selected_indices = indices[start:end] 
            
            # 3. [OPTIMIZACIÓN] Ordenar los índices seleccionados.
            # El slicing de matrices dispersas (CSR/CSC) por columnas
            # es SIGNIFICATIVAMENTE más rápido si los índices de 
            # las columnas están ordenados.
            selected_indices_sorted = np.sort(selected_indices)
            
            column_sets.append(selected_indices_sorted) # Guardar los índices usados
    
            # 4. Realizar el slicing de la matriz dispersa
            # Esto selecciona columnas no contiguas de X_train
            XX = X_train[:, selected_indices_sorted] 

        try:
            # Resolver el sub-problema solo con estas columnas
            w_sub, b, xi = solve_svm_conic(XX, y_train, C=1.0, time_limit_sec=time_max)
            
            # Reconstruir el vector 'w' en el espacio original
            w_full = np.zeros(n_cols, dtype=np.float32)
            w_sub_limp = w_sub.astype(np.float32, copy=False)
            
            # --- MODIFICACIÓN DE RECONSTRUCCIÓN ---
            # Asignar los pesos de vuelta a sus posiciones originales
            # (usando los índices ordenados, que coinciden con el 'w_sub')
            w_full[selected_indices_sorted] = w_sub_limp
            # --- FIN MODIFICACIÓN ---
            
            K.append((w_full, float(b), xi))
            
            log_info.append({
                'part': i,
                'status': 'success',
                'n_features': len(selected_indices_sorted),
                'bias': float(b),
                'xi_sum': xi.sum()
                # ... puedes añadir más logs, ej. 'xi_sum': xi.sum()
            })
            
        except Exception as e:
            print(f"[i={i}] Error al resolver SVM en {len(selected_indices_sorted)} columnas aleatorias -> {e}")
            log_info.append({
                'part': i,
                'status': 'error',
                'n_features': len(selected_indices_sorted),
                'error_msg': str(e)
            })
    
    df_log = pd.DataFrame(log_info)
    
    print(f"Compactando {len(K)} columnas de subproblemas a formato sparse...")
    # 'compactar_K_a_sparse' debe funcionar igual, ya que recibe una 
    # lista de 'w_full' densos (pero que internamente son esparsos)
    K_sparse_list = compactar_K_a_sparse(K, keep_xi=keep_xi)
    print("Compactación finalizada.")
    
    return K_sparse_list, column_sets, df_log



# --- HELPERS PARA CANÓNICOS (SPARSE) ---

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

def generar_canonicos_filtro(indices_features, n_features, n_samples):
    """
    Genera una lista de vectores canónicos usando la función base existente.
    """
    nuevos_rayos = []
    for idx in indices_features:
        # Reutilizamos tu lógica existente
        columna_canonica = generar_canonico_sparse(
            n_features=n_features, 
            coordenada=idx, 
            n_samples=n_samples
        )
        nuevos_rayos.append(columna_canonica)
        
    return nuevos_rayos

def generar_canonico_con_signo_sparse(n_features, n_samples, j, signo=+1.0):
    (w_sparse, b_tag, xi_sparse) = generar_canonico_sparse(
        n_features=n_features, coordenada=int(j), n_samples=n_samples, dtype=np.float32)
    if w_sparse.nnz > 0:
        w_sparse.data[0] = float(np.sign(signo))
    return (w_sparse, b_tag, xi_sparse)

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

# --- HELPER DE CHEQUEO DE HULL (CORREGIDO) ---

def is_column_in_hull(w_new_sparse, K_list_sparse_w, tipo="convexo", tol=1e-6):
    K_len = len(K_list_sparse_w)
    if K_len == 0 or w_new_sparse is None or w_new_sparse.nnz == 0:
        return False 
    try:
        Ws_sparse = sp.hstack(K_list_sparse_w, format='csc')
    except ValueError as e:
        print(f"Error de forma en is_column_in_hull (sp.hstack): {e}")
        return False 
    
    theta = cp.Variable((K_len, 1)) # --- Theta como (K, 1) ---
    w_new_dense_col = w_new_sparse.toarray() # --- w_new como (N, 1) ---

    if w_new_dense_col.shape[0] != Ws_sparse.shape[0]:
         raise ValueError(f"Incompatibilidad de shape: w_new tiene {w_new_dense_col.shape[0]} filas, K tiene {Ws_sparse.shape[0]}")

    constraints = [Ws_sparse @ theta == w_new_dense_col] # (N, 1) == (N, 1)

    if tipo == "convexo": constraints += [cp.sum(theta) == 1, theta >= 0]
    elif tipo == "afin": constraints += [cp.sum(theta) == 1]
    elif tipo == "conico": constraints += [theta >= 0]

    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.MOSEK, mosek_params={"MSK_DPAR_INTPNT_TOL_PFEAS": tol * 10, 
                                            "MSK_DPAR_INTPNT_TOL_DFEAS": tol * 10})
    
    if prob.status == cp.OPTIMAL:
        print("Chequeo de columna: Columna SÍ está en el hull.")
        return True
    else:
        print("Chequeo de columna: Columna NO está en el hull.")
        return False

# --- HELPERS DE PRICING (SCREENING/FALLBACK) ---

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

# --- FUNCIÓN DE PARÁMETROS MOSEK (ACTUALIZADA) ---

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

# --- PROBLEMA MAESTRO (CONSOLIDADO Y ROBUSTO) ---

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
                           M_box=1e4,warm_start=True, verbose=True,tijonov=True):
    
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
        prob.solve(solver=cp.MOSEK, mosek_params=mosek_params, warm_start=warm_start, verbose=verbose) #el warm start a veces cagonea cuando la solucion numericamente cambia poco en el lategame. ayuda harto en el early eso si.  
    except cp.error.SolverError:
        print("⚠️ Master CRASH con params estrictos/warm_start. Reintentando relajado...")
        try:
            # Intento 2: Sin warm_start y tolerancias relajadas
            params_relaxed = mosek_params.copy()
            params_relaxed["MSK_DPAR_INTPNT_TOL_REL_GAP"] = 1e-5
            params_relaxed["MSK_DPAR_INTPNT_TOL_PFEAS"] = 1e-5
            params_relaxed["MSK_DPAR_INTPNT_TOL_DFEAS"] = 1e-5
            
            prob.solve(solver=cp.MOSEK, mosek_params=params_relaxed, 
                       warm_start=True, verbose=True)
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

def solve_master_primal_v2(X, y, K, tipo, C=1.0, tol=1e-6, mosek_params={}, M_box=1e4, verbose=False):
    y_neg = np.where(y <= 0, -1, 1)
    K_len = len(K)
    n_samples = len(y)
    n_features = X.shape[1] 

    # --- Variables ---
    if tipo in ["afin", "libre"]:
        theta = cp.Variable((K_len, 1))
    elif tipo in ["convexo", "conico", "mayor_uno"]:
        theta = cp.Variable((K_len, 1), nonneg=True)
    else:
        print(f"Error: tipo '{tipo}' no es válido.")
        return None, None, None, None, None, None, None, None, None

    eta = cp.Variable()
    b = cp.Variable()
    xi = cp.Variable((n_samples, 1), nonneg=True) 

    # --- Precomputar Ws_sparse (Robusto) ---
    sparse_Ws_list = []
    for k_idx, col in enumerate(K):
        w = col[0]
        if sp.issparse(w):
            if w.shape == (n_features, 1): sparse_Ws_list.append(w.astype(np.float32))
            elif w.shape == (1, n_features): sparse_Ws_list.append(w.T.astype(np.float32))
            else: raise ValueError(f"Columna esparsa {k_idx} tiene shape {w.shape}.")
        elif isinstance(w, np.ndarray):
            w_col = np.asarray(w, dtype=np.float32).reshape(-1, 1)
            if w_col.shape[0] != n_features: raise ValueError(f"Columna {k_idx} tiene {w_col.shape[0]} filas.")
            sparse_Ws_list.append(sp.csc_matrix(w_col))
        else: raise TypeError(f"Elemento w en K (índice {k_idx}) es tipo {type(w)}")
    
    Ws_sparse = sp.hstack(sparse_Ws_list, format='csc')
    w_combo = Ws_sparse @ theta   
    
    # --- Restricciones ---
    y_neg_col = y_neg.reshape(-1, 1) # Asegurar que y sea (M, 1)
    constraints_dict = {
        "soc_norm": cp.SOC(eta, w_combo),
        "classification": cp.multiply(y_neg_col, X @ w_combo + b) >= 1 - xi, 
        "master_box_pos": w_combo <= M_box, 
        "master_box_neg": w_combo >= -M_box
    }
    if tipo in ["convexo", "afin"]: constraints_dict["theta_sum"] = (cp.sum(theta) == 1)
    elif tipo == "mayor_uno": constraints_dict["theta_sum"] = (cp.sum(theta) >= 1)
    elif tipo == "conico": constraints_dict["theta_sum"] = (cp.sum(theta) >= 0)
    
    objective = cp.Minimize(eta + C * (cp.sum(xi)))
    prob = cp.Problem(objective, list(constraints_dict.values()))
    prob.solve(solver=cp.MOSEK, mosek_params=mosek_params, warm_start=True, verbose=verbose) 
    
    # --- CÁLCULO DE GAP Y GRADIENTE ---
    primal_value_UB = prob.value
    
#    dual_value_LB = prob.dual_value
#    print(f"Master UB (Primal Obj): {primal_value_UB} / Master LB (Dual Obj): {dual_value_LB}")
#    if primal_value_UB is not None and dual_value_LB is not None:
#        print(f"GAP ACTUAL (UB - LB): {primal_value_UB - dual_value_LB}")

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
            else: print("ADVERTENCIA: No se pudieron obtener las variables duales (valores None).")
        except Exception as e:
            print(f"ADVERTENCIA: Error al calcular el gradiente: {e}"); alpha = None
    else: print(f"ADVERTENCIA: Master no resolvió óptimamente (status: {prob.status}).")

    print(f"Norma del gradiente correcto para w: {np.linalg.norm(grad_w_correcto):.4f}")
    print("la norma de w obtenido es de ", np.linalg.norm(w_combo.value))

    # --- ORDEN DE RETORNO (9 VALORES) ---
    theta_val = theta.value.flatten() if theta.value is not None else None
    w_combo_val = w_combo.value.flatten() if w_combo.value is not None else None
    b_val = b.value.item() if b.value is not None else None
    xi_val = xi.value.flatten() if xi.value is not None else None
    alpha_val = alpha.flatten() if alpha is not None else None

    return (theta_val, eta.value, alpha_val, w_combo_val, primal_value_UB, 
            b_val, xi_val, grad_w_correcto)

# --- FUNCION PRE PROCESAMIENTO DE SET K INICIAL  ---


def K_forest(X_train,y_train,n_iters, partes=101, time_max=60, tol=1e-06, keep_xi=False,solapar=True,random=1): #random es para ir cambiando la seed. pues es i * random 
    
    ''' La idea detras es que divides el espacio de features en las partes asociadas de tal manera que el problema conico se resuelve en menos de 1 minuto (time_max)
    Despues de eso viene la parte clave, la idea de crear_sub_problema_random es que elige ciertas features al azar para cada uno de los sub sets hecho por partes, y luego resuelve el K final con el master
    Ese master simplemente debe devolver las soluciones que son mas grandes que la tolerancia elegida. Si eliges solapar=True, se pueden repetir columnas en los sub sets aleatorios.
    Si es false, lo divide en partes iguales pero aleatoriza quien va donde. EL RANDOM FOREST CLASICO UTILIZA SOLAPAR = TRUE.'''
    
    n_samples, n_features = X_train.shape 
    cero= generar_canonico_sparse(n_features, coordenada=-1, n_samples=n_samples) #el cero pa ayudar al convexo
    K_ini_end=[cero]
    for i in range(n_iters):
        try:
            K_list,_ , df_log = crear_sub_problema_random(X_train, y_train, partes=partes, time_max=time_max, tol=tol, keep_xi=keep_xi, random_state=i*int(random),solapar=solapar)
            # 2. VALIDACIÓN (PROTECCIÓN CONTRA K_list VACÍO)
            if not K_list:
                print(f"[Iter {i}] Advertencia: No se generaron columnas. Saltando al siguiente estado.")
                continue # Saltar a la siguiente iteración del bucle
            master_results = solve_master_primal_v2(
                X_train, y_train, K_list, 
                tipo="convexo", # Specify the combination 
                C=1.0, 
                tol=1e-6,
                mosek_params=mosek_params_from_tol(1e-6, presolve_level=1, optimizer_code=0) # Use default solver settings
            )
            
            theta= master_results[0]
            master_obj = master_results[4] # Get the objective value 
            print(f"Master Objective: {master_obj:.4f}")
            gc.collect()
            active_indices = np.where(np.abs(theta) > tol)[0]
            n_activos = len(active_indices)
            print(f"[Iter {i}] Columnas activas encontradas: {n_activos}")
            if n_activos == 0:
                print(f"[Iter {i}] Ninguna columna activa, nada que añadir al pool.")
                del master_results, K_list, theta # Limpieza
                gc.collect()
                continue # Saltar a la siguiente iteración
        # Crear la lista de columnas limpias
            K_list_cleaned = [(K_list[idx][0], K_list[idx][1], None) for idx in active_indices]
        
            # Usar .extend() para modificar la lista 'in-place'
            K_ini_end.extend(K_list_cleaned)
            del master_results, K_list, theta, active_indices, K_list_cleaned
            gc.collect() 
        except Exception as e:
            # 7. CAPTURA DE ERRORES
            # Si el solver *realmente* falla, sabremos exactamente en qué iteración 'i'
            print(f"!!!!!!!! ERROR FATAL EN ITERACIÓN {i} (Random State={i}) !!!!!!!!")
            print(f"Error: {e}")
            # Dependiendo de la gravedad, puedes 'break' o 'continue'
            # 'continue' intentará la siguiente iteración
            continue 

    print("\n--- Construcción del Pool Finalizada ---")
    print(f"Columnas totales recolectadas en K_ini_end: {len(K_ini_end)}") 
    return K_ini_end #este set es el filtrado despues de haber hecho n_iter combinaciones distintas de las features para construir el problema. 

#data set de 1000 x 1000 se demora aproximadamente por iteracion 10 segundos con 101 partes. entonces por hay que poner a prueba la velocidad de esto. 

def K_forest_2(X_train, y_train, n_iters, partes=101, time_max=60, tol=1e-06, 
             keep_xi=False, solapar=True, random=1, fixed_sample_size=None):
    
    ''' 
    Versión mejorada con Subsampling Estratificado (Estilo Random Forest).
    
    Args:
        fixed_sample_size (int): Número de filas a utilizar en cada sub-problema. 
                                 Si es None, usa todo el dataset.
    '''
    
    n_samples, n_features = X_train.shape 
    
    # Vector cero inicial para factibilidad del Master
    cero = generar_canonico_sparse(n_features, coordenada=-1, n_samples=n_samples) 
    K_ini_end = [cero]
    
    # Calcular porcentaje si se define un tamaño fijo
    train_size_param = None
    if fixed_sample_size is not None and fixed_sample_size < n_samples:
        train_size_param = fixed_sample_size
        ratio = fixed_sample_size / n_samples
        print(f"🌲 K_forest Config: Subsampling activado. Usando {fixed_sample_size} filas ({ratio:.2%}) por árbol.")
    else:
        print(f"🌲 K_forest Config: Usando Full Batch ({n_samples} filas).")

    for i in range(n_iters):
        try:
            current_seed = i * int(random)
            
            # --- 1. MUESTREO DE DATOS (BAGGIN STRATIFIED) ---
            if train_size_param is not None:
                # Generar sub-dataset estratificado
                # X_sub, y_sub tendrán el tamaño reducido, manteniendo distribución de clases
                X_sub, _, y_sub, _ = train_test_split(
                    X_train, y_train, 
                    train_size=train_size_param, 
                    stratify=y_train, 
                    random_state=current_seed
                )
                # Si usamos subsample, el xi local no sirve para el master global
                keep_xi_iter = False 
            else:
                # Usar dataset completo
                X_sub, y_sub = X_train, y_train
                keep_xi_iter = keep_xi

            # --- 2. GENERACIÓN DE COLUMNAS (SUB-PROBLEMA) ---
            # Se entrena con X_sub, y_sub (más rápido)
            K_list, _, df_log = crear_sub_problema_random(
                X_sub, y_sub, 
                partes=partes, 
                time_max=time_max, 
                tol=tol, 
                keep_xi=keep_xi_iter, 
                random_state=current_seed, 
                solapar=solapar
            )
            
            # Validación
            if not K_list:
                print(f"[Iter {i}] Advertencia: No se generaron columnas. Saltando.")
                continue

            # --- 3. ADAPTACIÓN DE DIMENSIONES ---
            # Si usamos subsample, K_list trae vectores xi de tamaño 'fixed_sample_size'.
            # El Master necesita vectores de tamaño 'n_samples' (Total).
            # Forzamos xi=None para que el Master recalcule el error real sobre todo el set.
            if train_size_param is not None:
                K_list = [(col[0], col[1], None) for col in K_list]

            # --- 4. RESOLVER MASTER (FILTRADO GLOBAL) ---
            # El Master evalúa las columnas candidatas (entrenadas en el sub-set)
            # contra el dataset COMPLETO (X_train, y_train).
            master_results = solve_master_primal_v2(
                X_train, y_train, K_list, 
                tipo="convexo", 
                C=1.0, 
                tol=1e-6,
                mosek_params=mosek_params_from_tol(1e-6, presolve_level=1, optimizer_code=0)
            )
            
            theta = master_results[0]
            master_obj = master_results[4] 
            print(f"Master Objective (Iter {i}): {master_obj:.4f}")
            
            gc.collect()
            
            # Filtrar columnas con peso relevante
            active_indices = np.where(np.abs(theta) > tol)[0]
            n_activos = len(active_indices)
            print(f"[Iter {i}] Columnas activas seleccionadas: {n_activos}")
            
            if n_activos == 0:
                print(f"[Iter {i}] Ninguna columna pasó el filtro del Master.")
                del master_results, K_list, theta
                gc.collect()
                continue 

            # Guardar columnas limpias
            K_list_cleaned = [(K_list[idx][0], K_list[idx][1], None) for idx in active_indices]
            
            K_ini_end.extend(K_list_cleaned)
            
            # Limpieza de memoria
            del master_results, K_list, theta, active_indices, K_list_cleaned
            if train_size_param is not None:
                del X_sub, y_sub
            gc.collect() 
            
        except Exception as e:
            print(f"!!!!!!!! ERROR FATAL EN ITERACIÓN {i} (Random State={current_seed}) !!!!!!!!")
            print(f"Error: {e}")
            continue 

    print("\n--- Construcción del Pool Finalizada ---")
    print(f"Columnas totales recolectadas en K_ini_end: {len(K_ini_end)}") 
    return K_ini_end


    
def K_forest_fast_sklearn(X_train, y_train, n_iters=50, 
                          n_subspaces=100, # Equivalente a 'partes' (para cubrir todo el ancho)
                          max_features_ratio=0.1, 
                          fixed_sample_size=None,
                          tol_pruning=1e-5, 
                          random_state=42, solapar=True,tol=1e-06):

    
    n_samples, n_features = X_train.shape
    K_ini_end=[]
    
    # Asegurar formato binario -1, 1
    y_train_sign = np.where(y_train <= 0, -1, 1)
    
    # Configurar tamaño de filas
    train_size_param = None
    if fixed_sample_size is not None and fixed_sample_size < n_samples:
        train_size_param = fixed_sample_size
    
    print(f"🌲 K_forest_fast (OneVsRest+LinearSVC) | N={n_samples}, D={n_features}")
    print(f"   Config: {n_iters} iters x {n_subspaces} subspaces.")

    rng = np.random.default_rng(random_state)
    
    for i in range(n_iters):
        K_pool = []
        # 1. Sampling de FILAS (Una vez por iteración principal)
        if train_size_param is not None:
            X_sub_rows, _, y_sub_rows, _ = train_test_split(
                X_train, y_train_sign, 
                train_size=train_size_param, 
                stratify=y_train_sign, 
                random_state=i + random_state
            )
        else:
            X_sub_rows, y_sub_rows = X_train, y_train_sign

        # 2. Bucle de Subespacios (Columnas)
        for j in range(n_subspaces):
            
            # Sampling de COLUMNAS
            n_sub_feat = int(n_features * max_features_ratio)
            if n_sub_feat < 1: n_sub_feat = 1
            
            idx_cols = rng.choice(n_features, size=n_sub_feat, replace=False)
            idx_cols_sorted = np.sort(idx_cols)
            
            # Cortar datos
            X_sub_final = X_sub_rows[:, idx_cols_sorted]
            
            try:
                # 3. Resolver con TU CONFIGURACIÓN EXACTA
                # Corregido: eliminada la doble coma y asegurado dual=True
                base_estimator = LinearSVC(
                    C=1.0, 
                    loss="hinge", 
                    dual=True, # Obligatorio para Hinge y rápido si n_feat > n_samples
                    tol=1e-3, 
                    random_state=i*j, 
                    max_iter=10000
                )
                
                # Envolver en OneVsRestClassifier como pediste
                clf = OneVsRestClassifier(base_estimator, n_jobs=1) 
                
                clf.fit(X_sub_final, y_sub_rows)
                
                # --- Recuperar w y b ---
                # LinearSVC en OneVsRest binario puede tener 1 solo estimador.
                # Verificamos si estimators_ tiene 1 elemento (caso binario normal)
                if hasattr(clf, 'estimators_') and len(clf.estimators_) == 1:
                    est = clf.estimators_[0]
                    # OJO: OneVsRest a veces entrena para la clase 0. 
                    # LinearSVC solo tiene coef_
                    w_sub = est.coef_.flatten()
                    b_sub = est.intercept_[0]
                    
                    # Verificación de seguridad de signo (opcional pero recomendada):
                    # Si las clases son [-1, 1], sklearn suele mapear el estimador a la clase 1.
                    # Asumimos que está correcto.
                    
                else:
                    # Caso raro o multiclass real, tomamos el correspondiente a la clase 1
                    # (No debería pasar si y_train es binario puro)
                    w_sub = clf.coef_.flatten()
                    b_sub = clf.intercept_[0]

                # 4. Proyección al espacio original
                w_full_sparse = sp.csr_matrix(
                    (w_sub, idx_cols_sorted, np.array([0, len(w_sub)])), 
                    shape=(1, n_features)
                ).T 
                
                # Esto hace que todas las columnas "pesen" lo mismo numéricamente ####### QUIZAS HAYA QUE CAMBIARLO. 
                nrm = sp.linalg.norm(w_full_sparse)
                if nrm > 1e-9:
                    w_full_sparse = w_full_sparse / nrm
                else:
                    # Si el vector es nulo, lo saltamos
                    continue
                # Pruning
                w_full_sparse.data[np.abs(w_full_sparse.data) < tol_pruning] = 0
                w_full_sparse.eliminate_zeros()
                
                if w_full_sparse.nnz == 0: continue

                # Guardar (sin xi para ahorrar memoria)
                K_pool.append((w_full_sparse.astype(np.float32), float(b_sub), None))
                
            except Exception as e:
                # print(f"   [Iter {i}.{j}] Error: {e}")
                pass
        
        # Limpieza periódica
        gc.collect()
        master_results = solve_master_primal_v2(
                X_train, y_train, K_pool, 
                tipo="convexo", 
                C=1.0, 
                tol=1e-6,
                mosek_params=mosek_params_from_tol(1e-6, presolve_level=1, optimizer_code=0)
            )

        theta = master_results[0]
        master_obj = master_results[4] 
        print(f"Master Objective (Iter {i}): {master_obj:.4f}")
            
        gc.collect()
            
        # Filtrar columnas con peso relevante
        active_indices = np.where(np.abs(theta) > tol)[0]
        n_activos = len(active_indices)
        print(f"[Iter {i}] Columnas activas seleccionadas: {n_activos}")
        
        if n_activos == 0:
            print(f"[Iter {i}] Ninguna columna pasó el filtro del Master.")
            del master_results, K_pool, theta
            gc.collect()
            continue 

        # Guardar columnas limpias
        K_list_cleaned = [(K_pool[idx][0], K_pool[idx][1], None) for idx in active_indices]
        
        K_ini_end.extend(K_list_cleaned)
        
        # Limpieza de memoria
        del master_results, K_pool, theta, active_indices, K_list_cleaned
        if train_size_param is not None:
            del X_sub_rows, y_sub_rows
        gc.collect() 
    return K_ini_end
                
    
    













# --- FUNCIONES DE PRICING (ACTUALIZADAS) ---

def _solve_pricing_base(X, y, grad_w_vector, C, PRICING_PARAMS, M_box, 
                        add_stabilization_cut=False, add_component_cut=False,warm_start=True,verbose=True):
    """Función base interna para todos los pricings"""
    
    #        res = _solve_pricing_base(X, y, grad_w, C, PRICING_PARAMS, M_box, add_stabilization_cut=False)
    #        w_res, _, _, obj_res = res
    n_samples, n_features = X.shape
    y_neg = np.where(y <= 0, -1, 1)

    w = cp.Variable(n_features)
    b = cp.Variable()
    xi = cp.Variable(n_samples, nonneg=True)
    
    grad_w_flat = grad_w_vector.flatten() 
    if M_box is not None:
        constraints = [
            cp.multiply(y_neg, X @ w + b) >= 1 - xi,
            xi >= 0,
            w <= M_box,
            w >= -M_box
        ]
    else:
        constraints = [
            cp.multiply(y_neg, X @ w + b) >= 1 - xi,
            xi >= 0,
        ]
    
    # Añadir cortes de estabilización opcionales
    if add_stabilization_cut:
        constraints.append(-grad_w_flat @ w >= 0)
    if add_component_cut:
        constraints.append(cp.multiply(w, -grad_w_flat) >= 0)
    
    objective = cp.Minimize(C * cp.sum(xi) - grad_w_flat @ w)
    prob_pricing = cp.Problem(objective, constraints)
    
    prob_pricing.solve(
        solver=cp.MOSEK, 
        mosek_params=PRICING_PARAMS, 
        warm_start=warm_start, #afectara el apagarlo? 
        verbose=verbose # Silenciado por defecto
    )
    
    pricing_obj_val = prob_pricing.value 

    st = (prob_pricing.status or "").lower()
    ok_opt = st.startswith("optimal")
    ok_has_vals = (w.value is not None) and (b.value is not None) and (xi.value is not None)
    
    if ok_opt or ok_has_vals:
        if not ok_opt: print(f"[pricing] status={prob_pricing.status} -> guardo sol. factible")
        try:
            print(f"Pricing Obj Val (LB): {float(pricing_obj_val):.4f}")
            print(f"Norma de w generada: {float(np.linalg.norm(w.value)):.4f}")
        except Exception: pass
            
        w_sparse = sp.csc_matrix(w.value.reshape(-1, 1), dtype=np.float32)
        xi_sparse = sp.csc_matrix(xi.value.reshape(-1, 1), dtype=np.float32)
        
        return (w_sparse, b.value, xi_sparse, pricing_obj_val) 
    else:
        # Fallback (solo retornamos None, la lógica de fallback se maneja en la clase)
        print(f"[pricing] status={prob_pricing.status} sin solución usable.")
        return (None, None, None, pricing_obj_val) # Retorna el LB aunque w sea None

def solve_pricing_problem_caja(X, y, grad_w, K=None, C=1.0, PRICING_PARAMS={}, M_box=1e6,add_stabilization_cut=False, warm_start=True,verbose=True):
    """
    Wrapper ultra-robusto con 3 niveles de defensa.
    1. IPM (Rápido) -> Si falla...
    2. Simplex Dual (Robusto) -> Si crashea...
    3. Fallback Heurístico (Emergencia).
    """
    
    # --- INTENTO 1: Configuración Estándar (IPM) ---
    try:
        res = _solve_pricing_base(X, y, grad_w, C, PRICING_PARAMS, M_box, warm_start=warm_start,verbose=verbose,add_stabilization_cut=False)
        w_res, _, _, obj_res = res
        
        # Chequeo de éxito numérico
        if w_res is not None and np.isfinite(obj_res):
            return res # Éxito a la primera
            
    except Exception as e:
        print(f"⚠️ Intento 1 (IPM) crasheó: {e}")

    # --- INTENTO 2: Estrategia Robusta (Simplex Dual + No Presolve) ---
    print(f"⚠️ Pricing inestable. Reintentando con Simplex Dual sin Presolve...")
    
    params_rescue = PRICING_PARAMS.copy()
    params_rescue["MSK_IPAR_OPTIMIZER"] = 2  # Simplex Dual
    params_rescue["MSK_IPAR_PRESOLVE_USE"] = 0 # Sin Presolve
    
    try:
        # AQUÍ ESTABA EL PROBLEMA: Si esto lanza error, el script moría.
        # Ahora lo capturamos.
        res_retry = _solve_pricing_base(X, y, grad_w, C, params_rescue, M_box, warm_start=True,verbose=True, add_stabilization_cut=False)
        w_retry, _, _, obj_retry = res_retry
        
        if w_retry is not None and np.isfinite(obj_retry):
            print("✅ Pricing recuperado con Simplex Dual.")
            return res_retry 
        else:
            print("⚠️ Simplex Dual terminó pero no dio solución usable.")
            
    except Exception as e:
        # Captura el SolverError de Mosek si explota el Simplex
        print(f"⚠️ CRASH en Simplex Dual: {e}")
        print("   -> Pasando a protocolo de emergencia.")

    # --- INTENTO 3: Fallback Heurístico (Emergencia) ---
    print(f"🔥 Pricing falló totalmente. Activando Fallback Heurístico...")
    
    # Devolvemos None explícitamente en la primera posición para que
    # iteracion_pricing detecte el fallo y active el Rayo de Gradiente o Canónicos
    return generar_set_columnas_costos_reducidos_sparse(
        X, y, grad_w, K, X.shape[1], X.shape[0], eps=1e-8, k_max=50
    )


def solve_pricing_problem_combinado_sumar_restricto_2(X, y, grad_w, K=None, C=1.0, PRICING_PARAMS={}, M_box=1e4):
    
    
    res = _solve_pricing_base(X, y, grad_w, C, PRICING_PARAMS, M_box, add_stabilization_cut=True)
    # Manejo del fallback
    if res[0] is None:
        print(f"Activando fallback heurístico para 'caja_restricto'...")
        return generar_set_columnas_costos_reducidos_sparse(
            X, y, grad_w, K, X.shape[1], X.shape[0], eps=1e-8, k_max=50
        )
    return res
def solve_pricing_problem_combinado_componente(X, y, grad_w, K=None, C=1.0, PRICING_PARAMS={}, M_box=1e4):
    # Nota: M_box se pasa pero no se usa si add_component_cut es True y no hay BBox
    # La he refactorizado para usar la base, que SÍ usa BBox.
    res= _solve_pricing_base(X, y, grad_w, C, PRICING_PARAMS, M_box, 
                               add_component_cut=True)
    # Manejo del fallback
    if res[0] is None and res[3] == np.inf:
        print(f"Activando fallback heurístico para 'caja_restricto'...")
        return generar_set_columnas_costos_reducidos_sparse(
            X, y, grad_w, K, X.shape[1], X.shape[0], eps=1e-8, k_max=25
        )
    return res    
    

# --- CLASE PRINCIPAL (ACTUALIZADA) ---

class generacion_columnas:
    def __init__(self, tol, M_box=None):
        self.M_box = M_box
        self.tol = tol
        
        self.opt_val_fin = [] # Guarda el Upper Bound (UB) del master
        self.lb_fin = []      # Guarda el Lower Bound (LB) del pricing
        self.alpha_set_fin = [] # Guarda el alpha PURO
        self.grad_w_actual = None # Guarda el gradiente (completo o solo alpha)
        self.gradient_strategy = "full_gradient"
        self.smoothing_factor = 0.0 # 0.0 = sin suavizado
        self.memoria_theta = {}
        self.memoria_permanente = {}
        self.current_mosek_params = {}
        self.master_mosek_params = {}
        self.terminamos = False
        self.status = "init"
        self.i = 0
        self.K_ini_dict = {}
        self.K_generados = {}
        self.contador_columnas = 0
        self.n_data = 0
        self.n_columns = 0
        self.X = None
        self.y = None
        self.columnas_eliminadas_iteracion = []

    def ingresar_data_(self, X_train, y_train):
        self.X = X_train
        self.y = y_train
        self.n_data, self.n_columns = X_train.shape
    
    def ingresar_parametros_master_pricing(self, C, M, K_ini, tipo="convexo", 
                                           pricing=solve_pricing_problem_caja, 
                                           gradient_strategy="full_gradient"):
        
        # K_ini debe ser una LISTA de tuplas (w,b,xi) esparsas
        self.K_ini_dict = K_ini_version_dict(K_ini) 
        self.contador_columnas = len(self.K_ini_dict)
        for i in self.K_ini_dict.keys(): 
            self.memoria_theta[i] = [] 
            self.memoria_permanente[i] = []
        self.C = C        
        self.tipo = tipo
        self.M = M
        self.pricing = pricing
        self.gradient_strategy = gradient_strategy
        
    def filtrar_columnas_inutiles(self, K_dict):
        columnas_a_eliminar = []
        for nombre, valores_theta in self.memoria_theta.items():
            if len(valores_theta) >= self.n_periodos:
                recientes = valores_theta[-self.n_periodos:]
                if all(abs(theta) < self.umbral_theta for theta in recientes):
                    columnas_a_eliminar.append(nombre)
        for nombre in columnas_a_eliminar:
            del K_dict[nombre]
            del self.memoria_theta[nombre]
            self.memoria_permanente[nombre].append(-self.M) # -M es un marcador
        if columnas_a_eliminar:
            self.columnas_eliminadas_iteracion.append(columnas_a_eliminar)
            print(f"🔍 Columnas eliminadas en iteración {self.i}: {columnas_a_eliminar}")
        return K_dict

    def iteracion_generacion_master(self, K_dict, master=solve_master_primal_v2):
        K_list, K_list_keys = convertir_dict_a_K(K_dict)
        
        resultados_master_ = master(self.X, self.y, K_list, self.tipo, self.C, self.tol, 
                                    mosek_params=self.master_mosek_params, M_box=self.M_box)
        
        (theta_opt, eta, alpha_puro, w_combo, primal_UB, b, xi, 
         grad_w_completo) = resultados_master_
        print("***"*10)
        print(f"Master UB (Primal Obj): {primal_UB}")
        print("***"*10)
        
        self.opt_val_fin.append(primal_UB)
      
        self.alpha_set_fin.append(alpha_puro)
        
        # Seleccionar el gradiente para el PRÓXIMO pricing
        current_grad = None
        if self.gradient_strategy == "alpha_only":
            current_grad = alpha_puro
            if alpha_puro is not None: print(f"Estrategia de Pricing: ALPHA_ONLY (Norma: {np.linalg.norm(alpha_puro):.4f})")
        else:
            current_grad = grad_w_completo
            if grad_w_completo is not None: print(f"Estrategia de Pricing: FULL_GRADIENT (Norma: {np.linalg.norm(grad_w_completo):.4f})")

        # Aplicar suavizado (si está habilitado)
        if self.smoothing_factor > 0 and current_grad is not None:
            if self.grad_w_actual is None: # grad_w_actual ahora es el suavizado
                self.grad_w_actual = current_grad
            else:
                self.grad_w_actual = ( (1 - self.smoothing_factor) * self.grad_w_actual + 
                                       self.smoothing_factor * current_grad )
            print(f"Gradiente SUAVIZADO (Norma: {np.linalg.norm(self.grad_w_actual):.4f})")
        else:
            self.grad_w_actual = current_grad # Sin suavizado
        
        # Registrar theta
        if theta_opt is not None:
            for j, key in enumerate(K_list_keys):
                if j < len(theta_opt):
                    self.memoria_theta[key].append(theta_opt[j])
                    self.memoria_permanente[key].append(theta_opt[j])
        
        return K_dict

    def iteracion_generacion_pricing(self, K_dict): #revisar porque este normaliza el W que se va a agregar a K. 
            keys_antes = set(K_dict.keys())
            
            grad_w = self.grad_w_actual
            if grad_w is None:
                print("ERROR: El gradiente es None. Deteniendo la corrida.")
                self.terminamos = True; self.status = "ERROR_NO_GRADIENT"; return K_dict
    
            norm_grad = np.linalg.norm(grad_w)
            grad_w_normalizado = grad_w

            grad_w_normalizado = grad_w / norm_grad if norm_grad > 1e-8 and self.tipo =="conico" else grad_w #esto entorpece la convergencia. quizas es util si es conico
            print(f"Llamando al pricing con norma de gradiente: {norm_grad:.2f}")
    
            # Llamar al pricing. Espera 4 valores: (w, b, xi, obj_val)
            res = self.pricing(self.X, self.y, grad_w_normalizado, K_dict, self.C, 
                               self.current_mosek_params, self.M_box)
            
            # Caso A: El pricing retornó una tupla (w, b, xi, obj_val)
            if isinstance(res, tuple) and len(res) == 4:
                w_new_sparse, b_new, xi_new_pricing, pricing_obj_val = res 
                
                # --- TAREA 1: Guardar el Lower Bound ---
                self.lb_fin.append(pricing_obj_val) 
                print(f"GAP Actual (UB - LB): {self.opt_val_fin[-1] - pricing_obj_val}")
                
                if w_new_sparse is None:
                    print("[pricing] El solver no retornó una columna válida. No se agrega nada.")
                    return K_dict
    
                # --- TAREA 2: Normalizar la Columna ---
                norma_w_new = sp.linalg.norm(w_new_sparse)
                w_para_K = w_new_sparse 
                if norma_w_new > 1e-8 and self.tipo=="conico": #normalizamos solo si es conico. 
                    w_para_K = w_new_sparse / norma_w_new
                
                    print(f"Columna generada (Norma {norma_w_new:.2f}) normalizada a Norma 1.0")
                else:
                    print("Advertencia: Pricing generó una columna casi nula.")
                b_para_K = b_new; xi_para_K = None
                
                # --- TAREA 3: Chequeo de Hull ---
#                K_list_sparse_w = [col[0] for col in K_dict.values()]
#                if is_column_in_hull(w_para_K, K_list_sparse_w, self.tipo, self.tol):
#                    print("Columna (normalizada) ya está en el hull. CONVERGENCIA ALCANZADA.")
#                    self.status = "optimo_col_in_hull"; self.terminamos = True
#                    return K_dict
    
                # --- TAREA 4: Agregar Columna ---
                nombre_columna = f'k_{self.contador_columnas}'; self.contador_columnas += 1
                self.memoria_theta[nombre_columna] = [np.nan] * (self.i + 1)
                self.memoria_permanente[nombre_columna] = [np.nan] * (self.i + 1)
                K_dict[nombre_columna] = (w_para_K, b_para_K, xi_para_K) 
                self.K_generados[nombre_columna] = (w_para_K, b_para_K, xi_para_K)
                print(f"-> Agregada 1 nueva columna normalizada: {nombre_columna}")
                return K_dict
            
            # Caso B: Fallback retornó un dict
            elif isinstance(res, dict):
                K_new = res 
                nuevas = [k for k in K_new.keys() if k not in keys_antes]
                for k in nuevas:
                    self.memoria_theta[k] = [np.nan] * (self.i + 1)
                    self.memoria_permanente[k] = [np.nan] * (self.i + 1)
                    self.K_generados[k] = K_new[k]
                self.contador_columnas = max(self.contador_columnas, len(K_new))
                print(f"-> Agregadas {len(nuevas)} columnas por fallback.")
                # El fallback no tiene un LB, así que duplicamos el UB
                self.lb_fin.append(self.opt_val_fin[-1])
                return K_new
                
            else:
                print("[pricing] Salida no reconocida o fallida; no se agregan columnas")
                self.lb_fin.append(self.opt_val_fin[-1]) # No hay LB
                return K_dict
        
    def run(self, max_iter, umbral_theta, n_periodos, frecuencia_check, master=solve_master_primal_v2):
        self.i = 0
        time_ini = time.time()
        self.umbral_theta = umbral_theta
        self.n_periodos = n_periodos
        self.frecuencia_check = frecuencia_check
        self.columnas_eliminadas_iteracion = []
        self.K_generados = {}
        K_dict = dict(self.K_ini_dict)
        self.terminamos = False
        self.status = "running"
        
        while self.terminamos == False:
            print("***" * 30); print(f"inicio con iteracion # {self.i} ")
            
            if self.i > 0 and self.i % frecuencia_check == 0 and self.i != max_iter:
                K_dict = self.filtrar_columnas_inutiles(K_dict)
                
            K_dict = self.iteracion_generacion_master(K_dict, master=master)
             
            # Criterio de parada: Gap de dualidad
            if self.i > 0 and self.opt_val_fin[-1] is not None and self.lb_fin[-1] is not None:
                gap = self.opt_val_fin[-1] - self.lb_fin[-1]
                if abs(gap) <= self.tol:
                    print("****"*10, " fin ", "****"*8)
                    print(f"Criterio de parada: GAP de dualidad < tol ({gap:.2e} <= {self.tol})")
                    self.status = "optimo_gap"
                    self.terminamos = True
                    self.K_fin = K_dict
                    break
            # Criterio de parada: mejora marginal en F.O.
            if self.i > 1 and np.linalg.norm(self.opt_val_fin[-1] - self.opt_val_fin[-2]) <= self.tol:
                print("****"*10, " fin ", "****"*8)
                print("la diferencia de las fo fue", np.linalg.norm(self.opt_val_fin[-2]-self.opt_val_fin[-1]))
                print("Criterio de parada: mejora marginal alcanzada.")
                self.status="optimo"
                self.terminamos=True
                self.K_fin=K_dict
                break
            
            if self.i >= max_iter:
                print("****"*10, " fin ", "****"*8)
                print("Criterio de parada: máximo de iteraciones.")
                self.status = "max_iter"
                self.terminamos = True
                self.K_fin = K_dict
                break
                
            K_dict = self.iteracion_generacion_pricing(K_dict)
            if self.terminamos: # Chequear si el pricing (hull check) nos paró
                break
            
            self.i = self.i + 1
            
        time_fin = time.time()
        print("⏱ Tiempo total:", round((time_fin - time_ini)/60, 2), "minutos")
        self.time_total=time_fin - time_ini
        return self
    

class GeneracionColumnasDW:
    def __init__(self, tol,tol_master, M_box=None,threads=None,presolve_level=1,optimizer_code=0,umbral_theta=1e-6):
        self.M_box = M_box
        if M_box is None:
            self.M_box_pricing=1e6
        else:
            self.M_box_pricing=M_box
        self.tol = tol
        self.umbral_theta = umbral_theta
        
        # --- Resultados Globales ---
        self.opt_val_fin = [] 
        self.lb_fin = []      
        self.alpha_set_fin = [] 
        
        # --- Estado ---
        self.grad_w_actual = None 
        self.gradient_strategy = "full_gradient"
        self.smoothing_factor = 0.0 
        
        # --- Memoria de Historial (Puntos vs Rayos) ---
        self.memoria_theta = {} # Para Puntos (Theta)
        self.memoria_mu = {}    # Para Rayos (Mu)
        
        self.terminamos = False
        self.status = "init"
        self.i = 0
        
        # --- Almacenes de Columnas ---
        self.K_points_dict = {} # Puntos
        self.K_rays_dict = {}   # Rayos
        
        # Contadores para IDs únicos
        self.cnt_points = 0
        self.cnt_rays = 0
        
        self.X = None
        self.y = None
        
        # Configuraciones
        self.master_mosek_params = mosek_params_from_tol(
            tol_master, 
            threads=threads 
            #presolve_level=presolve_level, 
            #optimizer_code=optimizer_code
        )
        
        self.current_mosek_params = mosek_params_from_tol(
            self.tol, 
            threads=threads, 
            presolve_level=presolve_level, 
            optimizer_code=optimizer_code
        )
        self.verbose=True
        self.warm_start=True
        

    def ingresar_data(self, X_train, y_train):
        self.X = X_train
        self.y = y_train
    
    def ingresar_parametros(self, C, M, K_ini_points, K_ini_rays=None, 
                            tipo="convexo", 
                            pricing=solve_pricing_problem_caja, 
                            gradient_strategy="full_gradient", pricing_acceleration=False,pricing_acceleration_cota=False):
        
        self.C = C        
        self.tipo = tipo
        self.M = M
        self.pricing = pricing
        self.gradient_strategy = gradient_strategy
        self.pricing_acceleration = pricing_acceleration
        self.pricing_acceleration_cota = pricing_acceleration_cota
        
    
        
        # Inicializar Puntos
        # Asumimos que K_ini_points viene como LISTA de tuplas, lo pasamos a DICT
        if isinstance(K_ini_points, list):
            for col in K_ini_points:
                name = f'p_{self.cnt_points}' 
                self.K_points_dict[name] = col
                self.memoria_theta[name] = []
                self.cnt_points += 1
        elif isinstance(K_ini_points, dict):
            self.K_points_dict = K_ini_points.copy()
            self.cnt_points = len(self.K_points_dict)
            for k in self.K_points_dict: self.memoria_theta[k] = []

        # Inicializar Rayos (si hay)
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
                for k in self.K_rays_dict: self.memoria_mu[k] = []
        
    def limpiar_columnas(self, n_periodos):
        """
        Elimina columnas inactivas tanto de Puntos (theta) como de Rayos (mu).
        """
        # 1. Limpiar Puntos
        eliminar_p = []
        for nombre, hist in self.memoria_theta.items():
            if len(hist) >= n_periodos:
                recientes = hist[-n_periodos:]
                # Chequeo robusto de NaN
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
        # Convertir dicts a listas para el solver v3
        list_points, keys_points = convertir_dict_a_K(self.K_points_dict)
        list_rays, keys_rays = convertir_dict_a_K(self.K_rays_dict)
        
        prev_obj = self.opt_val_fin[-1] if self.opt_val_fin else float('inf')
        
        
        # LLAMADA AL NUEVO MASTER v3
        # Nota: Asegúrate de importar solve_master_primal_v3
        resultados = solve_master_primal_v3(
            self.X, self.y, 
            K=list_points, 
            tipo=self.tipo, 
            K_rayos=list_rays, # <--- Input nuevo
            C=self.C,  
            mosek_params=self.master_mosek_params, 
            M_box=self.M_box,
            verbose= self.verbose
        )
        
        # Desempaquetar los 9 valores (MU está al final)
        (theta_vals, eta, alpha, w_combo, obj_val, b, xi, grad_w, mu_vals) = resultados
        
        # --- CHEQUEO DE SEGURIDAD (NUEVO) ---
        if obj_val is None:
            print("🛑 Error Crítico: El Master no devolvió solución. Abortando iteración.")
            self.terminamos = True
            self.status = "MASTER_FAILURE"
            return self
        
        # Logging básico
        print(f"  >>> Master Obj: {obj_val:.7f} | Puntos: {len(list_points)} | Rayos: {len(list_rays)}")
        self.opt_val_fin.append(obj_val)
        self.alpha_set_fin.append(alpha)
        
        # Estrategia de Gradiente
        if self.gradient_strategy == "alpha_only":
            self.grad_w_actual = alpha
        else:
            self.grad_w_actual = grad_w

        # --- Guardar Historial Puntos (Theta) ---
        # Rellenar con 0.0 o NaN la iteración actual
        for k in keys_points: self.memoria_theta[k].append(0.0)
        
        if theta_vals is not None and len(theta_vals) > 0:
            for j, val in enumerate(theta_vals):
                if j < len(keys_points):
                    key = keys_points[j]
                    self.memoria_theta[key][-1] = val # Sobreescribir con valor real
        
        # --- Guardar Historial Rayos (Mu) ---
        for k in keys_rays: self.memoria_mu[k].append(0.0)
        
        if mu_vals is not None and len(mu_vals) > 0:
            for j, val in enumerate(mu_vals):
                if j < len(keys_rays):
                    key = keys_rays[j]
                    self.memoria_mu[key][-1] = val
                    

        
        if obj_val > prev_obj + self.tol: # Si sube más que un epsilon
            print(f"🚨 ALERTA: Master Obj subió de {prev_obj:.7f} a {obj_val:.7f} (Diff: {obj_val - prev_obj:.7f})")
            print("   -> Posible inestabilidad numérica o Warm Start fallido.")

    def _generar_rayos_aceleracion_cota(self, w_sparse, max_rayos=50):
            """
            OBSOLETAAAAA
            Revisa w_sparse. Si alguna coordenada w_i >= M_box (o <= -M_box),
            genera un rayo canónico en esa dirección.
            """
            nuevos_rayos = []
            if w_sparse is None: return nuevos_rayos
            
            # --- CORRECCIÓN DE SEGURIDAD ---
            # Obtenemos dimensiones directamente de X para evitar AttributeError
            if self.X is None: return []
            n_samples_local, n_features_local = self.X.shape
            # -------------------------------
            
            tol_borde = 1e-4 
            limite = self.M_box - tol_borde
            
            indices = w_sparse.indices
            data = w_sparse.data
            count = 0
            
            for idx, val in zip(indices, data):
                if count >= max_rayos: 
                    break
                
                # Caso 1: Golpeó el techo positivo (+M)
                if val >= limite:
                    # Usamos las variables locales
                    rayo = generar_canonico_sparse(n_features_local, idx, n_samples_local)
                    nuevos_rayos.append(rayo)
                    count += 1
                    
                # Caso 2: Golpeó el piso negativo (-M)
                elif val <= -limite:
                    # Usamos las variables locales
                    rayo = generar_canonico_con_signo_sparse(n_features_local, n_samples_local, idx, signo=-1.0)
                    nuevos_rayos.append(rayo)
                    count += 1
                    
            return nuevos_rayos
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
        
        if len(data) == 0: return []

        # 2. Determinar el Pico Máximo (Magnitud)
        max_abs_val = np.max(np.abs(data))
        
        # Si el vector es puro ruido (muy pequeño), no aceleramos
        if max_abs_val < 1.0: 
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
    def iteracion_pricing(self,max_rayos):
            # 1. Validación inicial
            if self.grad_w_actual is None:
                print("⚠️ Error: Gradiente nulo.")
                self.terminamos = True; return
    
            # 2. Llamada al Pricing (Pasando K_rays para el fallback)
            
            grad_w_scaled=self.grad_w_actual
            if np.linalg.norm(self.grad_w_actual)>10:
                grad_w_scaled=self.grad_w_actual/np.linalg.norm(self.grad_w_actual)
            
            M_box_pricing= self.M_box_pricing
            
            
            res = self.pricing(self.X, self.y, grad_w_scaled, 
                               self.K_rays_dict, 
                               self.C, self.current_mosek_params, M_box_pricing,add_stabilization_cut=False, warm_start= self.warm_start, verbose=self.verbose)
            
            # ---------------------------------------------------------
            # CASO A: Pricing Exitoso (Encontró un PUNTO acotado)
            # ---------------------------------------------------------
            if isinstance(res, tuple) and len(res) == 4:
                w_sp, b_val, xi_sp, lb_val = res
                self.lb_fin.append(lb_val)
                
                if w_sp is None: 
                    print("  [Pricing] No se generó columna."); return
    
                # Normalización Vectorial (Si es Cónico)
                if self.tipo == "conico":
                    norm_w = sp.linalg.norm(w_sp)
                    if norm_w > 1e-9: w_sp = w_sp / norm_w
                
                # Guardar como PUNTO
                name = f'p_{self.cnt_points}'
                self.cnt_points += 1
                self.memoria_theta[name] = [0.0] * (self.i + 1) 
                self.K_points_dict[name] = (w_sp, b_val, xi_sp)
                print(f"  -> Nuevo PUNTO agregado: {name}")
                
                # --- ACELERACIÓN UNIFICADA ---
                # Usamos la función que combina lógica de "Golpe de Caja" y "Magnitud Grande"
                if self.pricing_acceleration:
                    w_original = res[0] # Usamos el w original (sin normalizar)
                    
                    # Generar hasta 20 rayos extra de las coordenadas dominantes
                    rayos_extra = self._generar_rayos_aceleracion(w_original, max_rayos=max_rayos)
                    
                    if rayos_extra:
                        for ray in rayos_extra:
                            r_name = f'r_{self.cnt_rays}'
                            self.cnt_rays += 1
                            self.memoria_mu[r_name] = [0.0] * (self.i + 1)
                            self.K_rays_dict[r_name] = ray
                        
                        print(f"  🚀 Aceleración: {len(rayos_extra)} RAYOS extra (Drivers Principales).")
    
            # ---------------------------------------------------------
            # CASO B: Pricing Falló -> Se activó Fallback (Son RAYOS)
            # ---------------------------------------------------------
            elif isinstance(res, tuple) and len(res) == 2:
                K_rays_modified, info = res 
                agregadas = info.get('agregadas', 0)
                
                # Sub-caso B1: El fallback Canónico funcionó
                if agregadas > 0:
                    # Renombrar k_ (del fallback) a r_ (nuestra clase)
                    keys_to_rename = [k for k in self.K_rays_dict.keys() if k.startswith('k_')]
                    for old_key in keys_to_rename:
                        val = self.K_rays_dict[old_key]
                        del self.K_rays_dict[old_key]
                        new_name = f'r_{self.cnt_rays}'
                        self.cnt_rays += 1
                        self.K_rays_dict[new_name] = val
                        self.memoria_mu[new_name] = [0.0] * (self.i + 1)
                    
                    print(f"  -> Fallback Canónico: {len(keys_to_rename)} nuevos RAYOS agregados.")
                    self.lb_fin.append(-6742069) # Mantener GAP abierto
    
                # Sub-caso B2: El fallback Canónico falló (Ya tenemos todos esos ejes)
                else:
                    # ==============================================================
                    # ESTRATEGIA DE EMERGENCIA: RAYO DE GRADIENTE (CORREGIDO)
                    # ==============================================================
                    print("  ⚠️ Fallback Canónico AGOTADO (Sin columnas nuevas).")
                    
                    # 1. Obtener el gradiente actual
                    gradiente = self.grad_w_actual.flatten()
                    n_features_total = gradiente.shape[0]
                    
                    # 2. Configuración de Velocidad DINÁMICA
                    # Usamos min() para que funcione tanto en datasets pequeños (1000)
                    # como en el gigante (13M).
                    n_top = min(20000, n_features_total) #los top 5000 mejores.
                    
                    # 3. Identificar índices más importantes
                    if n_top == n_features_total:
                        # Si queremos todos, no necesitamos argpartition
                        idx_top = np.arange(n_features_total)
                    else:
                        # np.argpartition es mucho más rápido que sort
                        idx_top = np.argpartition(np.abs(gradiente), -n_top)[-n_top:]
                    
                    # 4. Extraer valores (Signo negativo para descenso)
                    vals_top = -gradiente[idx_top]
                    
                    # 5. Construcción SPARSE Consistente
                    data = vals_top
                    indices = idx_top
                    indptr = np.array([0, len(data)]) 
                    
                    rayo_grad_sparse = sp.csc_matrix(
                        (data, indices, indptr), 
                        shape=(n_features_total, 1), 
                        dtype=np.float32
                    )
                    
                    # 6. Inserción
                    r_name = f'r_grad_{self.i}' 
                    self.cnt_rays += 1
                    
                    self.K_rays_dict[r_name] = (rayo_grad_sparse, 0.0, None)
                    self.memoria_mu[r_name] = [0.0] * (self.i + 1)
                    
                    print(f"  🔥 ACTIVANDO EMERGENCIA: Rayo de Gradiente agregado ({n_top} nnz).")
                    
                    # 7. Forzar continuación
                    self.lb_fin.append(-6742069)
    
            else:
                print("  [Pricing] Formato de retorno desconocido.")
                self.lb_fin.append(-6742069)
                
    def actualizar_parametros_solver(self):
        """
        Ajusta dinámicamente las tolerancias y configuraciones de Mosek
        basándose en la mejora de la función objetivo del Master.
        Objetivo: Ir de 'Rápido y Laxo' a 'Lento y Preciso'.
        """
        # Necesitamos al menos 2 iteraciones para comparar
        if len(self.opt_val_fin) < 2:
            return

        # Calcular mejora absoluta (descenso)
        curr_obj = self.opt_val_fin[-1]
        prev_obj = self.opt_val_fin[-2]
        diff = abs(prev_obj - curr_obj)
        
        # Definir estados
        # Estado 1: Cambios grandes -> Priorizar Velocidad
        if diff > 1e-1:
            new_tol = 1e-5
            new_presolve = 1 # ON
            optimizer_code=0 #descenso
            mode_name = "VELOCIDAD (Coarse)"
            VER=False
            WS=True
            self.n_periodos= self.n_periodos
            print("*"*10,mode_name,"*"*10)
            
            
        # Estado 2: Cambios medios -> Precisión Estándar
        elif diff > 1e-3:
            new_tol = 1e-6
            new_presolve = 1 # ON
            optimizer_code=2 #dual
            mode_name = "ESTÁNDAR (Medium)"
            VER= True
            WS= True
            self.n_periodos= self.n_periodos*1.1
            print("*"*10,mode_name,"*"*10)
        # Estado 3: Cambios finos -> Precisión Máxima (Usamos la definida en __init__)
        else:
            # Aquí usamos self.tol_master que definiste al inicio (ej. 1e-8)
            # Y apagamos presolve para evitar mentiras numéricas
            new_tol = self.master_mosek_params.get("MSK_DPAR_INTPNT_TOL_REL_GAP", 1e-8) 
            # O forzamos una muy baja si no la leemos:
            if new_tol > 1e-6: new_tol = 1e-7 
            
            new_presolve = 0 # OFF (Crítico para convergencia final)
            optimizer_code=2 #dual
            mode_name = "PRECISIÓN (Fine-Tuning)"
            M= None
            self.M_box=M
            self.M_box_pricing=1e6
            VER= True
            WS= False
            self.n_periodos= self.n_periodos*1.5
            print("*"*10,mode_name,"*"*10)
        # --- Actualizar Parámetros ---
        
        # 1. Actualizar Master Params
        # Usamos tu helper 'mosek_params_from_tol'
        # Nota: Mantenemos optimizer_code=0 (Auto/IPM) por defecto, 
        # ya que tu wrapper de rescate se encarga de cambiar a Simplex si falla.
        
        self.master_mosek_params = mosek_params_from_tol(
            new_tol 
            #,presolve_level=new_presolve, 
            #optimizer_code=optimizer_code
        )
        
        # 2. Actualizar Pricing Params (Opcional: puedes querer que el pricing sea siempre estricto)
        # Generalmente es bueno que el pricing también se relaje al principio para generar columnas rápido.
        self.current_mosek_params = mosek_params_from_tol(
            new_tol, 
            presolve_level=new_presolve, 
            optimizer_code=optimizer_code
        )
        self.verbose= VER
        self.warm_start= WS

    def run(self, max_iter, n_periodos=5, frecuencia_check=5,max_rayos=20):
        self.i = 0
        self.terminamos = False
        time_ini = time.time()
        self.n_periodos=n_periodos
        while not self.terminamos and self.i < max_iter:
            self.actualizar_parametros_solver() #revisa master y pricing 
            print(f"\n=== Iteración {self.i} ===")
            
            # 1. Limpieza Periódica
            if self.i > 0 and self.i % frecuencia_check == 0:
                n_periodos=self.n_periodos
                self.limpiar_columnas(n_periodos)
                
            
            # 2. Resolver Master
            self.iteracion_master()
            
            # 3. Chequeo de Convergencia (GAP)
            if self.i > 0 and self.opt_val_fin[-1] is not None and self.lb_fin[-1] is not None:
                # Cuidado: lb_fin[-1] viene del pricing de la iteración ANTERIOR?
                # No, lb_fin se llena en el paso de pricing. 
                # En la iteración 0, hacemos master -> pricing (llena lb).
                # Al inicio de iter 1, chequeamos gap.
                
                # Necesitamos validar indices. Si acabamos de hacer master, aun no hacemos pricing de esta vuelta.
                # Usamos el LB de la vuelta pasada para comparar con el UB actual? 
                # O mejor chequeamos al final del loop.
                
                pass

            # 4. Resolver Pricing (Genera Puntos o Rayos)

            self.iteracion_pricing(max_rayos)
            
            # 5. Chequeo de Convergencia Final de Iteración
            if len(self.opt_val_fin) > 0 and len(self.lb_fin) > 0:
                gap = self.opt_val_fin[-1] - self.lb_fin[-1]
                print("GAP MASTER-PRICING ES DE: ", abs(gap))
                # El gap puede ser negativo si hay imprecisiones numéricas o si comparamos tiempos distintos
                # Usamos abs() o max(0, gap)
                
                # Nota: Si lb es -inf (fallback), el gap es grande, seguimos.
                if gap < self.tol and gap > self.tol:
                    print(f"✅ Convergencia alcanzada por GAP: {gap:.2e}")
                    self.status = "optimo_gap"
                    self.terminamos = True
            
            # Chequeo Estancamiento (Mejora marginal del UB)
            if self.i > 5:
                mejora = abs(self.opt_val_fin[-2] - self.opt_val_fin[-1])
                if mejora < self.tol:
                    print(f"⚠️ Convergencia por estancamiento de Obj (Diff: {mejora:.2e})")
                    self.status = "estancamiento"
                    self.terminamos = True

            self.i += 1
        
        if not self.terminamos:
            self.status = "max_iter"
            print("🛑 Máximo de iteraciones alcanzado.")

        print(f"⏱ Tiempo Total: {(time.time() - time_ini)/60:.2f} min")
        
        # Retornar estructura consolidada
        return {
            "K_points": self.K_points_dict,
            "K_rays": self.K_rays_dict,
            "opt_vals": self.opt_val_fin,
            "status": self.status
        }    
    
