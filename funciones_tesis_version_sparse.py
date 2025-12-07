# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 15:52:35 2025

@author: matta
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 11 17:10:30 2025
#solve_pricing_problem_combinado_componente, solve_pricing_problem_combinado_sumar_restricto , solve_pricing_problem_restricto_componente , solve_pricing_problem_restricto , solve_pricing_problem_original
@author: matta
"""
# In[0]:   
import gc #garbage collector
import cvxpy as cp
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import os #cargar archivos
import numpy as np
#from funciones_tesis import *
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import time

import scipy.sparse as sp


# dentro del for
# In[1]:

    
# funcion que arregla la data para que no tenga tanto diferencia de magnitud. Mosek tiene algo similar en presolve. 
def normalizar_ttsplit_data(X, y, test_size=0.2, random_state=2):
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

# one vs rest de skl esta optimizado para datasets grandes. tiene tanto error lineal o cuadratico, como tambien otras cosas de calculo.

def skl_svm(X, y, X_test, C=1.0, loss="hinge", max_iter=50000, solo_w_b_xi=False):
    y = np.where(y <= 0, -1, 1)
    clf = OneVsRestClassifier(LinearSVC(C=C, loss=loss, max_iter=max_iter))
    clf.fit(X, y)
    y_pred = clf.predict(X_test)

    # Extraer w y b del clasificador One-vs-Rest (usamos solo la primera clase binaria)
    est = clf.estimators_[0]
    w = est.coef_.flatten()
    b = est.intercept_[0]

    # Producto escalar seguro con sparse matrix (no usamos .toarray())
    margins = y * (X @ w + b)
    xi = np.maximum(0, 1 - margins)

    # Valor objetivo del problema primal:
    obj_val = np.linalg.norm(w) + C * np.sum(xi)
    print("🔍 Suma de ξ_i:", np.sum(xi))
    print("‖w‖ (norma 2):", np.linalg.norm(w))
    print("🔍 Valor óptimo (formulación tipo tesis):", obj_val)
    print(f"Puntos con ξ > 0: {np.sum(xi > 0)}")

    if solo_w_b_xi == True:
        return (w, b, xi)
    else:  # los parametros
        return [(w, b, xi), obj_val, y_pred]

#solve_svm_conic(X_train_ct, y_train, C=1.0,)

# el support vector machine base del paper. Se le agrega un time limit de 10 minutos. 

def solve_svm_conic(X, y, C=1.0, time_limit_sec=3600*60):
    n_samples, n_features = X.shape
    y_neg = np.where(y <= 0, -1, 1)  # usualmente o son 0, o son 1.

    # Variables
    w = cp.Variable(n_features)
    b = cp.Variable()
    xi = cp.Variable(n_samples, nonneg=True)
    tau = cp.Variable()  # reemplaza la norma 2

    # Constraints
    constraints = [
        cp.multiply(y_neg, X @ w + b) >= 1 - xi,
        cp.norm(w, 2) <= tau
    ]

    # Objective: tau + C * sum(xi)
    objective = cp.Minimize(tau + C * cp.sum(xi))  # cp.Minimize(0)
    # Solve
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, mosek_params={
               "MSK_DPAR_OPTIMIZER_MAX_TIME": float(time_limit_sec)})  # o ECOS para velocidad
    print("el valor objetivo es", prob.value)
    print("el valor de la norma de w es", np.linalg.norm(w.value))
    print("valor de F.O calculada es", np.linalg.norm(
        w.value) + C*np.sum(xi.value))
    # print(prob.value)
    return (w.value, b.value, xi.value)

def compactar_K(K, keep_xi=False, dtype=np.float32): 
    K2 = []
    for (w, b, xi) in K:
        w = np.asarray(w, dtype=dtype, order="C")   # denso contiguo y float32
        b = float(b) if b is not None else 0.0
        xi = np.asarray(xi, dtype=dtype, order="C") if (keep_xi and xi is not None) else None
        K2.append((w, b, xi))
    return K2
def compactar_K_a_sparse(K, keep_xi=True, dtype=np.float32):
    """
    Toma una lista K de tuplas (w_dense, b, xi_dense) y la convierte
    en una lista de tuplas (w_sparse (col), b, xi_sparse (col)).
    """
    K_sparse = []
    if not K:
        return K_sparse

    # --- Determinar dimensiones desde el primer elemento válido ---
    n_features = 0
    if K[0][0] is not None:
        n_features = len(K[0][0])
    else:
        # Buscar un 'w' no nulo para obtener n_features
        for col in K:
            if col[0] is not None:
                n_features = len(col[0])
                break
    if n_features == 0:
        raise ValueError("No se pudo determinar n_features desde el conjunto K.")
        
    n_samples = 0
    if keep_xi:
        for col in K:
            if col[2] is not None:
                n_samples = len(col[2])
                break
        if n_samples == 0:
             print("Advertencia: No se pudo determinar n_samples para xi en compactar_K_a_sparse.")

    # --- Iterar y Convertir ---
    for (w, b, xi) in K:
        
        # --- Convertir w ---
        if w is None:
            w_sparse = sp.csc_matrix((n_features, 1), dtype=dtype)
        else:
            # Asegurar que es 2D columna (N, 1) antes de esparsificar
            w_dense_col = np.asarray(w, dtype=dtype).reshape(-1, 1)
            w_sparse = sp.csc_matrix(w_dense_col)
            w_sparse.eliminate_zeros() # Limpiar ceros explícitos
            
        # --- Convertir b ---
        b_float = float(b) if b is not None else 0.0
        
        # --- Convertir xi ---
        if keep_xi:
            if xi is None:
                # Crear placeholder esparso de ceros
                xi_sparse = sp.csc_matrix((n_samples, 1), dtype=dtype)
            else:
                xi_dense_col = np.asarray(xi, dtype=dtype).reshape(-1, 1)
                xi_sparse = sp.csc_matrix(xi_dense_col)
                xi_sparse.eliminate_zeros()
        else:
            xi_sparse = None
            
        K_sparse.append((w_sparse, b_float, xi_sparse))
        
    return K_sparse
def crear_sub_problema(X_train, y_train, partes=101, time_max=60, tol=1e-06, keep_xi=True):
    K = [] # Esta K contendrá tuplas con w densos
    column_sets = []  
    log_info = []
    n_cols = X_train.shape[1]
    
    for i in range(1, partes):
        gc.collect() #porsiacaso jaja
        
        frac = i / (partes - 1)
        sub_frac = (i - 1) / (partes - 1)
        
        start = int(n_cols * sub_frac)
        end = int(n_cols * frac)    
        selected_indices = np.arange(start, end)
        XX = cortar(X_train, start, end)
        
        try:
            w_sub, b, xi = solve_svm_conic(XX, y_train, C=1.0, time_limit_sec=time_max) # w_sub y xi son densos
            
            w_full = np.zeros(n_cols, dtype=np.float32) # Sigue creando w_full denso
            w_sub_limp = w_sub.astype(np.float32, copy=False)
            w_full[start:end] = w_sub_limp
            
            # K es una lista de (dense, float, dense)
            K.append((w_full, float(b), xi)) 
            
            # ... (código para log_info sin cambios) ...
    
        except Exception as e:
            print(f"[i={i}] Error al resolver SVM en columnas {start}:{end} -> {e}")
            # ... (código para log_info de error sin cambios) ...
            
    df_log = pd.DataFrame(log_info)
    
    # --- CAMBIO CLAVE AQUÍ ---
    # Llamamos a la nueva función de esparsificación al final
    print(f"Compactando {len(K)} columnas de subproblemas a formato sparse...")
    K_sparse_list = compactar_K_a_sparse(K, keep_xi=keep_xi)
    print("Compactación finalizada.")
    
    return K_sparse_list, column_sets, df_log

# In[0]:
    
#funcion orientada a crear sub matrices de tamaño menor, para resolver el problema en un % de las dimensiones en vez de todas.
def cortar(X, start, end):

    X_slice = X[:, start:end]  # cortamos por los W.
    return X_slice

#funcion orientada a analizar el vector de resultados. se encarga de contar que features tienen un impacto muy pequeño.

def contar_pequeños(w, thresholds=[1e-2, 1e-4, 1e-6, 1e-8]):
    stats = {f"below_{t:.0e}": int((np.abs(w) < t).sum()) for t in thresholds}
    stats["exact_0"] = int((w == 0).sum())
    stats["over_0"] = int((w > 0).sum())
    stats["under_0"] = int((w < 0).sum())
    return stats

#El master problem propuesto por renault. Es importante notar que se requieren los XI. 

def solve_master_primal_original(X, y, K, tipo, C=1.0, tol=1e-6):
    y_neg = np.where(y <= 0, -1, 1)  # usualmente o son 0, o son 1.

    K_len = len(K)  # la dimension de theta
    n = len(y_neg)  # tamaño de xi # n_data

    # Variables ["convexo", "conico", "mayor_uno", "afin", "libre"]:
        
        #convexo es la combinacion convexa de las soluciones
        # conico es la combinacion conica, es decir, theta no negativo y que la suma de las soluciones por theta sea positiva. 
        # mayor 1 es conico, pero sin admitir las soluciones entre 0 y 1 de magnitud de theta 
        # afin es la combinacion afin. esta permite negativos pero la suma de los theta debe ser igual a 1. se sospecha que es la mejor.
        # libre no añade restricciones en theta, que se combine como prefiera el algoritmo. se sospecha que no es tan buena idea sin presolve.
        
    theta = cp.Variable(K_len, nonneg=True) #theta>=0 del tamaño del vector K.
    eta = cp.Variable()
    
    #K es de estructura lista de listas. K=[[w,b,xi]_k  ]
    # Precomputar cosas
    # shape (K, d) matriz de W_k es fila
    Ws = np.stack([col[0] for col in K])

    # Construcción de la restricción del cono
    w_combo = Ws.T @ theta     # sum_k θ_k * w^k, shape (d,)
    constraints = [cp.SOC(eta, w_combo)]        # restricción de norma
    # Agregar restricción sobre suma de theta si corresponde # combinación convexa #combinacion afin #algun tipo de combinacion que se debe buscar
    constraints.append(cp.sum(theta) == 1)

    # tipo "libre" no añade ninguna restricción adicional
    # Función objetivo: η + C ∑ θ_k * sum(ξ^k)
    sum_xi_k = np.array([np.sum(col[2]) for col in K])
    objective_term = sum_xi_k @ theta

    objective = cp.Minimize(eta + C * objective_term)

    # Definición del problema

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK,warm_start=True) #en esta parte es donde uno usualmente le agrega parametros al solver de mosek. CVXPY tambien tiene parametros por cierto. 
    

    # Extraer los multiplicadores duales del cono
    dual_alpha = constraints[0].dual_value  # esta es la norma 2 ≤ eta

    # crear soluciones presentables

    # theta
    theta_opt = theta.value
    # theta_opt[np.abs(theta_opt)<tol]=0 #destruyo los 0 falsos

    # alpha
    alpha = dual_alpha[1]
    # print(np.linalg.norm(alpha)) #debemos ver si da menor que 1
    # alpha[np.abs(alpha)<tol]=0 #eliminar los 0 falsos.
    #print("el alfa es: ",alpha )
    print("norma de alfa es", np.linalg.norm(alpha))

    return theta_opt, eta.value, alpha, w_combo.value, prob.value
# el Master problem.  Toma la data X, las asignacion de clase y (0 o 1), un string sobre que tipo se admiten, un C y una tolerancia para las soluciones

def solve_master_primal(X, y, K, tipo, C=1.0, tol=1e-6,mosek_params={}):
    y_neg = np.where(y <= 0, -1, 1)  # usualmente o son 0, o son 1.

    K_len = len(K)  # la dimension de theta
    n = len(y)  # tamaño de xi # n_data

    # Variables ["convexo", "conico", "mayor_uno", "afin", "libre"]:
        
        #convexo es la combinacion convexa de las soluciones
        # conico es la combinacion conica, es decir, theta no negativo y que la suma de las soluciones por theta sea positiva. 
        # mayor 1 es conico, pero sin admitir las soluciones entre 0 y 1 de magnitud de theta 
        # afin es la combinacion afin. esta permite negativos pero la suma de los theta debe ser igual a 1. se sospecha que es la mejor.
        # libre no añade restricciones en theta, que se combine como prefiera el algoritmo. se sospecha que no es tan buena idea sin presolve.
        
    if tipo in ["afin", "libre"]:
        theta = cp.Variable(K_len)  # permite negativos
    elif tipo in ["convexo", "conico", "mayor_uno"]:
        theta = cp.Variable(K_len, nonneg=True)
    else:
        print(" entregue un tipo como afin, convexo, conico, mayor_uno,o  libre")
        return None

    eta = cp.Variable()
    b = cp.Variable()
    xi = cp.Variable(n, nonneg=True)

    # Precomputar cosas
    # shape (K, d) matriz de W_k es fila
    Ws = np.stack([col[0] for col in K])

    # Construcción de la restricción del cono
    w_combo = Ws.T @ theta     # sum_k θ_k * w^k, shape (d,)
    constraints = [
        cp.SOC(eta, w_combo),        # restricción de norma
        cp.multiply(y_neg, X @ w_combo+b) >= 1-xi,
        xi >= 0,
    ]
    # Agregar restricción sobre suma de theta si corresponde # combinación convexa #combinacion afin #algun tipo de combinacion que se debe buscar
    if tipo in ["convexo", "afin"]:
        constraints.append(cp.sum(theta) == 1)
    elif tipo == "mayor_uno":
        constraints.append(cp.sum(theta) >= 1)
    elif tipo == "conico":
        constraints.append(cp.sum(theta) >= 0)
    else:   pass

    # tipo "libre" no añade ninguna restricción adicional
    # Función objetivo: η + C ∑ θ_k * sum(ξ^k)

    objective = cp.Minimize(eta + C * (cp.sum(xi)))

    # Definición del problema

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, mosek_params=mosek_params, warm_start=True) #en esta parte es donde uno usualmente le agrega parametros al solver de mosek. CVXPY tambien tiene parametros por cierto. 
    

    # Extraer los multiplicadores duales del cono
    dual_alpha = constraints[0].dual_value  # esta es la norma 2 ≤ eta

    # crear soluciones presentables

    # theta
    theta_opt = theta.value
    # theta_opt[np.abs(theta_opt)<tol]=0 #destruyo los 0 falsos

    # alpha
    alpha = dual_alpha[1]
    # print(np.linalg.norm(alpha)) #debemos ver si da menor que 1
    # alpha[np.abs(alpha)<tol]=0 #eliminar los 0 falsos.
    #print("el alfa es: ",alpha )
    print("norma de alfa es", np.linalg.norm(alpha))

    return theta_opt, eta.value, alpha, w_combo.value, prob.value, b.value, xi.value

#la version esparse del master primal. 

# En funciones_tesis_version_sparse.py
import cvxpy as cp
import numpy as np
import scipy.sparse as sp

def solve_master_primal_v2(X, y, K, tipo, C=1.0, tol=1e-6, 
                           mosek_params={}, M_box=1e4):
    """
    Versión consolidada y ROBUSTA del Problema Maestro.
    - Maneja una lista K con 'w' densos (1D) o esparsos (N, 1).
    - Añade Bounding Box (M_box) para 'w_combo'.
    - Acepta parámetros de MOSEK y usa Warm Start.
    - Calcula y retorna el gradiente correcto.
    """
    y_neg = np.where(y <= 0, -1, 1)
    K_len = len(K)
    n_samples = len(y)
    n_features = X.shape[1] # Obtener n_features desde X

    # --- Variables ---
    if tipo in ["afin", "libre"]:
        theta = cp.Variable((K_len, 1)) # Forma (K, 1) para consistencia
    elif tipo in ["convexo", "conico", "mayor_uno"]:
        theta = cp.Variable((K_len, 1), nonneg=True) # Forma (K, 1)
    else:
        print(f"Error: tipo '{tipo}' no es válido.")
        return None, None, None, None, None, None, None, None # Retornar Nones

    eta = cp.Variable()
    b = cp.Variable()
    xi = cp.Variable(n_samples, nonneg=True)

    # --- Precomputar Matriz W (MODIFICADO PARA ROBUSTEZ) ---
    sparse_Ws_list = []
    
    for k_idx, col in enumerate(K):
        w = col[0]
        
        if sp.issparse(w):
            # Ya es esparso, solo chequear forma
            if w.shape == (n_features, 1):
                sparse_Ws_list.append(w.astype(np.float32)) # Asegurar dtype
            elif w.shape == (1, n_features):
                # Es un vector fila esparso, transponerlo
                sparse_Ws_list.append(w.T.astype(np.float32))
            else:
                raise ValueError(f"Columna esparsa {k_idx} en K tiene forma incorrecta: {w.shape}. Se esperaba ({n_features}, 1)")
        
        elif isinstance(w, np.ndarray):
            # Es denso. Convertirlo a (N, 1) esparso.
            if w.ndim == 1:
                w_col = w.reshape(-1, 1) # Convertir 1D (N,) a 2D (N, 1)
            elif w.ndim == 2 and w.shape[1] == 1:
                w_col = w # Ya es (N, 1)
            elif w.ndim == 2 and w.shape[0] == 1:
                w_col = w.T # Es (1, N), transponer
            else:
                 raise ValueError(f"Array denso {k_idx} en K tiene forma incorrecta: {w.shape}")
            
            # Chequeo final de filas
            if w_col.shape[0] != n_features:
                raise ValueError(f"Columna {k_idx} tiene {w_col.shape[0]} filas, se esperaban {n_features}")

            sparse_Ws_list.append(sp.csc_matrix(w_col, dtype=np.float32))
        
        else:
            raise TypeError(f"Elemento w en K (índice {k_idx}) es de tipo desconocido: {type(w)}")

    # Ahora sparse_Ws_list SÓLO contiene matrices (n_features, 1)
    Ws_sparse = sp.hstack(sparse_Ws_list, format='csc')
    # --- FIN DE LA MODIFICACIÓN DE ROBUSTEZ ---

    # w_combo ahora es (N, 1) porque Ws_sparse es (N, K) y theta es (K, 1)
    w_combo = Ws_sparse @ theta   

    # --- Restricciones (MODIFICADO CON BBOX) ---
    # X@w_combo es (M, 1), b es escalar (broadcast), xi es (M,) -> error
    # Debemos asegurar que xi también sea (M, 1)
    xi = cp.Variable((n_samples, 1), nonneg=True) # Definir xi como (M, 1)
    y_neg_col = y_neg.reshape(-1, 1) # Asegurar que y sea (M, 1)

    constraints_dict = {
        "soc_norm": cp.SOC(eta, w_combo),
        # Ahora es (M, 1) >= (M, 1)
        "classification": cp.multiply(y_neg_col, X @ w_combo + b) >= 1 - xi, 
        # "slack_nonneg": xi >= 0, # Ya está en la definición de xi
        
        # --- CAJA AÑADIDA AL MASTER (NUEVO) ---
        "master_box_pos": w_combo <= M_box, 
        "master_box_neg": w_combo >= -M_box
    }
    
    # Restricciones de tipo de combinación (sin cambios)
    if tipo in ["convexo", "afin"]:
        constraints_dict["theta_sum"] = (cp.sum(theta) == 1)
    elif tipo == "mayor_uno":
        constraints_dict["theta_sum"] = (cp.sum(theta) >= 1)
    elif tipo == "conico":
        constraints_dict["theta_sum"] = (cp.sum(theta) >= 0)
    # 'libre' no añade nada

    # --- Objetivo (Sin cambios) ---
    objective = cp.Minimize(eta + C * (cp.sum(xi)))

    # --- Definición y Solución (MODIFICADO CON PARAMS Y WARM START) ---
    prob = cp.Problem(objective, list(constraints_dict.values()))
    
    prob.solve(solver=cp.MOSEK, mosek_params=mosek_params, warm_start=True) 
    
    # --- CÁLCULO DEL GRADIENTE (MODIFICADO CON FLATTEN) ---
    grad_w_correcto = np.zeros(n_features, dtype=np.float32) 
    alpha = None
    primal_value_UB = prob.value


    if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        try:
            alpha = constraints_dict["soc_norm"].dual_value[1]
            pi_dual = constraints_dict["classification"].dual_value

            if alpha is not None and pi_dual is not None:
                # pi_dual ahora es (M, 1), X.T es (N, M), y_neg es (M,)
                # Necesitamos pi_dual como (M,) para el producto
                sum_yxpi = X.T @ (y_neg * pi_dual.flatten())
                
                alpha_flat = alpha.flatten()
                sum_yxpi_flat = sum_yxpi.flatten()
                grad_w_correcto = alpha_flat + sum_yxpi_flat
            else:
                print("ADVERTENCIA: No se pudieron obtener las variables duales (valores None).")
        except Exception as e:
            print(f"ADVERTENCIA: Error al calcular el gradiente: {e}")
            alpha = None
    else:
        print(f"ADVERTENCIA: Master no resolvió óptimamente (status: {prob.status}). El gradiente puede no ser válido.")

    print(f"Norma del gradiente correcto para w: {np.linalg.norm(grad_w_correcto):.4f}")

    # --- ORDEN DE RETORNO (9 VALORES) ---
    # Aplanamos los valores de salida para que sean arrays 1D consistentes
    theta_val = theta.value.flatten() if theta.value is not None else None
    w_combo_val = w_combo.value.flatten() if w_combo.value is not None else None
    b_val = b.value if b.value is not None else None
    xi_val = xi.value.flatten() if xi.value is not None else None
    alpha_val = alpha.flatten() if alpha is not None else None

    return (theta_val, eta.value, alpha_val, w_combo_val, primal_value_UB, 
            b_val, xi_val, grad_w_correcto)

# Agrégalo a funciones_tesis_version_sparse.py

def is_column_in_hull(w_new_sparse, K_list_sparse, tipo="convexo", tol=1e-6):
    """
    Resuelve un problema de factibilidad para chequear si w_new_sparse
    está en el hull (convexo, cónico, etc.) de las columnas en K_list_sparse.
    
    Con convexo es revisar si es que existe en el K actual mezclando los otros K
    Con conico, es revisar si el subespacio vectorial generado por K incluye a w_new_sparse
    """
    
    
    K_len = len(K_list_sparse)
    
    if K_len == 0 or w_new_sparse is None or w_new_sparse.nnz == 0:
        return False # No puede estar en un hull vacío / columna nula

    # Apilar K en una gran matriz esparsa
    try:
        Ws_sparse = sp.hstack(K_list_sparse, format='csc')
    except ValueError as e:
        print(f"Error de forma en is_column_in_hull: {e}")
        return False # Asumir que no está

    theta = cp.Variable((K_len,1))
    
    # La columna w_new debe ser densa para el lado derecho de la restricción
    # Usamos .toarray() que es eficiente para una sola columna esparsa
    #w_new_dense = w_new_sparse.toarray()

    # Restricción de combinación
    constraints = [Ws_sparse @ theta == w_new_sparse]

    # Añadir restricciones del tipo de hull
    if tipo == "convexo":
        constraints += [cp.sum(theta) == 1, theta >= 0]
    elif tipo == "afin":
         constraints += [cp.sum(theta) == 1]
    elif tipo == "conico":
         constraints += [theta >= 0]
    # 'libre' no tiene restricciones extra

    # Problema de factibilidad
    prob = cp.Problem(cp.Minimize(0), constraints)
    
    # Usar un solver rápido y tolerancias relajadas
    # No necesitamos alta precisión, solo saber si es factible
    prob.solve(solver=cp.MOSEK, mosek_params={"MSK_DPAR_INTPNT_TOL_PFEAS": tol * 100, 
                                            "MSK_DPAR_INTPNT_TOL_DFEAS": tol * 100})
    
    if prob.status == cp.OPTIMAL:
        print("Chequeo de columna: Columna SÍ está en el hull.")
        return True
    else:
        print("Chequeo de columna: Columna NO está en el hull.")
        return False
# In[0]: En esta seccion estan todos los pricing que se pueden usar en este problema. Hay un caso importante eso si, algunos siempre dan unbounded.

#el caso que da unbounded es sospechoso. Se cree que es porque se crea una linea con manipulación de las restricciones de calculo de error cuando se retira la norma. 
# se debe lograr generar una combinacion LD de restricciones donde el vector b_i sea distinto, ahi se crea un espacio para que exista una linea donde haya decrecimiento marginal infinito. 



# inputs: DATA, variable dual ALPHA de la norma obtenida del master problem. Este es la relajacion lagrangeana del problema OG. 
# In[0]:
def rayo_surrogate(A, C, norm_cap=1.0, tol=1e-8, threads=None, verbose=False):
    """
    Construye una dirección 'r' que mantiene factibilidad al desplazarte (A r >= 0)
    y mejora el objetivo (maximiza C^T r) bajo la normalización ||r||_1 <= norm_cap.
    Si C^T r* > tol, úsalo como rayo para tu master (columna cónica con λ >= 0).
    """
    A = np.asarray(A, float)
    C = np.asarray(C, float).ravel()
    n_res, n_vars = A.shape

    r = cp.Variable(n_vars)
    cons = [
        A @ r >= 0,                       # no rompe factibilidad
        cp.norm2(r) <= float(norm_cap)    # normalización (acota el LP)
    ]
    obj = cp.Maximize(C @ r)
    prob = cp.Problem(obj, cons)

    mosek_params = {"MSK_IPAR_OPTIMIZER": 2}  # dual simplex para LP
    if threads is not None:
        mosek_params["MSK_IPAR_NUM_THREADS"] = int(threads)

    prob.solve(solver=cp.MOSEK, mosek_params=mosek_params, warm_start=True, verbose=verbose)
    print(r.value)
    status = (prob.status or "").lower()
    if status.startswith("optimal") and prob.value is not None and prob.value > tol:
        return {"status": "ray_found", "r": r.value, "gain": float(prob.value),"r_norm":r.value/np.linalg.norm(r.value)}
    return {"status": status, "r": None, "gain": None}
    
def funcion_esquina(C, A, b, *, optimizer="dual", tol=1e-8, threads=None, verbose=False):
    """
    Max: C^T x
    s.a. A x >= b
    
    - Fuerza MOSEK a usar SIMPLEX (primal o dual).
    - Devuelve dict con status, x, obj, y_dual (si hay).
    - Si el estado es 'unbounded', no hay rayo de solver (limitación de CVXPY),
      pero puedes usar la función 'rayo_surrogate' de más abajo para una dirección útil.
    """
    A = np.asarray(A, float)
    b = np.asarray(b, float).ravel()
    C = np.asarray(C, float).ravel()
    n_res, n_vars = A.shape

    # Variable
    x = cp.Variable(n_vars)

    # Restricción (forma >=)
    cons = [A @ x >= b]

    # Objetivo
    obj = cp.Maximize(C @ x)

    prob = cp.Problem(obj, cons)

    # Elegir optimizador de MOSEK:
    # 1 = primal simplex, 2 = dual simplex, 3 = interior-point (NO queremos esto aquí)
    opt_map = {"primal": 1, "dual": 2, "ipm": 3}
    opt_code = opt_map.get(optimizer, 2)

    mosek_params = {
        "MSK_IPAR_OPTIMIZER": opt_code,           # simplex por defecto (dual)
        "MSK_DPAR_INTPNT_TOL_REL_GAP": float(tol) # no afecta simplex, pero inofensivo
    }
    if threads is not None:
        mosek_params["MSK_IPAR_NUM_THREADS"] = int(threads)

    prob.solve(solver=cp.MOSEK, mosek_params=mosek_params, warm_start=True, verbose=verbose)

    status = (prob.status or "").lower()  # 'optimal', 'infeasible', 'unbounded', etc.

    # Nota: los nombres válidos en CVXPY son SIN espacios y en minúsculas.
    if status == "optimal" or status == "optimal_inaccurate":
        # solución primal
        x_val = x.value
        obj_val = float(prob.value)
        # duales de A x >= b (siempre >=0 en LP dual)
        y_dual = cons[0].dual_value  # puede ser None en algunos estados inexactos
        return {
            "status": status,
            "x": x_val,
            "obj": obj_val,
            "y_dual": y_dual,
        }

    if "unbounded" in status:
        # CVXPY NO entrega rayo del solver; devuelve solo el estado.
        # Usa rayo_surrogate(...) (abajo) para construir una dirección útil.
        return {"status": status, "x": None, "obj": None, "y_dual": None}

    if "infeasible" in status:
        # Sin certificado Farkas vía CVXPY (limitación de la interfaz).
        return {"status": status, "x": None, "obj": None, "y_dual": None}

    # Otros estados (p.ej. 'unknown')
    return {"status": status, "x": None, "obj": None, "y_dual": None}






# In[0]:


def solve_pricing_problem_original(X, y, alpha, C=1.0): 

    #print("‖alpha‖:", np.linalg.norm(alpha))
    #print("alpha min/max:", np.min(alpha), np.max(alpha))

    n_samples, n_features = X.shape
    y_neg = np.where(y <= 0, -1, 1)  # usualmente o son 0, o son 1.

    # Variables
    w = cp.Variable(n_features)
    b = cp.Variable()
    xi = cp.Variable(n_samples, nonneg=True)
    alpha = alpha.flatten()

    # Constraints
    constraints = [
        cp.multiply(y_neg, X @ w + b) >= 1 - xi,
        xi >= 0,

    ]

    # Objective: tau + C * sum(xi)
    # cp.Minimize(0) #+ |alpha @ w|
    objective = cp.Minimize(C * cp.sum(xi) - alpha @ w)
    # Solve
    prob_pricing = cp.Problem(objective, constraints)
    prob_pricing.solve(
        solver=cp.MOSEK
    )
    # asegurarse que el W sea el W correcto. #Esto de aqui sesga mucho el resultado, no puede generar nuevas columnas y se traba, Revisar.
#    w_pricing_1_clean = w.value.copy()
#    mask_keep = (w_pricing_1_clean >= M) | (w_pricing_1_clean <= M)
#    w_pricing_1_clean[~mask_keep] = 0
#    mask_alpha_nonzero = np.abs(alpha) > 0
#    w_pricing_1_masked[~mask_alpha_nonzero] = 0
    print("el valor objetivo es", prob_pricing.value)
    print("el valor de la norma de w es", np.linalg.norm(w.value))
    print("valor de F.O calculada es", np.linalg.norm(
        w.value) + C*np.sum(xi.value))

    # w_pricing_1_masked #, prob_pricing.value, prob_pricing.status
    return (w.value, b.value, xi.value)




def solve_pricing_problem_restricto(X, y, alpha, C=1.0):  # sin la fo cambiada 

    #print("‖alpha‖:", np.linalg.norm(alpha))
    #print("alpha min/max:", np.min(alpha), np.max(alpha))

    n_samples, n_features = X.shape
    y_neg = np.where(y <= 0, -1, 1)  # usualmente o son 0, o son 1.

    # Variables
    w = cp.Variable(n_features)
    b = cp.Variable()
    xi = cp.Variable(n_samples, nonneg=True)
    alpha = alpha.flatten()

    # Constraints
    constraints = [
        cp.multiply(y_neg, X @ w + b) >= 1 - xi,
        xi >= 0,
        # es menos el gradiente de la aproximacion de la norma con alfa, por el vector w
        -alpha @ w >= 0
    ]

    objective = cp.Minimize(C * cp.sum(xi))  # sin alfa en la FO
    # Solve
    prob_pricing = cp.Problem(objective, constraints)
    prob_pricing.solve(
        solver=cp.MOSEK
    )
    print("el valor objetivo es", prob_pricing.value)
    print("el valor de la norma de w es", np.linalg.norm(w.value))
    print("valor de F.O calculada es", np.linalg.norm(
        w.value) + C*np.sum(xi.value))

    # w_pricing_1_masked #, prob_pricing.value, prob_pricing.status
    return (w.value, b.value, xi.value)


def solve_pricing_problem_restricto_componente(X, y, alpha, C=1.0): 

    #print("‖alpha‖:", np.linalg.norm(alpha))
    #print("alpha min/max:", np.min(alpha), np.max(alpha))

    n_samples, n_features = X.shape
    y_neg = np.where(y <= 0, -1, 1)  # usualmente o son 0, o son 1.

    # Variables
    w = cp.Variable(n_features)
    b = cp.Variable()
    xi = cp.Variable(n_samples, nonneg=True)
    alpha = alpha.flatten()

    # Constraints
    constraints = [
        cp.multiply(y_neg, X @ w + b) >= 1 - xi,
        xi >= 0,
        # un poco mas apretado que el de arriba, pero logra guiar hacia el 0.
        cp.multiply(-alpha, w) >= 0
    ]

    # Objective: tau + C * sum(xi)
    # cp.Minimize(0) #+ alpha @ w
    objective = cp.Minimize(C * cp.sum(xi) - alpha @ w)
    # Solve
    prob_pricing = cp.Problem(objective, constraints)
    prob_pricing.solve(
        solver=cp.MOSEK
    )
    # asegurarse que el W sea el W correcto. #Esto de aqui sesga mucho el resultado, no puede generar nuevas columnas y se traba, Revisar.
#    w_pricing_1_clean = w.value.copy()
#    mask_keep = (w_pricing_1_clean >= M) | (w_pricing_1_clean <= M)
#    w_pricing_1_clean[~mask_keep] = 0
#    mask_alpha_nonzero = np.abs(alpha) > 0
#    w_pricing_1_masked[~mask_alpha_nonzero] = 0
    print("el valor objetivo es", prob_pricing.value)
    print("el valor de la norma de w es", np.linalg.norm(w.value))
    print("valor de F.O calculada es", np.linalg.norm(
        w.value) + C*np.sum(xi.value))

    # w_pricing_1_masked #, prob_pricing.value, prob_pricing.status
    return (w.value, b.value, xi.value)

# antes era el solve pricing 2 (el bueno)
def solve_pricing_problem_caja(X, y, alpha, K=None, C=1.0,PRICING_PARAMS= {},M_box=1e4): 
    
    #print("‖alpha‖:", np.linalg.norm(alpha))
    #print("alpha min/max:", np.min(alpha), np.max(alpha))
    n_samples, n_features = X.shape
    y_neg = np.where(y <= 0, -1, 1)  # usualmente o son 0, o son 1.

    # Variables
    w = cp.Variable(n_features)
    b = cp.Variable()
    xi = cp.Variable(n_samples, nonneg=True)
    alpha = alpha.flatten()

    # Constraints
    constraints = [
        cp.multiply(y_neg, X @ w + b) >= 1 - xi,
        xi >= 0,
      #  -alpha @ w >= 0,  # con esta busca asegurarse de guiar bien la F.O
        w <= M_box,
        w >= -M_box
    ]

    # Objective: tau + C * sum(xi)
    # cp.Minimize(0) #- alpha @ w #esta de aca esta asignando todas a xi=0. es pesima. DEBO DARLE PRIORIDAD AL W.
    objective = cp.Minimize(C * cp.sum(xi) - alpha @ w)
    # Solve
    prob_pricing = cp.Problem(objective, constraints)
    prob_pricing.solve( 
        solver=cp.MOSEK,mosek_params=PRICING_PARAMS,warm_start=True, # HABILITAR WARM START
        verbose=True
    )## warm_start=True, #si la FO cambia mucho, quizas no es buena idea. 
    
    
    
    # --- LÓGICA DE FALLBACK AÑADIDA ---
    st = (prob_pricing.status or "").lower()
    ok_opt = st.startswith("optimal")
    ok_has_vals = (w.value is not None) and (b.value is not None) and (xi.value is not None)
    
    # CASO 1: El solver encontró una solución utilizable
    if ok_opt or ok_has_vals:
        if not ok_opt:
            print(f"[pricing] status={prob_pricing.status} -> guardo solución factible pero no óptima")
        
        try:
            print("Valor objetivo del pricing:", float(prob_pricing.value))
            print("Norma de w generada:", float(np.linalg.norm(w.value)))
        except Exception:
            pass # Evitar que falle si los valores son None a pesar de ok_has_vals
        # w.value es (N,), lo pasamos a (N, 1) y luego a sparse
        w_sparse = sp.csc_matrix(w.value.reshape(-1, 1), dtype=np.float32)
        
        # xi.value es (M,), lo pasamos a (M, 1) y luego a sparse
        xi_sparse = sp.csc_matrix(xi.value.reshape(-1, 1), dtype=np.float32)    
        return (w_sparse, b.value, xi_sparse)

    # CASO 2: El solver falló, activamos el Plan B
    else:
        print(f"[pricing] status={prob_pricing.status} sin solución usable -> Activando fallback heurístico.")
        
        # El fallback necesita K para no agregar duplicados
        if K is None:
            print("ADVERTENCIA: Fallback necesita el conjunto K, pero K es None. No se pueden generar columnas.")
            return (None, None, None) # Retorna una señal de fallo claro

    # --- CAMBIO A SPARSE ---
        # El fallback ahora debe generar columnas canónicas esparsas
        k_max = int(n_features / 100)
        
        # Pasamos n_features y n_samples para los nuevos helpers esparsos
        K_actualizado, info = generar_set_columnas_costos_reducidos_sparse(
            X, y, alpha, K, 
            n_features=n_features, 
            n_samples=n_samples, 
            eps=1e-8, k_max=k_max
        )
        return K_actualizado
def solve_pricing_problem_combinado_sumar_restricto_2(X, y, alpha, K=None, C=1.0,PRICING_PARAMS= {    "MSK_IPAR_PRESOLVE_USE": 1,    "MSK_IPAR_PRESOLVE_ELIMINATOR_MAX_NUM_TRIES": 0,  "MSK_IPAR_PRESOLVE_LINDEP_USE": 0},M_box=1e4): 
    
    #print("‖alpha‖:", np.linalg.norm(alpha))
    #print("alpha min/max:", np.min(alpha), np.max(alpha))
    n_samples, n_features = X.shape
    y_neg = np.where(y <= 0, -1, 1)  # usualmente o son 0, o son 1.

    # Variables
    w = cp.Variable(n_features)
    b = cp.Variable()
    xi = cp.Variable(n_samples, nonneg=True)
    alpha = alpha.flatten()

    # Constraints
    constraints = [
        cp.multiply(y_neg, X @ w + b) >= 1 - xi,
        xi >= 0,
        -alpha @ w >= 0,  # con esta busca asegurarse de guiar bien la F.O
        w <= M_box,
        w >= -M_box
    ]

    # Objective: tau + C * sum(xi)
    # cp.Minimize(0) #- alpha @ w #esta de aca esta asignando todas a xi=0. es pesima. DEBO DARLE PRIORIDAD AL W.
    objective = cp.Minimize(C * cp.sum(xi) - alpha @ w)
    # Solve
    prob_pricing = cp.Problem(objective, constraints)
    prob_pricing.solve( 
        solver=cp.MOSEK,mosek_params=PRICING_PARAMS,warm_start=True, # HABILITAR WARM START
        verbose=True
    )## warm_start=True, #si la FO cambia mucho, quizas no es buena idea. 
    
    
    
    # --- LÓGICA DE FALLBACK AÑADIDA ---
    st = (prob_pricing.status or "").lower()
    ok_opt = st.startswith("optimal")
    ok_has_vals = (w.value is not None) and (b.value is not None) and (xi.value is not None)
    
    # CASO 1: El solver encontró una solución utilizable
    if ok_opt or ok_has_vals:
        if not ok_opt:
            print(f"[pricing] status={prob_pricing.status} -> guardo solución factible pero no óptima")
        
        try:
            print("Valor objetivo del pricing:", float(prob_pricing.value))
            print("Norma de w generada:", float(np.linalg.norm(w.value)))
        except Exception:
            pass # Evitar que falle si los valores son None a pesar de ok_has_vals
        # w.value es (N,), lo pasamos a (N, 1) y luego a sparse
        w_sparse = sp.csc_matrix(w.value.reshape(-1, 1), dtype=np.float32)
        
        # xi.value es (M,), lo pasamos a (M, 1) y luego a sparse
        xi_sparse = sp.csc_matrix(xi.value.reshape(-1, 1), dtype=np.float32)    
        return (w_sparse, b.value, xi_sparse)

    # CASO 2: El solver falló, activamos el Plan B
    else:
        print(f"[pricing] status={prob_pricing.status} sin solución usable -> Activando fallback heurístico.")
        
        # El fallback necesita K para no agregar duplicados
        if K is None:
            print("ADVERTENCIA: Fallback necesita el conjunto K, pero K es None. No se pueden generar columnas.")
            return (None, None, None) # Retorna una señal de fallo claro

    # --- CAMBIO A SPARSE ---
        # El fallback ahora debe generar columnas canónicas esparsas
        k_max = int(n_features / 100)
        
        # Pasamos n_features y n_samples para los nuevos helpers esparsos
        K_actualizado, info = generar_set_columnas_costos_reducidos_sparse(
            X, y, alpha, K, 
            n_features=n_features, 
            n_samples=n_samples, 
            eps=1e-8, k_max=k_max
        )
        return K_actualizado

# parece que esta es la mejor
def solve_pricing_problem_combinado_componente(X, y, alpha,K, C=1.0,PRICING_PARAMS= {    "MSK_IPAR_PRESOLVE_USE": 1,    "MSK_IPAR_PRESOLVE_ELIMINATOR_MAX_NUM_TRIES": 0,  "MSK_IPAR_PRESOLVE_LINDEP_USE": 0},M_box=None):
    
    #print("‖alpha‖:", np.linalg.norm(alpha))
    #print("alpha min/max:", np.min(alpha), np.max(alpha))
    n_samples, n_features = X.shape
    y_neg = np.where(y <= 0, -1, 1)  # usualmente o son 0, o son 1.

    # Variables
    w = cp.Variable(n_features)
    b = cp.Variable()
    xi = cp.Variable(n_samples, nonneg=True)
    
    #parametro
    alpha = alpha.flatten()

    # Constraints
    constraints_2 = [
        cp.multiply(y_neg, X @ w + b) >= 1 - xi,
        xi >= 0,-alpha @ w >= 0
        
    ]# cp.multiply(w, -alpha) >= 0

    # Objective: tau + C * sum(xi)
    # cp.Minimize(0) #- alpha @ w #esta de aca esta asignando todas a xi=0. es pesima. DEBO DARLE PRIORIDAD AL W.
    objective_2 = cp.Minimize(C * cp.sum(xi)-alpha @ w)
    # Solve
    prob_pricing = cp.Problem(objective_2, constraints_2)
    prob_pricing.solve( 
        solver=cp.MOSEK,mosek_params=PRICING_PARAMS,warm_start=True, # HABILITAR WARM START
        verbose=True
    )## warm_start=True, #si la FO cambia mucho, quizas no es buena idea. 
    
    
    # --- LÓGICA DE FALLBACK AÑADIDA ---
    st = (prob_pricing.status or "").lower()
    ok_opt = st.startswith("optimal")
    ok_has_vals = (w.value is not None) and (b.value is not None) and (xi.value is not None)
    
    # CASO 1: El solver encontró una solución utilizable
    if ok_opt or ok_has_vals:
        if not ok_opt:
            print(f"[pricing] status={prob_pricing.status} -> guardo solución factible pero no óptima")
        
        try:
            print("Valor objetivo del pricing:", float(prob_pricing.value))
            print("Norma de w generada:", float(np.linalg.norm(w.value)))
        except Exception:
            pass # Evitar que falle si los valores son None a pesar de ok_has_vals
        # w.value es (N,), lo pasamos a (N, 1) y luego a sparse
        w_sparse = sp.csc_matrix(w.value.reshape(-1, 1), dtype=np.float32)
        
        # xi.value es (M,), lo pasamos a (M, 1) y luego a sparse
        xi_sparse = sp.csc_matrix(xi.value.reshape(-1, 1), dtype=np.float32)    
        return (w_sparse, b.value, xi_sparse)

    # CASO 2: El solver falló, activamos el Plan B
    else:
        print(f"[pricing] status={prob_pricing.status} sin solución usable -> Activando fallback heurístico.")
        
        # El fallback necesita K para no agregar duplicados
        if K is None:
            print("ADVERTENCIA: Fallback necesita el conjunto K, pero K es None. No se pueden generar columnas.")
            return (None, None, None) # Retorna una señal de fallo claro

    # --- CAMBIO A SPARSE ---
        # El fallback ahora debe generar columnas canónicas esparsas
        k_max = int(n_features / 100)
        
        # Pasamos n_features y n_samples para los nuevos helpers esparsos
        K_actualizado, info = generar_set_columnas_costos_reducidos_sparse(
            X, y, alpha, K, 
            n_features=n_features, 
            n_samples=n_samples, 
            eps=1e-8, k_max=k_max
        )
        return K_actualizado
        
   

    # , prob_pricing.value, prob_pricing.status #el VALOR OPTIMO ES EL MEJOR (4) EL ALPHA QUE ENTREGA ES PESIMO, DESTRUYENDO SIGUIENTE ITERACION PRICING. sera ruido?
    return (w.value, b.value, xi.value)


def solve_pricing_problem_signo(X, y, alpha, C=1.0):

    #print("‖alpha‖:", np.linalg.norm(alpha))
    #print("alpha min/max:", np.min(alpha), np.max(alpha))

    n_samples, n_features = X.shape
    y_neg = np.where(y <= 0, -1, 1)  # usualmente o son 0, o son 1.

    # Variables
    w = cp.Variable(n_features)
    b = cp.Variable()
    xi = cp.Variable(n_samples, nonneg=True)
    alpha = alpha.flatten()

    # Constraints
    constraints = [
        cp.multiply(y_neg, X @ w + b) >= 1 - xi,
        xi >= 0,
        # si alfa pos, no puede hacer que w sea neg #necesito producto component wise, no producto punto.
        cp.multiply(w, -alpha) >= 0
    ]

    # Objective: tau + C * sum(xi)
    # cp.Minimize(0) #+ alpha @ w
    objective = cp.Minimize(C * cp.sum(xi) - alpha @ w)
    # Solve
    prob_pricing = cp.Problem(objective, constraints)
    prob_pricing.solve( 
        solver=cp.MOSEK,mosek_params=PRICING_PARAMS,warm_start=True, # HABILITAR WARM START
        verbose=True
    )## warm_start=True, #si la FO cambia mucho, quizas no es buena idea. 
    # asegurarse que el W sea el W correcto. #Esto de aqui sesga mucho el resultado, no puede generar nuevas columnas y se traba, Revisar.
#    w_pricing_1_clean = w.value.copy()
#    mask_keep = (w_pricing_1_clean >= M) | (w_pricing_1_clean <= M)
#    w_pricing_1_clean[~mask_keep] = 0
#    mask_alpha_nonzero = np.abs(alpha) > 0
#    w_pricing_1_masked[~mask_alpha_nonzero] = 0
    print("el valor objetivo es", prob_pricing.value)
    print("el valor de la norma de w es", np.linalg.norm(w.value))
    print("valor de F.O calculada es", np.linalg.norm(
        w.value) + C*np.sum(xi.value))

    # w_pricing_1_masked #, prob_pricing.value, prob_pricing.status
    return (w.value, b.value, xi.value)


#[solve_pricing_problem_restricto_componente, solve_pricing_problem_restricto],
#[solve_pricing_problem_original, solve_pricing_problem_combinado_sumar_restricto, solve_pricing_problem_combinado_componente, solve_pricing_problem_signo]

def solo_los_w(K):
    return list(set(tuple(row) for row in np.array(K, object).T[0]))


def esta_ki_en_K(ki, K):
    if len(ki) == 0 or len(K) == 0:
        return False
    else:
        return tuple(ki) in solo_los_w(K)

# X_train_ct=cvect.transform(X_train) # (13854, 2264923)
# X_test_ct = cvect.transform(X_test) #(3464, 2264923)


# K es una lista de listas de tamaño 3 que guardan los vectores [w,b,xi] #pricing una ffuncion de las de arriba 

#column generation detallado esta como obsoleto, debe ser mejorado.

def Column_generation_detallado(X, y, K, tipo, pricing, C=1, tol=1e-6, M=1e3):

    time_ini = time.time()
    K_fin = K.copy()
    K_generados = []
    alpha_set_fin = [M*M]
    opt_val_fin = [M*M]
    theta_iter = []
    i = 0

    while True:
        resultados_master_ = solve_master_primal(X, y, K_fin, tipo, C, tol)
        print("***"*10)
        print("el valor optimo actual del master es ", resultados_master_[-1])
        print("***"*10)
        # valor optimo iteraccion actual
        opt_val_fin.append(resultados_master_[-1])
        # que vector alfa se genero
        alpha_set_fin.append(resultados_master_[2])
        theta_iter.append(resultados_master_[0])  # que columnas se usaron
        if np.linalg.norm(opt_val_fin[-1]-opt_val_fin[-2]) <= tol:
            print("la diferencia de las fo fue", np.linalg.norm(
                opt_val_fin[-2]-opt_val_fin[-1]))  # la diferencia de las F.O

            print("****"*10, " fin ", "****"*8)
            break
        # usaremos el alfa para generar un nuevo vector para K.
        ki = pricing(X, y, resultados_master_[2], C)
        if esta_ki_en_K(ki[0], K_fin) == True:
            print("columna ya generada anteriormente ")
            print("****"*10, " fin ", "****"*8)
            break
        if i > M:
            print("Demasiadas iteraciones")
            print("****"*10, " fin ", "****"*8)
            break
        i = i+1
        K_fin.append(ki)
        K_generados.append(ki)
    time_fin = time.time()
    print("se demopro estos minuto:", (time_fin-time_ini)/60)
    return opt_val_fin, alpha_set_fin, K_generados, K_fin, theta_iter

# desde aca son cosas para que funcione el metodo 


def predict_t(b, w, x_test):
    estimate = x_test.dot(w) + b
    prediction = np.sign(estimate)
    return prediction  # np.where(prediction == -1, 0, 1)
#y_pred_f=predict_t(k_fin[500][1], k_fin[500][0], X_test_ct)


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# Sugerencia para hacer generar_canonico más robusto
def generar_canonico(w, coordenada=0, random=False, n_samples=None):
    temp = np.zeros(len(w), dtype=np.float32)
    # Para el xi, si no tenemos info, devolvemos None o un placeholder numérico
    xi_placeholder = np.zeros(n_samples, dtype=np.float32) if n_samples is not None else None

    if coordenada == -1:
        return (np.array(temp), 0.0, xi_placeholder) # Usar 0.0 para b
    # ... resto de la lógica ...
    else:
        temp[coordenada] = 1
        return (np.array(temp), 0.0, xi_placeholder)
    
    
#Parche Sparse
def generar_canonico_sparse(n_features, coordenada, n_samples, dtype=np.float32):
    """
    Genera un vector canónico w como una matriz sparse CSC (n_features x 1).
    n_features: La longitud del vector w (int).
    coordenada: El índice del elemento no-cero (-1 para vector cero).
    n_samples: Número de muestras para el placeholder xi.
    dtype: Tipo de dato.
    """
    # Placeholder para xi, también sparse
    xi_placeholder = sp.csc_matrix((n_samples, 1), dtype=dtype) if n_samples is not None else None

    if coordenada == -1: # Vector cero
        w_sparse = sp.csc_matrix((n_features, 1), dtype=dtype)
        return (w_sparse, 0.0, xi_placeholder)
    else:
        # Vector canónico (un solo no-cero)
        w_sparse = sp.csc_matrix(
            (np.array([1.0], dtype=dtype), (np.array([int(coordenada)]), np.array([0]))),
            shape=(n_features, 1),
            dtype=dtype
        )
        return (w_sparse, 0.0, xi_placeholder)
    
def generar_K_canonico_sparse(n_features, n_samples, tamaño=0.1):
    """
    Genera un set K inicial con vectores canónicos ESPARSOS.
    n_features: La longitud del vector w (int).
    n_samples: El número de filas (int) para el placeholder xi.
    tamaño: Proporción de coordenadas a incluir (0.0 a 1.0).
    """
    np.random.seed(420)
    
    # Incluir el vector cero sparse
    K_generado = [generar_canonico_sparse(n_features, coordenada=-1, n_samples=n_samples)] 
    print("0 generado")
    num_canonicos = int(n_features * tamaño)
    if num_canonicos >= n_features:
        columnas = np.arange(n_features)
    else:
        columnas = np.random.choice(n_features, num_canonicos, replace=False)
        
    print(f"Generando {len(columnas)} vectores canónicos sparse...")
    for i in columnas:
        K_generado.append(generar_canonico_sparse(n_features, coordenada=i, n_samples=n_samples))
        
    print("Set K canónico sparse generado.")
    return K_generado

def generar_K_canonico(w, tamaño=0.1):  # cantidad de features #debe haber una manera mas eficiente de hacer esto. 
    np.random.seed(420)  # fija la aleatoriedad!!!
    K_generado = [generar_canonico(w, coordenada=-1)]
    columnas = np.random.choice(len(w), int(len(w)*tamaño), replace=False)
    for i in columnas:
        K_generado.append(generar_canonico(w, coordenada=i, random=False))
    return K_generado

def generar_canonico_con_signo(w_proto, j, signo=+1.0):
    """
    Usa tu generar_canonico y aplica signo ±1 a la coord. j.
    Devuelve (w, ['b'], ['xi']) en float32.
    """
    w_pos, b_tag, xi_tag = generar_canonico(w_proto, coordenada=int(j), random=False)
    w_pos = np.asarray(w_pos, dtype=np.float32)
    w_pos[int(j)] *= float(np.sign(signo))
    return (w_pos, b_tag, xi_tag)

def generar_canonico_con_signo_sparse(n_features, n_samples, j, signo=+1.0):
    """
    Usa generar_canonico_sparse y aplica el signo.
    Retorna (w_sparse, b, xi_sparse) en float32.
    """
    # Llama al nuevo helper esparso
    (w_sparse, b_tag, xi_sparse) = generar_canonico_sparse(
        n_features=n_features, 
        coordenada=int(j), 
        n_samples=n_samples, 
        dtype=np.float32
    )
    
    # Modificar el dato del vector esparso
    if w_sparse.nnz > 0: # Si no es el vector cero
        w_sparse.data[0] = float(np.sign(signo)) # data[0] es el valor '1.0'
        
    return (w_sparse, b_tag, xi_sparse)

def pares_en_K(K): #son los canonicos
    """
    Extrae pares (j, sign) que ya existen en K, para evitar duplicados exactos.
    Soporta K como list o dict con valores tipo (w, ["b"], ["xi"]).
    """
    ya = set()
    # Permite dict (keys 'K_1', etc.) o list de tuplas
    it = K.values() if isinstance(K, dict) else K
    for tpl in it:
        w = np.asarray(tpl[0]).ravel()
        nz = np.flatnonzero(w)
        if nz.size:                           # canónicas: usualmente 1 no-cero
            j = int(nz[0])
            sg = int(np.sign(w[j]))   or 1   # +1 o -1
            ya.add((j, sg))
    return ya
def pares_en_K_sparse(K):
    """
    Extrae pares (j, sign) de un K que contiene w esparsos.
    """
    ya = set()
    it = K.values() if isinstance(K, dict) else K
    
    for tpl in it:
        w_sparse = tpl[0] # w_sparse es una csc_matrix (n_features x 1)
        
        if w_sparse.nnz == 1: # Si es un vector canónico (solo 1 no-cero)
            j = int(w_sparse.indices[0]) # El índice de la fila del no-cero
            sg = int(np.sign(w_sparse.data[0])) or 1 # El signo del valor
            ya.add((j, sg))
        elif w_sparse.nnz == 0: # Vector cero
            pass # No es un par (j, sign)
        else:
            # Es una columna densa (de un pricing) o un subproblema
            # No lo contamos como un par canónico
            pass 
    return ya

def screen_from_dual_inequalities(X, alpha, Y, *, eps=1e-8, k_max=50, already=None): #COSTOS REDUCIDOS AAAAAAAAAAAAAAAAAA
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

    # Asegurar tipos/formas
    a  = np.asarray(alpha, float).ravel()
    Yv = np.asarray(Y,     float).ravel()

    # s = X^T Y (vector de tamaño n_features). Si X es sparse, .dot usa rutina eficiente.
    s = X.T.dot(Yv) if hasattr(X, "dot") else X.T @ Yv
    s = np.asarray(s, float).ravel()

    # Desigualdades (deben ser >= 0)
    v1 = -a - s #>=0 para que sea opt entonces <0 para que no. 
    v2 =  a + s #>=0 entonces <0 para que se agregue
    sumY = float(np.sum(Yv))
    v3 = -sumY
    v4 = sumY

    # Violaciones estrictas por debajo de -eps (eps ~ tol para ignorar ruido numérico) #este es el 0
    viol1 = v1 < 0-eps    # ⇔ a + s > eps ⇒ favorece usar e_j
    viol2 = v2 < 0-eps     # ⇔ a + s < -eps ⇒ favorece usar -e_j

    # Dedupe: si ya trae pares (j,sign), bloquea solo ese signo de ese índice
    if already:
        mask1 = np.ones_like(a, dtype=bool)  # v1 (+e_j)
        mask2 = np.ones_like(a, dtype=bool)  # v2 (-e_j)
        if any(isinstance(t, tuple) and len(t) == 2 for t in already):
            ban_pos = {j for (j,sg) in already if int(sg) == +1}
            ban_neg = {j for (j,sg) in already if int(sg) == -1}
            if ban_pos:
                mask1[np.fromiter(ban_pos, dtype=int)] = False
            if ban_neg:
                mask2[np.fromiter(ban_neg, dtype=int)] = False
        else:
            # Si ‘already’ fueran solo índices, bloquearía ambos signos de esos índices
            ban = np.fromiter(already, dtype=int)
            mask1[ban] = False
            mask2[ban] = False
        viol1 &= mask1
        viol2 &= mask2

    # Severidad positiva de la violación (qué tanto “rompe” la desigualdad)
    sev1 = -(v1[viol1])    # = (a + s)[viol1]  > 0
    sev2 = -(v2[viol2])    # = -(a + s)[viol2] > 0

    # Índices de columnas que violan v1 o v2
    idx1 = np.flatnonzero(viol1)
    idx2 = np.flatnonzero(viol2)

    # Concateno todo para seleccionar top-K por severidad sin ordenar todo el vector
    idx_all  = np.concatenate([idx1, idx2])
    sev_all  = np.concatenate([sev1, sev2])

    # Signo recomendado para la columna canónica:
    #  - si viola v1 ⇒ a + s > 0 ⇒ conviene -e_j (sign = -1)
    #  - si viola v2 ⇒ a + s < 0 ⇒ conviene +e_j (sign = +1)
    sign_all = np.concatenate([
        -np.ones_like(idx1, float),
        +np.ones_like(idx2, float)
    ])

    # Si no hay violaciones, no hay columnas a proponer
    if sev_all.size == 0:
        return (np.array([], dtype=int),
                np.array([], dtype=float),
                np.array([], dtype=float),
                {"sumY": sumY, "b_suggest": 0, "n_viol1": 0, "n_viol2": 0})

    # Selección top-K eficiente:
    # np.argpartition toma los K más grandes sin ordenar todo (O(n)), luego ordeno esos K. #REGLA DE BLAND LESSGOOOOOO
    k = int(min(k_max, sev_all.size))
    top_idx_local = np.argpartition(sev_all, -k)[-k:]
    order = np.argsort(sev_all[top_idx_local])[::-1]  # orden descendente por severidad
    sel = top_idx_local[order]

    # Salidas seleccionadas
    idx  = idx_all[sel]     # índices de features
    sign = sign_all[sel]    # +1: +e_j,  -1: -e_j
    sev  = sev_all[sel]     # severidad de violación

    # Sugerencia para sesgo 'b' usando v3/v4 (1^T Y):
    #  - si v3 < -eps (sumY < 0), sugiere +b
    #  - si v4 < -eps (sumY > 0), sugiere -b
    b_suggest = 0
    if v3 < -eps:   b_suggest = +1
    elif v4 < -eps: b_suggest = -1

    meta = {
        "sumY": sumY,
        "b_suggest": b_suggest,          # {+1, 0, -1}
        "n_viol1": int(viol1.sum()),
        "n_viol2": int(viol2.sum()),
    }
    return idx, sign, sev, meta

def generar_set_columnas_costos_reducidos(
    X, y, alpha, K, eps=1e-8, k_max=25, w_proto=None, debug=False):
    """
    Usa el screening anterior y agrega hasta k_max canónicas ±e_j a K,
    evitando duplicados por (j,sign). Mantiene claves 'k_{i}' en minúsculas
    cuando K es dict. Devuelve (K_actualizado, info).
    """
    a  = np.asarray(alpha, float).ravel()
    yv = np.asarray(y,     float).ravel()

    # pares ya presentes (j,sign)
    already_pairs = pares_en_K(K)  # debe devolver set((j,sign))

    # screening (respeta 'already_pairs')
    idx, sign, sev, meta = screen_from_dual_inequalities(
        X, a, yv, eps=eps, k_max=k_max, already=already_pairs
    )

    if idx.size == 0:
        info = {"agregadas": 0, "idx": idx, "sign": sign, "sev": sev, "meta": meta}
        return K, info

    # seguridad extra: evita duplicados dentro del mismo batch
    unique_idx, unique_sign, unique_sev = [], [], []
    seen_batch = set()
    for j, s, v in zip(idx, sign, sev):
        key = (int(j), int(np.sign(s)) or 1)
        if key in already_pairs or key in seen_batch:
            continue
        seen_batch.add(key)
        unique_idx.append(int(j))
        unique_sign.append(int(np.sign(s)) or 1)
        unique_sev.append(float(v))

    if not unique_idx:
        info = {"agregadas": 0,
                "idx": np.array([], dtype=int),
                "sign": np.array([], dtype=float),
                "sev": np.array([], dtype=float),
                "meta": meta}
        return K, info

    if w_proto is None:
        w_proto = np.zeros(X.shape[1], dtype=np.float32)

    # construir columnas con tu helper
    nuevas = [generar_canonico_con_signo(w_proto, j, s)
              for j, s in zip(unique_idx, unique_sign)]

    # insertar en K respetando 'k_{i}' minúscula cuando es dict
    agregadas = 0
    if isinstance(K, dict):
        # siguiente índice libre (usa patrón k_### si existe)
        try:
            next_idx = max(int(str(k).split('_')[1]) for k in K.keys() if '_' in str(k)) + 1
        except ValueError:
            next_idx = len(K)
        except Exception:
            next_idx = len(K)
        for col in nuevas:
            K[f'k_{next_idx}'] = col
            next_idx += 1
            agregadas += 1
        K_actualizado = K
    else:
        K.extend(nuevas)
        agregadas = len(nuevas)
        K_actualizado = K

    info = {
        "agregadas": agregadas,
        "idx": np.asarray(unique_idx, dtype=int),
        "sign": np.asarray(unique_sign, dtype=float),
        "sev": np.asarray(unique_sev, dtype=float),
        "meta": meta
    }
    return K_actualizado, info
def generar_set_columnas_costos_reducidos_sparse(
    X, y, alpha, K, 
    n_features, n_samples, # Nuevos argumentos requeridos
    eps=1e-8, k_max=25, debug=False):
    """
    Versión sparse del fallback. Usa helpers esparsos.
    """
    a = np.asarray(alpha, float).ravel()
    yv = np.asarray(y, float).ravel()

    # pares ya presentes (j,sign)
    # pares_en_K ahora debe manejar w esparsos
    already_pairs = pares_en_K_sparse(K) 

    # screening (usa el 'alpha' que es grad_w_correcto)
    # Esta función (screen_from_dual_inequalities) no crea vectores,
    # solo hace cálculos en NumPy, por lo que puede seguir igual.
    idx, sign, sev, meta = screen_from_dual_inequalities(
        X, a, yv, eps=eps, k_max=k_max, already=already_pairs
    )

    if idx.size == 0:
        info = {"agregadas": 0, "idx": idx, "sign": sign, "sev": sev, "meta": meta}
        return K, info

    # ... (lógica de 'unique_idx' y 'seen_batch' sin cambios) ...
    # (El código de unique_idx, unique_sign, unique_sev va aquí)
    
    unique_idx, unique_sign, unique_sev = [], [], []
    seen_batch = set()
    for j, s, v in zip(idx, sign, sev):
        key = (int(j), int(np.sign(s)) or 1)
        if key in already_pairs or key in seen_batch:
            continue
        seen_batch.add(key)
        unique_idx.append(int(j))
        unique_sign.append(int(np.sign(s)) or 1)
        unique_sev.append(float(v))

    if not unique_idx:
        info = {"agregadas": 0, "idx": [], "sign": [], "sev": [], "meta": meta}
        return K, info

    # --- CAMBIO A SPARSE ---
    # construir columnas esparsas
    nuevas = [generar_canonico_con_signo_sparse(n_features, n_samples, j, s)
              for j, s in zip(unique_idx, unique_sign)]

    # ... (lógica para insertar en K (dict o list) sin cambios) ...
    
    agregadas = 0
    if isinstance(K, dict):
        try:
            next_idx = max(int(str(k).split('_')[1]) for k in K.keys() if '_' in str(k)) + 1
        except ValueError:
            next_idx = len(K)
        for col in nuevas:
            K[f'k_{next_idx}'] = col
            next_idx += 1
            agregadas += 1
        K_actualizado = K
    else:
        K.extend(nuevas)
        agregadas = len(nuevas)
        K_actualizado = K

    info = {
        "agregadas": agregadas,
        "idx": np.asarray(unique_idx, dtype=int),
        "sign": np.asarray(unique_sign, dtype=float),
        "sev": np.asarray(unique_sev, dtype=float),
        "meta": meta
    }
    return K_actualizado, info
def mosek_params_from_tol(tol, *, threads=None, presolve_level=1, optimizer_code=0):
    """
    Crea un diccionario de parámetros para MOSEK.
    
    Args:
        tol: Tolerancia numérica principal.
        threads: Número de hilos (None para default).
        presolve_level: Nivel de presolve (1=ON, 0=OFF).
        optimizer_code: Código del optimizador (0=Auto, 1=Primal Simplex, 2=Dual Simplex).
    """
    tol = float(tol)
    p = {
        # Tolerancias generales (aplican a IPM y Simplex)
        "MSK_DPAR_OPTIMIZER_MAX_TIME": 7200.0, # Límite de tiempo por solve (ej. 1 hora)
        "MSK_DPAR_INTPNT_TOL_REL_GAP": tol,
        "MSK_DPAR_INTPNT_TOL_PFEAS":   tol * 10, # A veces ser más laxo aquí ayuda
        "MSK_DPAR_INTPNT_TOL_DFEAS":   tol * 10,
        "MSK_DPAR_BASIS_TOL_X":     max(tol, 1e-9),
        "MSK_DPAR_BASIS_TOL_S":     max(tol, 1e-9),
        
        # Configuración principal
        "MSK_IPAR_OPTIMIZER": int(optimizer_code),
        "MSK_IPAR_PRESOLVE_USE": int(presolve_level), 
    }
    
    # Configuraciones específicas si presolve está OFF
    if presolve_level == 0:
        p["MSK_IPAR_PRESOLVE_ELIMINATOR_MAX_NUM_TRIES"] = 0
        p["MSK_IPAR_PRESOLVE_LINDEP_USE"] = 0
        
    if threads is not None:
        p["MSK_IPAR_NUM_THREADS"] = int(threads)
        
    return p



def mosek_params_gap_only(tol, *, threads=None, presolve=None, disable_eliminator=False):
    """
    Solo ajusta el relative gap de interior-point. (Simplex no tiene 'SIM_TOL_REL_GAP'.)
    Añade tolerancias de base por si acaso.
    """
    tol = float(tol)
    p = {
        "MSK_DPAR_INTPNT_TOL_REL_GAP": tol,
        # por si toca simplex en algún momento:
        "MSK_DPAR_BASIS_TOL_X":     max(tol, 1e-9),
        "MSK_DPAR_BASIS_TOL_S":     max(tol, 1e-9),
        "MSK_DPAR_BASIS_REL_TOL_S": max(tol, 0.0),
    }
    if threads is not None:
        p["MSK_IPAR_NUM_THREADS"] = int(threads)
    if presolve is not None:
        p["MSK_IPAR_PRESOLVE_USE"] = int(presolve)
    if disable_eliminator:
        p["MSK_IPAR_PRESOLVE_ELIMINATOR_MAX_NUM_TRIES"] = 0
        p["MSK_IPAR_PRESOLVE_LINDEP_USE"] = 0
    return p
# In[4]: Desdes aqui en adelante es todo orientado a trabajar el cojunto K, theta, y alfa de mejor manera.
#se busca que el algoritmo vaya mucho mas rapido
#se busca que MOSEK no muere
def convertir_K_a_dict(K_lista_values,K_lista_keys):
        return dict(zip(K_lista_keys,K_lista_values))
    
def convertir_dict_a_K(K_dict):
        return list(K_dict.values()),list(K_dict.keys())    
    
def K_ini_version_dict(K_lista):
        return {f'k_{i}': col for i, col in enumerate(K_lista)} #son strings los k_i

class generacion_columnas:
    def __init__(self,tol,M_box=None,solver=2):
        self.M_box=M_box
        self.tol=tol
        self.opt_val_fin=[]
        self.alpha_set_fin=[]
        self.memoria_theta={}
        self.memoria_permanente={}
        self.current_mosek_params = {}
        self.master_mosek_params = {}
        self.terminamos=False
        
        #        params= mosek_params_from_tol(tol, threads=None, presolve=1, disable_eliminator=True) #puedes cambiar este a mano si es necesario para el pricing. 
                # --- AÑADIMOS EL PARÁMETRO PARA FORZAR SIMPLEX ---
                # 1 = Simplex Primal, 2 = Simplex Dual. Prueba con 2.
        #        params["MSK_IPAR_OPTIMIZER"] = solver #al parecer simplex dual es lento, falta ver simplex primal. 
        #        self.p = params

    def ingresar_data_(self,X_train,y_train):
        self.X=X_train
        self.y=y_train
        self.n_data, self.n_columns= X_train.shape
    def ingresar_parametros_master_pricing(self,C,M,K_ini,tipo="afin",pricing=solve_pricing_problem_combinado_componente):
        
        self.K_ini_dict=K_ini_version_dict(K_ini) #se usa asi para distinguir que columna es que. Se podria cambiar a futuro a que siempre sea dict.
        self.contador_columnas=len(self.K_ini_dict)
        for i in self.K_ini_dict.keys(): #los keys son "K_i"
            self.memoria_theta[i]=[] #se inicia la memoria de las veces que se usaron los ini
            self.memoria_permanente[i]=[]
        self.C=C        
        self.tipo=tipo
        self.M=M
        self.pricing=pricing
        
    def filtrar_columnas_inutiles(self,K_dict):
        
        columnas_a_eliminar = []
        for nombre, valores_theta in self.memoria_theta.items():
            if len(valores_theta) >= self.n_periodos:
                recientes = valores_theta[-self.n_periodos:] #guardo los valores actuales en memoria_theta
                if all(abs(theta) < self.umbral_theta for theta in recientes): #revisas los valores actuales
                    columnas_a_eliminar.append(nombre) #todas estas son las columnas que no cumplen umbral

        for nombre in columnas_a_eliminar:
            del K_dict[nombre] #eliminas los que en estas ultimas iteraciones son soluciones inutiles 
            del self.memoria_theta[nombre]
            self.memoria_permanente[nombre].append(-self.M)  
    
        if columnas_a_eliminar:
            self.columnas_eliminadas_iteracion.append(columnas_a_eliminar)
            print(f"🔍 Columnas eliminadas en iteración {self.i}: {columnas_a_eliminar}")
        return K_dict

    
    def iteracion_generacion_master(self,K_dict,master=solve_master_primal_v2): # el master devuelve cositas.
        
        # si es solve_master_primal_original entrega: 
        # theta_opt, eta.value, alpha, w_combo.value, prob.value
        
        #si es solve_master_primal entrega:
        # theta_opt, eta.value, alpha, w_combo.value, prob.value, b.value, xi.value
        
        #si es solve_master_primal_v2 entrega:
        # theta.value, eta.value, alpha, w_combo.value, prob.value, b.value, xi.value, grad_w_correcto 
        
        K_list,K_list_keys=convertir_dict_a_K(K_dict) #creamos las llaves y valores de K
            
        resultados_master_ = master(self.X, self.y, K_list, self.tipo, self.C, self.tol, 
                                mosek_params=self.master_mosek_params)
        #theta_opt, eta.value, alpha, b.value, xi.value, w_combo.value, prob.value
        print("***"*10)
        print("el valor optimo actual del master es ", resultados_master_[4]) #el prob value
        print("***"*10)
        self.opt_val_fin.append(resultados_master_[4])
        if master== solve_master_primal_v2: #vigilar el gradiente bueno. 
            self.alpha_set_fin.append(resultados_master_[7])
        else:
            self.alpha_set_fin.append(resultados_master_[2])
        
        theta_iter= resultados_master_[0] #el theta actual #este es 
        
        #memoria_theta es memoria global del uso de K_ini+K_generados.
        #K_dict es los K en uso de K_ini+K_generados, usualmente es menor. Es un sub conjunto del de arriba. 
        
        # Registrar θ en la memoria
        for j in range(len(theta_iter)):
            self.memoria_theta[K_list_keys[j]].append(theta_iter[j]) #usas los Keys de K_dict para rellenar la memoria theta.
            self.memoria_permanente[K_list_keys[j]].append(theta_iter[j])
            # 🔧 DEVOLVER K_dict para que quien llame pueda seguir usándolo
        
        return K_dict #veamos si rompe
    
    
    
    
        # Pricing
    # def iteracion_generacion_pricing(self,K_dict):       
    #     ki = self.pricing(self.X, self.y, self.alpha_set_fin[-1],K_dict, self.C)

    #     # Agregar nueva columna {f'k_{i}': col for i, col in enumerate(K_lista)}
        
    #     #si es que agrego aun mas columnas, tendria que revisar si es que estan o algo asi y agregarlas con un for
    #     nombre_columna = f'k_{self.contador_columnas}'
    #     self.contador_columnas += 1
        
    #     self.memoria_theta[nombre_columna] = [np.nan] * (self.i+1)
    #     self.memoria_permanente[nombre_columna] = [np.nan] * (self.i+1)
        
    #     K_dict[nombre_columna] = ki #se agrega la nueva columna


    #     self.K_generados[nombre_columna]=ki #se revisa  que se ha estado agregando 
    #     return K_dict
    
    def iteracion_generacion_pricing(self, K_dict):
        """
        Llama al pricing con (X, y, alpha, K_dict, C).
        Acepta dos salidas posibles:
          - (w, b, xi): agrega UNA columna nueva 'k_{..}' a K_dict.
          - dict/list con columnas: incorpora las nuevas y prepara la memoria.
        Devuelve el K_dict actualizado.
        """
        # --- snapshot de claves antes, para detectar qué se agregó
        keys_antes = set(K_dict.keys())
        grad_w = self.alpha_set_fin[-1]
        norm_grad = np.linalg.norm(grad_w)
        if norm_grad > 1e-8: # Evitar división por cero
            grad_w_normalizado = grad_w / norm_grad
        else:
            grad_w_normalizado = grad_w
        
        
        # el pricing ahora necesita K_dict
        res = self.pricing(self.X, self.y, grad_w_normalizado, K_dict, self.C, self.current_mosek_params, self.M_box)
        # Caso A: el pricing nos dio una sola columna (w,b,xi)
        if isinstance(res, tuple) and len(res) == 3:
            w_new, b_new, xi_new = res
            nombre_columna = f'k_{self.contador_columnas}'
            self.contador_columnas += 1
    
            # inicializa memoria para la nueva columna
            self.memoria_theta[nombre_columna] = [np.nan] * (self.i + 1)
            self.memoria_permanente[nombre_columna] = [np.nan] * (self.i + 1)
    
            # inserta en el K en uso y en el log de generadas
            K_dict[nombre_columna] = (w_new, b_new, xi_new)
            self.K_generados[nombre_columna] = (w_new, b_new, xi_new)
            return K_dict    
        # Caso B1: el pricing devolvió un dict (K mutado con nuevas canónicas)
        if isinstance(res, dict):
            K_new = res  # ya viene con las columnas insertadas por el screening
            nuevas = [k for k in K_new.keys() if k not in keys_antes]
            for k in nuevas:
                self.memoria_theta[k] = [np.nan] * (self.i + 1)
                self.memoria_permanente[k] = [np.nan] * (self.i + 1)
                self.K_generados[k] = K_new[k]
            # opcional: sincronizar contador (no exige renombrar)
            self.contador_columnas = max(self.contador_columnas, len(K_new))
            return K_new
        
        # Caso B2: el pricing devolvió una lista de columnas [(w,b,xi), ...]
        if isinstance(res, list):
            for col in res:
                nombre_columna = f'k_{self.contador_columnas}'
                self.contador_columnas += 1
                self.memoria_theta[nombre_columna] = [np.nan] * (self.i + 1)
                self.memoria_permanente[nombre_columna] = [np.nan] * (self.i + 1)
                K_dict[nombre_columna] = col
                self.K_generados[nombre_columna] = col
            return K_dict
        
        # Fallback: no reconocí la salida; no agrego nada
        print("[pricing] salida no reconocida; no se agregan columnas")
        return K_dict
        
        
        
        
    def run(self,max_iter,umbral_theta,n_periodos,frecuencia_check, nuevo=True, master=solve_master_primal_v2):
                                                                                                          
        self.i=0
        time_ini = time.time()
        self.umbral_theta=umbral_theta #el theta que es 0 para nosotros
        self.n_periodos=n_periodos #cuanto tiempo valiendo callampa
        self.frecuencia_check=frecuencia_check     #cada cuanto revisamos, debe ser mayor a n_periodos
        self.columnas_eliminadas_iteracion=[] #set donde se guarda quien se borro en que iteracion
        self.K_generados={} #guardamos quieres se crearon
        K_dict=dict(self.K_ini_dict) #el que se creo al ingresar las necesidasdes del solver. 
        self.terminamos=False
        
        while self.terminamos==False:
            print("***" * 30)
            print(f"inicio con iteracion # {self.i} ")
            
    
            # Filtrar cada cierto número de iteraciones
            if self.i > 0 and self.i % frecuencia_check == 0 and self.i!=max_iter: #no es ni la primera iteracion, ni la ultima, y tiene que cumplir frecuencia check. 
                K_dict = self.filtrar_columnas_inutiles(K_dict) #limpiamos las columnas malas.
                
            #cambio, ahora haremos el master post filtro para asi tener un vector alfa mejor 
            self.iteracion_generacion_master(K_dict,master=master) #obtenemos los thetas y los alfas
             
            
            # Criterio de parada: mejora marginal en F.O.
            if self.i > 1 and np.linalg.norm(self.opt_val_fin[-1] - self.opt_val_fin[-2]) <= self.tol:
                print("****"*10, " fin ", "****"*8)
                print("la diferencia de las fo fue", np.linalg.norm(self.opt_val_fin[-2]-self.opt_val_fin[-1]))
                print("Criterio de parada: mejora marginal alcanzada.")
                self.status="optimo"
                self.terminamos=True
                self.K_fin=K_dict
                break
                
            if self.i>=max_iter:
                
                print("****"*10, " fin ", "****"*8)
                print("Criterio de parada: máximo de iteraciones.")
                self.status="max_iter"
                self.terminamos=True
                self.K_fin=K_dict
                break
                
            #ponemos el pricing post criterios de salida, para no tener que generar una columna innecesaria que aparece con puros NAN.
            
            K_dict=self.iteracion_generacion_pricing(K_dict) #se actualiza el k_dict temporal de aca.  #el pricing parte creando una columna nueva.    
            # Si no cambió el tamaño, no hubo columnas nuevas: puedes decidir cortar
            self.i=self.i+1
            
        time_fin = time.time()
        print("⏱ Tiempo total:", round((time_fin - time_ini)/60, 2), "minutos")
        return self     
    
    # def run_canonico(self, max_iter, umbral_theta, n_periodos, frecuencia_check):
    #     """
    #     Igual que 'run', pero agrega columnas canónicas (±e_j) usando
    #     generar_set_columnas_costos_reducidos(...) en vez de llamar al pricing.
    #     """
    #     if not hasattr(self, "K_ini_dict"):
    #         raise RuntimeError("[run_canonico] Falta self.K_ini_dict. Llama antes a ingresar_parametros_master_pricing().")
    
    #     self.i = 0
    #     time_ini = time.time()
    #     self.umbral_theta = umbral_theta
    #     self.n_periodos = n_periodos
    #     self.frecuencia_check = frecuencia_check
    #     self.columnas_eliminadas_iteracion = []
    #     self.K_generados = {}
    
    #     # ✅ K seguro (si es None -> {})
    #     K_dict = dict(getattr(self, "K_ini_dict", {}) or {})
    #     self.terminamos = False
    
    #     eps_screen   = getattr(self, "screen_eps", getattr(self, "tol", 1e-8))
    #     k_max_screen = getattr(self, "k_max_screen", 25)
    
    #     for i in range(max_iter):
    #         self.i = i
    
    #         if frecuencia_check and (i % frecuencia_check == 0):
    #             if hasattr(self, "filtro_columnas") and callable(self.filtro_columnas):
    #                 K_dict = self.filtro_columnas(K_dict)
    
    #         # 1) Master
    #         K_dict = self.iteracion_generacion_master(K_dict)
    
    #         # 2) Paro por mejora marginal en θ
    #         if len(getattr(self, "theta_set_fin", [])) >= n_periodos + 1:
    #             theta_now  = self.theta_set_fin[-1]
    #             theta_prev = self.theta_set_fin[-1 - n_periodos]
    #             if abs(theta_now - theta_prev) <= umbral_theta:
    #                 if getattr(self, "verbose", False):
    #                     print("[run_canonico] paro por mejora marginal THETA")
    #                 self.status = "optimo_theta"
    #                 self.terminamos = True
    #                 self.K_fin = K_dict
    #                 break
    
    #         # 2b) Paro por mejora marginal en FO (si ya tienes dos FO)
    #         if len(getattr(self, "opt_val_fin", [])) >= 2:
    #             if np.linalg.norm(self.opt_val_fin[-1] - self.opt_val_fin[-2]) <= getattr(self, "tol", 1e-8):
    #                 if getattr(self, "verbose", False):
    #                     print("[run_canonico] paro por mejora marginal FO")
    #                 self.status = "optimo_fo"
    #                 self.terminamos = True
    #                 self.K_fin = K_dict
    #                 break
    
    #         # 3) Screening y alta de columnas (si hay alpha)
    #         if not getattr(self, "alpha_set_fin", None):
    #             if getattr(self, "verbose", False):
    #                 print("[run_canonico] no hay alpha esta iteración")
    #             continue
    
    #         alpha = self.alpha_set_fin[-1]
    
    #         keys_antes = set(K_dict.keys())
    #         K_dict, info = generar_set_columnas_costos_reducidos(
    #             self.X, self.y, alpha, K_dict,
    #             eps=eps_screen, k_max=k_max_screen, w_proto=None, debug=False
    #         )
    
    #         # memorias para las nuevas
    #         nuevas = [k for k in K_dict.keys() if k not in keys_antes]
    #         for k in nuevas:
    #             self.memoria_theta[k] = [np.nan] * (i + 1)
    #             self.memoria_permanente[k] = [np.nan] * (i + 1)
    #             self.K_generados[k] = K_dict[k]
    
    #         # sincroniza contador
    #         try:
    #             max_idx = max(int(str(k).split('_')[1]) for k in K_dict.keys() if '_' in str(k))
    #             self.contador_columnas = max(self.contador_columnas, max_idx + 1)
    #         except Exception:
    #             self.contador_columnas = max(self.contador_columnas, len(K_dict))
    
    #         # si no se agregó nada, fin
    #         if info.get("agregadas", 0) == 0:
    #             if getattr(self, "verbose", False):
    #                 print("[run_canonico] screening no encontró columnas nuevas; fin.")
    #             self.status = "sin_mejoras"
    #             self.terminamos = True
    #             self.K_fin = K_dict
    #             break
    
    #     # fin de loop (si no rompió)
    #     if not getattr(self, "terminamos", False):
    #         self.status = "max_iter"
    #         self.K_fin = K_dict
    
    #     self.K_ini_dict = K_dict
    #     print("⏱ Tiempo total:", round((time.time() - time_ini) / 60, 2), "min")
    #     return K_dict
    def run_canonico(
    self,
        max_iter,
        umbral_theta,
        n_periodos,
        frecuencia_check,
        *,
        rc_eps=None,          # tolerancia para el screening (default: self.tol o 1e-8)
        rc_kmax=None,         # cuántas canónicas por iteración (default: 25)
        pricing_every=5,      # cada cuántas iteraciones forzar un pricing “normal”
        patience=2,            # si la FO no mejora 'patience' veces seguidas, llamar pricing
        master=solve_master_primal):
        """
        Bucle híbrido:
          1) Resuelve MASTER.
          2) Por defecto hace screening de costos reducidos y agrega ±e_j (sin duplicados).
          3) Cada 'pricing_every' iters o si hay estancamiento ('patience'), llama al pricing normal.
          4) Si el screening no agrega nada, también cae a pricing como fallback.
    
        Requisitos: haber llamado antes a: 
          - ingresar_data_(X, y)
          - ingresar_parametros_master_pricing(C, M, K_ini, tipo=..., pricing=..., ...)
        """
    
        # ----------- chequeos y setup -----------
        if not hasattr(self, "K_ini_dict"):
            raise RuntimeError("[run_canonico] Falta self.K_ini_dict. Llama antes a ingresar_parametros_master_pricing().")
        if not hasattr(self, "X") or not hasattr(self, "y"):
            raise RuntimeError("[run_canonico] Falta data (X,y). Llama antes a ingresar_data_().")
    
        # K de trabajo (copia independiente) y estado
        K_dict = dict(self.K_ini_dict)
        self.K_generados = {}
        self.columnas_eliminadas_iteracion = []
        self.umbral_theta = umbral_theta
        self.n_periodos = n_periodos
        self.frecuencia_check = frecuencia_check
        self.terminamos = False
        self.status = None
    
        # parámetros de screening
        eps_screen = rc_eps if rc_eps is not None else getattr(self, "tol", 1e-8)
        k_max_screen = int(rc_kmax if rc_kmax is not None else 25)
    
        # contadores de estancamiento / control de llamadas a pricing
        no_improve_count = 0
        added_last = 0
    
        t0 = time.time()
    
        for i in range(max_iter):
            self.i = i
    
            # (opcional) mantenimiento de K cada 'frecuencia_check' iteraciones
            if self.frecuencia_check and (i % self.frecuencia_check == 0):
                if hasattr(self, "filtro_columnas") and callable(self.filtro_columnas):
                    K_dict = self.filtro_columnas(K_dict)
    
            # -------- 1) MASTER --------
            # Calcula alpha, theta, etc. y los deja en los *sets* de la clase
            K_dict = self.iteracion_generacion_master(K_dict,master=master)
    
            # -------- 2) Criterios de parada --------
            # a) por theta (ventana temporal)
            if len(getattr(self, "theta_set_fin", [])) >= self.n_periodos + 1:
                theta_now = self.theta_set_fin[-1]
                theta_prev = self.theta_set_fin[-1 - self.n_periodos]
                if abs(theta_now - theta_prev) <= self.umbral_theta:
                    if getattr(self, "verbose", False):
                        print("[run_canonico] paro por mejora marginal THETA.")
                    self.status = "optimo"
                    self.terminamos = True
                    break
    
            # b) por mejora marginal en FO (usa lista self.opt_val_fin)
            if len(getattr(self, "opt_val_fin", [])) >= 2:
                delta = abs(self.opt_val_fin[-1] - self.opt_val_fin[-2])
                if delta <= getattr(self, "tol", 1e-8):
                    no_improve_count += 1
                else:
                    no_improve_count = 0
    
                if no_improve_count >= self.n_periodos:
                    if getattr(self, "verbose", False):
                        print("[run_canonico] paro por FO casi constante.")
                    self.status = "optimo"
                    self.terminamos = True
                    break
    
            # Si todavía no hay alpha, no tiene sentido screenear ni hacer pricing
            if not getattr(self, "alpha_set_fin", None):
                if getattr(self, "verbose", False):
                    print("[run_canonico] aún no hay alpha; continuar.")
                continue
    
            alpha = self.alpha_set_fin[-1]
    
            # -------- 3) Decidir: screening vs pricing --------
            # Disparadores para pricing normal:
            trigger_pricing = (
                (pricing_every is not None and pricing_every > 0 and (i % pricing_every == 0) and i > 0) or
                (no_improve_count >= patience) or
                (added_last == 0)  # si el screening anterior no agregó nada, intentamos pricing
            )
    
            if trigger_pricing:
                # ----- PRICING NORMAL -----
                keys_antes = set(K_dict.keys())
                K_dict = self.iteracion_generacion_pricing(K_dict)  # esta función ya sabe manejar tuple/list/dict
                nuevas = [k for k in K_dict.keys() if k not in keys_antes]
    
                # Inicializa memorias para nuevas columnas del pricing
                for k in nuevas:
                    self.memoria_theta[k] = [np.nan] * (i + 1)
                    self.memoria_permanente[k] = [np.nan] * (i + 1)
                    self.K_generados[k] = K_dict[k]
    
                # sincroniza contador (por seguridad; no renombramos claves existentes)
                try:
                    max_idx = max(int(str(k).split('_')[1]) for k in K_dict.keys() if '_' in str(k))
                    self.contador_columnas = max(self.contador_columnas, max_idx + 1)
                except Exception:
                    self.contador_columnas = max(self.contador_columnas, len(K_dict))
    
                added_last = len(nuevas)
                if getattr(self, "verbose", False):
                    print(f"[run_canonico] pricing agregó {added_last} columnas.")
    
                # Si tampoco el pricing pudo agregar nada, ya estamos estancados
                if added_last == 0 and no_improve_count >= patience:
                    if getattr(self, "verbose", False):
                        print("[run_canonico] ni screening ni pricing agregan; fin.")
                    break
    
            else:
                # ----- SCREENING DE COSTOS REDUCIDOS (±e_j) -----
                keys_antes = set(K_dict.keys())
                K_dict, info = generar_set_columnas_costos_reducidos(
                    self.X, self.y, alpha, K_dict,
                    eps=eps_screen, k_max=k_max_screen, w_proto=None, debug=False
                )
    
                nuevas = [k for k in K_dict.keys() if k not in keys_antes]
                for k in nuevas:
                    self.memoria_theta[k] = [np.nan] * (i + 1)
                    self.memoria_permanente[k] = [np.nan] * (i + 1)
                    self.K_generados[k] = K_dict[k]
    
                try:
                    max_idx = max(int(str(k).split('_')[1]) for k in K_dict.keys() if '_' in str(k))
                    self.contador_columnas = max(self.contador_columnas, max_idx + 1)
                except Exception:
                    self.contador_columnas = max(self.contador_columnas, len(K_dict))
    
                added_last = int(info.get("agregadas", 0))
                if getattr(self, "verbose", False):
                    print(f"[run_canonico] screening agregó {added_last} columnas (k_max={k_max_screen}).")
    
                # Si el screening no encontró nada, vamos a pricing *en la próxima iteración*
                # (o lo forzamos inmediatamente cambiando trigger_pricing aquí si prefieres).
    
        # cierre
        self.K_fin = K_dict
        tf = time.time()
        print("⏱ Tiempo total:", round((tf - t0)/60, 2), "min")
        return K_dict


def graficar_memoria_theta(memoria_permanente, titulo="Evolución de θ (Column Generation)", M=None):
    """
    Grafica la evolución de los valores de theta por columna generada.

    Parámetros:
    - memoria_permanente: dict con claves tipo 'k_0', 'k_1', ... y listas de theta por iteración.
    - titulo: título opcional del gráfico.
    - M: si se especifica, marcará con 'X' en rojo donde theta == -M.
    """
    # Detectar largo máximo de iteraciones
    max_len = max(len(thetas) for thetas in memoria_permanente.values())

    # Inicializar matriz con NaN
    theta_mat = np.full((len(memoria_permanente), max_len), np.nan)

    # Ordenar nombres
    nombres_ordenados = sorted(memoria_permanente.keys(), key=lambda x: int(x.split("_")[1]))

    # Rellenar la matriz
    for i, nombre in enumerate(nombres_ordenados):
        valores = memoria_permanente[nombre]
        theta_mat[i, :len(valores)] = valores

    # Crear figura
    plt.figure(figsize=(20, 16))
    im = plt.imshow(theta_mat, cmap="viridis", aspect="auto")
    plt.colorbar(im, label="Valor de θ")
    plt.xlabel("Iteración")
    plt.ylabel("Columna generada")
    plt.yticks(ticks=np.arange(len(nombres_ordenados)), labels=nombres_ordenados)
    plt.title(titulo)
    
    # Opcional: marcar -M si se entrega
    if M is not None:
        for i in range(theta_mat.shape[0]):
            for j in range(theta_mat.shape[1]):
                if theta_mat[i, j] == -M:
                    plt.plot(j, i, 'rx', markersize=6, markeredgewidth=1.5)

    plt.tight_layout()
    plt.show()
    return theta_mat

 
#graficar_memoria_theta(win_maxima.memoria_permanente, M=10000)
#mt=win_maxima.memoria_theta
#mp=win_maxima.memoria_permanente



def graficar_memoria_theta_percentil(memoria_permanente, titulo="Log₁₀(|θ|): Column Generation", M=None, eps=1e-10):
    """
    Grafica log10 del valor absoluto de theta a través de las iteraciones para cada columna generada.

    Parámetros:
    - memoria_permanente: dict tipo {'k_0': [...], 'k_1': [...]}
    - titulo: título opcional del gráfico
    - M: valor usado para marcar columnas eliminadas (-M). Se destaca con una X roja si se especifica
    - eps: pequeño valor para evitar log(0)
    """
    max_len = max(len(thetas) for thetas in memoria_permanente.values())
    theta_mat = np.full((len(memoria_permanente), max_len), np.nan)
    nombres_ordenados = sorted(memoria_permanente.keys(), key=lambda x: int(x.split("_")[1]))

    for i, nombre in enumerate(nombres_ordenados):
        valores = memoria_permanente[nombre]
        valores = np.array(valores, dtype=float)
        # Transformación logarítmica (salvo -M o 0)
        valores_log = np.where(valores == -M, np.nan, np.log10(np.clip(np.abs(valores), eps, None)))
        theta_mat[i, :len(valores_log)] = valores_log

    plt.figure(figsize=(20, 16))
    im = plt.imshow(theta_mat, cmap="viridis", aspect="auto")
    cbar = plt.colorbar(im, label="log₁₀(|θ|)")
    plt.xlabel("Iteración")
    plt.ylabel("Columna generada")
    plt.yticks(ticks=np.arange(len(nombres_ordenados)), labels=nombres_ordenados)
    plt.title(titulo)

    # Marcar donde hubo -M (columnas eliminadas)
    if M is not None:
        for i in range(theta_mat.shape[0]):
            nombre = nombres_ordenados[i]
            for j in range(theta_mat.shape[1]):
                if j < len(memoria_permanente[nombre]) and memoria_permanente[nombre][j] == -M:
                    plt.plot(j, i, 'rx', markersize=6, markeredgewidth=1.5)

    plt.tight_layout()
    plt.show()
    return theta_mat
    
 
# In[4]:   
# parece que esta es la mejor
def duales_pricing_revisar(X, y, alpha, C=1.0):

    n_samples, n_features = X.shape
    y_neg = np.where(y <= 0, -1, 1)  # usualmente o son 0, o son 1.

    # Variables
    
    pi= cp.Variable(n_samples, nonneg=True)
    lmbd_r1= cp.Variable(nonneg=True)
    lmbd_nfeatures=cp.Variable(n_features, nonneg=True) #el que iba transpuesto con el w. 

    #parametro
    alpha = alpha.flatten()


    
    # Precomputar cosas
    

    xpr=-cp.multiply(alpha ,lmbd_nfeatures)+X.T@pi
            
            
            
    constraints_caso_coordenada = [y_neg @ pi == 0, pi<=C, xpr== -alpha] #las 2 que van siempre. # cp.multiply(-alpha,lmbd_nfeatures)+X.T @pi == - alpha 
    constraints_caso_suma=  [y_neg @ pi == 0, pi<=C, -lmbd_r1 * alpha + X.T @ pi == - alpha  ] #la tercera restriccion tiene siempre el mismo lambda

    objective_ = cp.Maximize(cp.sum(pi))
    # Solve
    prob_coordenada= cp.Problem(objective_, constraints_caso_coordenada)
    prob_suma= cp.Problem(objective_, constraints_caso_suma)
    
    resultados={}    
    prob_coordenada.solve(
        solver=cp.MOSEK, mosek_params={"MSK_IPAR_PRESOLVE_LINDEP_USE": 0, "MSK_DPAR_OPTIMIZER_MAX_TIME": 1200.0
                                       }, verbose=True)
    resultados["coordenada"]= [[pi.value,lmbd_nfeatures.value],prob_coordenada.value]
    
    prob_suma.solve(
        solver=cp.MOSEK, mosek_params={"MSK_IPAR_PRESOLVE_LINDEP_USE": 0, "MSK_DPAR_OPTIMIZER_MAX_TIME": 1200.0
                                       }, verbose=True)
    resultados["suma"]= [[pi.value,lmbd_r1.value],prob_suma.value]
    


    # , prob_pricing.value, prob_pricing.status #el VALOR OPTIMO ES EL MEJOR (4) EL ALPHA QUE ENTREGA ES PESIMO, DESTRUYENDO SIGUIENTE ITERACION PRICING. sera ruido?
    cc= solve_pricing_problem_combinado_componente(X, y, alpha, C=1.0)
    sr= solve_pricing_problem_combinado_sumar_restricto(X, y, alpha, C=1.0)
    return resultados,[cc,sr]

# In[4]:    ESTO ES DE CHATGPT PARA CORRER EL EXPERIMENTO ALGO MAS SEGURO

from sklearn.metrics import accuracy_score, classification_report

# --- Helper seguro para inicializar la clase con distintas firmas ---
def _init_gc_with_params(gcg, C, M, K_ini, tipo=None, pricing=None): #esta funcion se siente super innecesaria. 
    """
    Intenta llamar ingresar_parametros_master_pricing con diferentes firmas:
    (C,M,K_ini,tipo,pricing) -> (C,M,K_ini,tipo) -> (C,M,K_ini)
    Además, si no hay argumento 'pricing', asigna gcg.pricing = pricing si existe.
    """
    called = False
    if not called and (tipo is not None) and (pricing is not None):
        try:
            gcg.ingresar_parametros_master_pricing(C, M, K_ini, tipo=tipo, pricing=pricing)
            called = True
        except TypeError:
            pass
    if not called and (tipo is not None):
        try:
            gcg.ingresar_parametros_master_pricing(C, M, K_ini, tipo=tipo)
            if pricing is not None:
                try:
                    setattr(gcg, "pricing", pricing)
                except Exception:
                    pass
            called = True
        except TypeError:
            pass
    if not called:
        # Firma básica
        gcg.ingresar_parametros_master_pricing(C, M, K_ini)
        if pricing is not None:
            try:
                setattr(gcg, "pricing", pricing)
            except Exception:
                pass

def run_gc_once(X_train, y_train, X_test, y_test,
                tipo, pricing,
                C=1.0, M=1e4, tol=1e-6,
                max_iter=50, tamaño_ini=0.10,K_inicial=None): #corre generacion columnas base
    """ 
    Ejecuta una corrida de generación de columnas (master+pricing) para (tipo, pricing).
    Devuelve un dict con métricas y snapshots útiles.
    """
    n_features = X_train.shape[1]
    # --- NUEVO: K_ini manual o canónico ---
    if K_inicial is None:
        K_ini = generar_K_canonico(range(n_features), tamaño=tamaño_ini)
    else:
        K_ini = K_inicial
        
    gcg = generacion_columnas(tol)
    gcg.ingresar_data_(X_train, y_train)

    # Inicialización robusta a la firma que tengas
    _init_gc_with_params(gcg, C=C, M=M, K_ini=K_ini, tipo=tipo, pricing=pricing)

    # Parámetros de run, siguiendo tu estilo
    t0 = time.time()
    max_iter = int(max_iter)
    umbral_theta = 1.0 / (gcg.n_columns * 10.0) #esto es un 10% mas chico que repartir equitativamente
    n_periodos = max(1, max_iter // 10)
    frecuencia_check = max(1, max_iter // 5)

    gcg.run(max_iter, umbral_theta, n_periodos, frecuencia_check)
    t1 = time.time()

    # Resuelve master final explícitamente para obtener (w,b,xi) y métricas
    # convertir_dict_a_K(...) -> tuple (K_list, ?) según tu implementación
    K_list = convertir_dict_a_K(gcg.K_fin)[0]

    # Si tu master requiere el 'tipo', lo pasamos; si no, hay try/except
    try:
        theta_opt, eta, alpha, b, xi, w, fo = solve_master_primal(
            X_train, y_train, K_list, tipo, C, tol
        )
    except TypeError:
        theta_opt, eta, alpha, b, xi, w, fo = solve_master_primal(
            X_train, y_train, K_list, C, tol
        )

    # Predicción y métricas
    y_pred = predict_t(b, w, X_test)

    acc = accuracy_score(y_test, y_pred)
    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # FO en tu formato (‖w‖ + C * ∑ξ)
    obj_val = float(np.linalg.norm(w) + C * np.sum(xi))
    tiempo = t1 - t0

    # Resumen de θ por columna (último valor)
    def _theta_stats(mem_theta, tolz=1e-12):
        if not mem_theta:
            return dict(theta_last_active=0, theta_last_nonzero=0,
                        theta_last_max=np.nan, theta_last_min=np.nan)
        ult = np.array([vals[-1] if len(vals) else np.nan for vals in mem_theta.values()], float)
        return dict(
            theta_last_active=int(np.sum(~np.isnan(ult))),
            theta_last_nonzero=int(np.sum(np.abs(ult) > tolz)),
            theta_last_max=float(np.nanmax(ult)),
            theta_last_min=float(np.nanmin(ult)),
        )

    theta_stats = _theta_stats(gcg.memoria_theta)

    # Nombre legible del pricing
    pricing_name = getattr(pricing, "__name__", str(pricing))

    fila = {
        "tipo": tipo,
        "pricing": pricing_name,
        "C": C, "M": M, "tol": tol,
        "max_iter": max_iter,
        "umbral_theta": umbral_theta,
        "n_periodos": n_periodos,
        "frecuencia_check": frecuencia_check,
        "iters": getattr(gcg, "i", None),
        "time_sec": tiempo,
        "fo_last": fo,
        "obj_val": obj_val,
        "accuracy": acc,
        "f1_macro": rep["macro avg"]["f1-score"],
        "f1_weighted": rep["weighted avg"]["f1-score"],
        "norm_w": float(np.linalg.norm(w)),
        "sum_xi": float(np.sum(xi)),
        "slacks_positivos": int(np.sum(xi > tol)),
        "n_columns_init": len(K_ini),
        "n_columns_final": len(gcg.K_fin),
        "n_generated": len(getattr(gcg, "K_generados", {})),
        "n_eliminated_total": sum(len(x) for x in getattr(gcg, "columnas_eliminadas_iteracion", [])),
        # snapshots para análisis futuro (quedan en el .pkl)
        "opt_val_series": list(getattr(gcg, "opt_val_fin", [])),
        "alpha_last": alpha,
        "K_fin_keys": list(gcg.K_fin.keys()),
        "memoria_theta": gcg.memoria_theta,
        "memoria_permanente": gcg.memoria_permanente,
    }
    fila.update(theta_stats)
    return fila, gcg

