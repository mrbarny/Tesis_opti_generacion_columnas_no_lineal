# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 18:35:44 2025

@author: matta
"""

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import time




# In[1]:  Automatizar  y ver resultados.


def evaluar_svm(X_train, y_train,X_test, C=1.0, test_size=0.2, random_state=2, print_report=False):
    y_train=np.where(y_train <= 0, -1, 1)
    start = time.time()

    clf = OneVsRestClassifier(LinearSVC(C=C, loss="hinge", max_iter=10000))
    clf.fit(X_train, y_train)
    end = time.time()
    tiempo = end - start

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0)
    est = clf.estimators_[0]
    w = est.coef_.flatten()
    b = est.intercept_[0]
    margins = y_train * (X_train @ w + b)
    xi = np.maximum(0, 1 - margins)
    obj_val = np.linalg.norm(w) + C * np.sum(xi)

    return {
        "accuracy": acc,
        "f1_macro": report["macro avg"]["f1-score"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        "norm_w": np.linalg.norm(w),
        "sum_xi": np.sum(xi),
        "slacks_positivos": np.sum(xi > 0),
        "obj_val": obj_val,
        "tiempo": tiempo
    }
  # In[1]:  Automatizar .
def crear_data(tamaño=[1e3,1e5], classes=2, info=0.5, redun=0.3, rep=0.2, peso=None,
               flipy=0.01, disper=1.0, clusters=2, hypercubo=False, desplazar=0.0, escalar=1.0,
               shufflear=True, semilla=2):

    X, y = make_classification(
        n_samples=int(tamaño[0]),
        n_features=int(tamaño[1]),
        n_informative=int(tamaño[1] * info),
        n_redundant=int(tamaño[1] * redun),
        n_repeated=int(tamaño[1] * rep),
        n_classes=classes,
        n_clusters_per_class=clusters,
        weights=peso,
        flip_y=flipy,
        class_sep=disper,
        # Para testear clasificadores con invariancia rotacional. #genera datos que se ven como aristas, estas se intersectan por clase.
        hypercube=hypercubo,

        shift=desplazar,
        scale=escalar,
        shuffle=shufflear,
        random_state=semilla
    )
    return X, y

# Benchmarks válidos (respetando info+redun+rep <= 1.0)

#metodo de crear varia bles dinamicamente en memoria. 


#make clasification esta funcionando por debajo. #el porcentaje va multiplicado por el tamano del dataset

#n_samples = n_features = tamano para mi funcion

# n_informative =  #cantidad de clusters  alrededor de vertices de un hipercubo de tamano esto. se ponen aleatoriamente y se combinan

# n_redundant = # combinacion lineal de las features informativas

# n_repeated = # numero de ffeatures repetidas tomadas de reduntante e informativas.

# flipy=  #fraccion de las samples cuyas clases son asignadas al azar.  

# disper= class_sep. ayuda a hacer el hipercubo separable. A veces ayuda a separar mejor base es 1.0

# deslplazar = mueve a las features por ese valor

# escalar = multiplica las features por ese valor, luego son escaladas por un valor aleatorio entre 1 y 100. este viene despues de desplazar

#shufflear = desordena el orden de las samples y features.

# semilla 

#detalle the total number of features. These comprise n_informative informative features, n_redundant redundant features, n_repeated duplicated features 
# and n_features-n_informative-n_redundant-n_repeated useless features drawn at random.

# entonces para notros es 1- %a -%b- %c = %INUTIL. 

#shifting after scaling 

tamano=[1e3,1e3]

#la data OG es 1e3 x 1e6 
benchmarks_1000= {
    "fácil": dict(tamaño=tamano, info=0.8, redun=0.1, rep=0.0, flipy=0.0, disper=2.0),
    "difícil": dict(tamaño=tamano, info=0.3, redun=0.4, rep=0.2, flipy=0.1, disper=0.5),
    "poca_info": dict(tamaño=tamano, info=0.05, redun=0.4, rep=0.1, flipy=0.05, disper=1.0),
    "desbalanceado": dict(tamaño=tamano, info=0.5, redun=0.3, rep=0.1, flipy=0.01, disper=1.5, peso=[0.95, 0.05]),
    "genómica": dict(tamaño=tamano, info=0.02, redun=0.3, rep=0.1, flipy=0.05, disper=0.5, escalar=10.0, desplazar=5.0, hypercubo=False),
    "texto_emb": dict(tamaño=tamano, info=0.1, redun=0.5, rep=0.0, flipy=0.1, disper=0.3, escalar=8.0, desplazar=2.0, hypercubo=False),
    "alta_disp": dict(tamaño=tamano, info=0.05, redun=0.2, rep=0.05, flipy=0.02, disper=2.0, escalar=5.0, desplazar=10.0, hypercubo=False)
}



# In[1]: base de datos normalizado vs sin normalizar  ESTE EXPLOTA CUIDADO.

X, y = crear_data(**benchmarks_1000["texto_emb"]) #creamos los datos. del tamano de _e6  
y=np.where(y <= 0, -1, 1)
#from sklearn.feature_extraction.text import TfidfVectorizer
#tfidf_vect=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_df=0.95)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y )


y_train = np.where(y_train <= 0, -1, 1)
y_test = np.where(y_test <= 0, -1, 1)


#resultados = {"svm": evaluar_svm(X_train, y_train,X_test)}

# In[1]: ver resultados base. 
from funciones_tesis_version_sparse_fast import *

#el resultado del OG. 
skl_svm_res= skl_svm(X_train, y_train, X_test)
#el resultado del conico
con_svm_res= solve_svm_conic(X_train, y_train,solo_w_b_xi=False)


# In[2]:  Automatizar .
import numpy as np 
#EN ESTA PARTE EMPEZAREMOS A TESTEAR EL ALGORITMO. EL MASTER, EL PRICING, EL SET INI, TODO.

#K_ini_subprob,column_sets,df_log= crear_sub_problema(X_train,y_train,11,keep_xi=True)


n_samples, n_features = X_train.shape
# Creas el K inicial pasándole los enteros
K_ini_canonico_list = generar_K_canonico_sparse(
    n_features=n_features, 
    n_samples=n_samples, 
    tamaño=0.05
)

n_iters=30 

K_ini_end=K_forest(X_train,y_train,n_iters, partes=101, time_max=60, tol=1e-04, keep_xi=False,solapar=True)


# In[2]: 

# =============================================================================
# SCRIPT DE EJECUCIÓN: GENERACIÓN DE COLUMNAS DANTZIG-WOLFE (SPARSE)
# =============================================================================

import time

# 1. Configuración de Hiperparámetros
C_param = 1
M_box = 1e6
tol= 1e-9
tol_master=1e-8
n_iters_forest = 30   # Iteraciones para K_forest (ajústalo según tiempo disponible)
max_iter_gc = 130   # Máximo de iteraciones del algoritmo GC
n_samples, n_features = X_train.shape
n_periodos= 10
frecuencia_check= 120
umbral_theta=1e-7
max_rayos=50 #deberia tener una formula para sacarlos. 
presolve_level=0
optimizer_code=0
print(f"Iniciando proceso para Dataset: {n_samples} muestras x {n_features} features")
# In[2]: 
# -----------------------------------------------------------------------------
# PASO 1: Generación de Sets Iniciales (Puntos y Rayos)
# -----------------------------------------------------------------------------

# A) Generar PUNTOS iniciales (K_forest)
# Estos son soluciones SVM acotadas, así que van a K_points
print("\n[1/4] Generando Puntos Iniciales con K_forest...")
t0 = time.time()
K_ini_end = K_forest(
    X_train, y_train, 
    n_iters=n_iters_forest, 
    partes=101, 
    time_max=60, 
    tol=1e-5, #la de aqui afecta cuandas columnas habran post filtro.  
    keep_xi=False, 
    solapar=True
)
print(f"-> Puntos generados: {len(K_ini_end)} (Tiempo: {time.time()-t0:.2f}s)")

# B) Generar RAYOS iniciales (Canónicos)
# Estos son vectores unitarios, conceptualmente son direcciones, van a K_rays
print("\n[2/4] Generando Rayos Iniciales (Canónicos)...")
K_ini_canonico_list = generar_K_canonico_sparse(
    n_features=n_features, 
    n_samples=n_samples, 
    tamaño=0.1 # 10% de las features como rayos iniciales
)
print(f"-> Rayos generados: {len(K_ini_canonico_list)}")
# In[2]: 
# -----------------------------------------------------------------------------
# PASO 2: Configuración de la Clase DW
# -----------------------------------------------------------------------------

print("\n[3/4] Configurando GeneracionColumnasDW...")
# Instanciar la nueva clase
gen_col = GeneracionColumnasDW(tol=tol,tol_master=tol_master, M_box=M_box,presolve_level=presolve_level,optimizer_code=optimizer_code, umbral_theta=umbral_theta)

# Ingresar Datos
gen_col.ingresar_data(X_train, y_train)

# Ingresar Parámetros (Aquí ocurre la magia de separar Puntos y Rayos)
gen_col.ingresar_parametros(
    C=C_param,
    M=M_box,
    K_ini_points=K_ini_end,          # Input de PUNTOS (se etiquetarán p_X)
    K_ini_rays=K_ini_canonico_list,    # Input de RAYOS (se etiquetarán r_X)
    
    #K_ini_points=list_points, 
    #K_ini_rays=list_rays, 
    
    tipo="convexo",                    # Los puntos suman 1 (los rayos son cónicos)
    pricing=solve_pricing_problem_caja,# Tu pricing estándar con fallback
    gradient_strategy="full_gradient",
    pricing_acceleration_cota=False,
    pricing_acceleration=True,           # <--- ¡ACTIVAMOS TU NUEVA HEURÍSTICA!
)

# -----------------------------------------------------------------------------
# PASO 3: Ejecución
# -----------------------------------------------------------------------------

print("\n[4/4] Ejecutando Algoritmo...")
resultado_final = gen_col.run(
    max_iter=max_iter_gc, 
    n_periodos=n_periodos,       # Esperar 5 iters antes de borrar
    frecuencia_check=frecuencia_check,  # Chequear limpieza cada 5 iters
    max_rayos=max_rayos #rayos que usa la aceleracion 
    )

# -----------------------------------------------------------------------------
# Resultados
# -----------------------------------------------------------------------------
if resultado_final["status"] in ["optimo_gap", "optimo", "estancamiento"]:
    print("\n✅ EJECUCIÓN EXITOSA")
    print(f"Estado Final: {resultado_final['status']}")
    print(f"Puntos Finales: {len(resultado_final['K_points'])}")
    print(f"Rayos Finales: {len(resultado_final['K_rays'])}")
    print(f"Valor Objetivo Final: {resultado_final['opt_vals'][-1]:.6f}")
else:
    print("\n⚠️ EJECUCIÓN TERMINADA (Máx Iter o Error)")

opt_vals = resultado_final['opt_vals']
K_points = resultado_final['K_points'] # Diccionario
K_rays = resultado_final['K_rays']     # Diccionario    
    
list_points, _ = convertir_dict_a_K(K_points)
list_rays, _ = convertir_dict_a_K(K_rays)
res_final = solve_master_primal_v3(
    X_train, y_train, 
    K=list_points, 
    K_rayos=list_rays, 
    tipo="convexo", 
    C=C_param, 
    M_box=None
)
print("EL OPTIMO FINAL ALCANZADO ES DE ", res_final[4],"con valor de eta de ", res_final[1])
# In[2]:

# =============================================================================
# PASO 5: RECONSTRUCCIÓN Y EVALUACIÓN
# =============================================================================

print("\n[5/5] Evaluando Resultados...")

# 1. Recuperar Historiales y Soluciones
opt_vals = resultado_final['opt_vals']
K_points = resultado_final['K_points'] # Diccionario
K_rays = resultado_final['K_rays']     # Diccionario

# Necesitamos los valores finales de theta y mu. 
# Como la clase guarda el historial, tomamos el último valor no nulo de cada uno.
# (O idealmente, resolvemos el Master una última vez con todo el set final)

print("Recalculando Master Final con todo el set generado...")
from funciones_tesis_version_sparse_fast import solve_master_primal_v3, convertir_dict_a_K

list_points, _ = convertir_dict_a_K(K_points)
list_rays, _ = convertir_dict_a_K(K_rays)

# Resolver una última vez para obtener el w_final exacto
res_final = solve_master_primal_v3(
    X_train, y_train, 
    K=list_points, 
    K_rayos=list_rays, 
    tipo="convexo", 
    C=C_param, 
    M_box=M_box
)

# Desempaquetar w_combo (el vector de pesos final)
w_final_vector = res_final[3] # w_combo_val
b_final = res_final[5]        # b_val

# 2. Métricas de Clasificación
if w_final_vector is not None:
    from sklearn.metrics import accuracy_score, classification_report
    
    # Predicción en Train
    # w_final_vector es denso (numpy array). X_train es sparse.
    train_pred = X_train.dot(w_final_vector) + b_final
    y_pred_train = np.sign(train_pred)
    # Ajustar ceros a clase negativa si es necesario (o dejarlos como error)
    y_pred_train[y_pred_train == 0] = -1 
    
    acc_train = accuracy_score(y_train, y_pred_train)
    print(f"\n---> Accuracy Train: {acc_train:.4f}")
    
    # Predicción en Test (Si tienes X_test, y_test cargados)
    if 'X_test' in globals() and 'y_test' in globals():
        test_pred = X_test.dot(w_final_vector) + b_final
        y_pred_test = np.sign(test_pred)
        y_pred_test[y_pred_test == 0] = -1
        
        acc_test = accuracy_score(y_test, y_pred_test)
        print(f"---> Accuracy Test:  {acc_test:.4f}")
        print("\nReporte de Clasificación (Test):")
        print(classification_report(y_test, y_pred_test))

    # 3. Análisis de Esparsidad del w final
    w_threshold = 1e-5
    n_zeros = np.sum(np.abs(w_final_vector) < w_threshold)
    sparsity = n_zeros / len(w_final_vector)
    print(f"\nEsparsidad del modelo final: {sparsity:.2%} (Ceros: {n_zeros}/{len(w_final_vector)})")

else:
    print("Error: No se pudo reconstruir w_final.")

# 4. Gráfico de Convergencia (Opcional pero recomendado)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(opt_vals, label="Master UB (Loss)")
# plt.plot(gen_col.lb_fin, label="Pricing LB") # Si quieres ver el gap
plt.xlabel("Iteraciones")
plt.ylabel("Valor Objetivo")
plt.title("Convergencia Generación de Columnas DW ACCELERADO CON TOL ACTUALIZADO y sin warm start 1e4 PRESOLVE 0")
plt.legend()
plt.grid(True)
plt.show()



# In[1]:  Automatizar .
import pickle
import datetime
ii=3
# Generar nombre único
timestamp = datetime.datetime.now().strftime("%H%M%S")
filename = f"RESPALDO_URGENTE_{timestamp}_console_{ii}.pkl"

print(f"⚡ INICIANDO RESPALDO DE EMERGENCIA en {filename}...")

try:
    # Opción A: Guardar el objeto completo (gen_col)
    # Esto guarda TODO: K_points, K_rays, historiales, parámetros, etc.
    with open(filename, 'wb') as f:
        pickle.dump(gen_col, f)
    print(f"\n✅ ¡LISTO! Objeto 'gen_col' guardado exitosamente.")
    print("Puedes cerrar y apagar el computador con seguridad.")

except Exception as e:
    print(f"\n⚠️ Falló el guardado del objeto completo: {e}")
    print("Intentando guardar SOLO los diccionarios de columnas (Lo más valioso)...")
    
    try:
        # Opción B: Guardar solo la data crítica si el objeto falla
        data_critica = {
            "K_points": gen_col.K_points_dict,
            "K_rays": gen_col.K_rays_dict,
            "opt_vals": gen_col.opt_val_fin,
            "lb_fin": gen_col.lb_fin
        }
        filename_data = f"RESPALDO_DATA_{timestamp}.pkl"
        with open(filename_data, 'wb') as f:
            pickle.dump(data_critica, f)
        print(f"\n✅ ¡SALVADO! Datos críticos guardados en {filename_data}")
    except Exception as e2:
        print("a")
# In[1]:  Automatizar .
import pickle

# Cargar
with open('RESPALDO_URGENTE_XXXXXX.pkl', 'rb') as f:
    gen_col_recuperado = pickle.load(f)

# Reanudar (si la clase lo permite) o extraer datos
print(len(gen_col_recuperado.K_points_dict))

