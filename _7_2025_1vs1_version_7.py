# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:55:12 2025

@author: matta
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # regular expression
import nltk #no me acuerdo lmao 
from funciones_tesis_version_sparse import *
import gc; gc.collect()

# In[1]: Usaremos la data construida en _7_2025_1vs1_version_5.

#data=pd.read_csv("texto_blog_limpios_completo.csv",keep_default_na=False, 
#    na_values=['']) # 681284,9 tamaño
sub_data=pd.read_csv("sub_texto_blog_limpios.csv",keep_default_na=False, 
    na_values=['']) # 17318,9 

#leerlos se demora 1 min.
# In[2]:
#sub_data=data[data["id"].isin([449628,734562,589736,1975546,958176,1476382,470861,780903,665500,1151815])]
# a la hora de construir databases de experimento, se debe separar antes de armar las matrices de vocabulario.  


#si cambias a sub_data, usas menos. 
X=sub_data.clean_data
y=sub_data.clase
X= X.fillna('')


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=2,test_size = 0.2, stratify=y) # este sera enorme, cuidado. 

cvect=CountVectorizer(ngram_range=(1,3)) #13 186 961 tamaño con 1,2 full data. # con sub_data es 1,3 de tamaño 2 196 391
# In[3]: #aque se aprenda el idioma
cvect.fit(X_train)
#Check the vocablury size
print(len(cvect.vocabulary_))


# In[3]: guardar el vocabulario
    
from joblib import dump, load
import os

ruta_archivo = 'count_vectorizer_vocabulario_sub_data.joblib' #quitar _sub_data para full 

dump(cvect, ruta_archivo)

#para cargarlo se usa cvect = load(ruta_archivo)


# load(ruta_archivo)

# In[4]: Se debe transformar a matriz de vocabulario como matriz sparce. 

X_train_ct=cvect.transform(X_train) # 545k x 13187k en full data # sub data es 4k x 2200k 
X_test_ct = cvect.transform(X_test) #

y_train_pm = np.where(y_train <= 0, -1, 1)
y_test_pm = np.where(y_test <= 0, -1, 1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=False) #se requiere with_mean=False porque es sparce y deja la caga des sparcearla 

scaler.fit(X_train_ct)
#normalizamos por rendimiento. 
X_train_ct_scaled = scaler.transform(X_train_ct)
X_test_ct_scaled = scaler.transform(X_test_ct) #se demora como 10 min

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect=TfidfVectorizer(ngram_range=(1,3))

tfidf_vect.fit(X_train)
#normalizamos por rendimiento. 
X_train_tfidf = tfidf_vect.transform(X_train)
X_test_tfidf =tfidf_vect.transform(X_test) #se demora como 10 min

del X_train_ct, X_test_ct, cvect, scaler #liberar memoria 
del X, X_test, X_train, y, y_test, y_train, sub_data


# In[5]:¿Sera clasificable por sklearn? 
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import time 


start_lsvc=time.time()

skl_sbm_r= skl_svm(X_train_ct_scaled,y_train_pm,X_test_ct_scaled,C=1.0, loss="hinge", max_iter=50000, solo_w_b_xi=False)

end_lsvc=time.time()

time_lsvc= end_lsvc-start_lsvc

print("Accuracy:", accuracy(y_test_pm, skl_sbm_r[2]))

print("el tiempo de ejecución fue de ",time_lsvc, "segundos")

from sklearn.metrics import classification_report

print(classification_report(y_test_pm, skl_sbm_r[2]))

# In[6]: Los resultados
resultados_=[]
#al parecer, este esta optimizado para datasets con muchas features
#no utiliza kerner=lineal pues se salta la matriz kernel. Se tira de hozico.
C=1.0
tol=1e-6
M=1e4
y_pred=skl_sbm_r[2]
acc = accuracy(y_test_pm, y_pred)
report = classification_report(
    y_test_pm, y_pred, output_dict=True, zero_division=0)

resultado_skl_sbm_r={
    "accuracy": acc,
    "f1_macro": report["macro avg"]["f1-score"],
    "f1_weighted": report["weighted avg"]["f1-score"],
    "norm_w": np.linalg.norm(skl_sbm_r[0][0]),
    "sum_xi": np.sum(skl_sbm_r[0][2]),
    "slacks_positivos": np.sum(skl_sbm_r[0][2] > tol),
    "obj_val": skl_sbm_r[1],
    "tiempo": time_lsvc,
    "tipo": "SKL"
}
resultados_.append(resultado_skl_sbm_r) #

# CON FULL DATA 1,3 EXPLOTA. 10 HORAS PARA TIRAR ERROR Y TENER ACCURACY DE 10% EN LA CLASE 1. 
# In[6]: Ahora optimizacion

solve_svm_conic(X_train_tfidf,y_train_pm)

#NO ME MOLESTARE EN ECHARLO A CORRER. 
# In[6]: Ahora armar los sets iniciales. 


#ESTE ES CON LOS SUB PROBLEMAS

K_ini_subprob,column_sets,df_log= crear_sub_problema(X_train_ct_scaled,y_train_pm,1001,keep_xi=True)



#K_ini_subprob.insert(0,generar_canonico(K_ini_subprob[1][0],-1)) #le agregamos el vector de 0 porque somos cools y ayuda a que no muera. 
#interesante en alrededor de 7000 filas muere x todas columnas
#ESTE ES CON LOS CANONICOS Se debe arreglar porque queda sin memoria. 
#K_ini_canonico= generar_K_canonico(K_ini_subprob[1][0],tamaño=0.1)
# In[6]: ELSET DE LOS CANONICOS

# Obtienes las dimensiones una vez
n_samples, n_features = X_train_ct_scaled.shape 

# Creas el K inicial pasándole los enteros
K_ini_canonico_list = generar_K_canonico_sparse(
    n_features=n_features, 
    n_samples=n_samples, 
    tamaño=0.01
)



# In[6]:  OPCIONAL: Ahora armar los sets iniciales conicamente alterados. 

print("--- Pre-processing: Solving Conic Master on K_ini_subprob ---")
K_filtrar=K_ini_subprob+K_ini_canonico_list
# Use solve_master_primal_v2 as it returns the most info, though we mainly need theta
conic_master_results = solve_master_primal_v2(
    X_train_ct_scaled, y_train_pm, K_filtrar, 
    tipo="conico", # Specify the conic combination
    C=1.0, 
    tol=1e-6,
    mosek_params=mosek_params_from_tol(1e-6, presolve_level=1, optimizer_code=0) # Use default solver settings
)

# Extract the theta weights
theta_conico = conic_master_results[0]
conic_master_obj = conic_master_results[4] # Get the objective value (~12 you mentioned)

if theta_conico is None:
    raise ValueError("Conic Master failed during pre-processing!")

print(f"Conic Master Objective: {conic_master_obj:.4f}")
print(f"Number of non-zero thetas in conic solution: {np.sum(np.abs(theta_conico) > 1e-8)}")

# Clean up memory if needed
del conic_master_results
gc.collect()

print("--- Creating K_new_ini (Option B: Subset based on non-zero Thetas) ---")

active_indices = np.where(np.abs(theta_conico) > 1e-6)[0] # Use a tolerance
K_new_ini_list = [K_filtrar[i] for i in active_indices]

if not K_new_ini_list:
     raise ValueError("Conic Master resulted in all zero thetas, cannot create subset.")

# Use the xi placeholder as None since the convex master recalculates it
K_new_ini_list_cleaned = [(w, b, None) for w, b, xi in K_new_ini_list] 

# Optionally add the zero column (using a representative w length and n_samples)
#representative_w = K_new_ini_list_cleaned[0][0] 
#n_samples = len(y_train_pm) # Get number of samples
#K_new_ini_list_cleaned.append(generar_canonico(representative_w, coordenada=-1, n_samples=n_samples)) 

#K_new_ini = K_ini_version_dict(K_new_ini_list_cleaned) # Convert to dict format
#print(f"K_new_ini created with {len(K_new_ini)} columns (Subset + Zero).")

# --- Now use this K_new_ini in your convex runs ---



# In[6]: generacion de columnas 2 


import pickle
import gc
import os
import time # Asegúrate de importar time
import pandas as pd # Necesario para el DataFrame del resumen
from itertools import product
from funciones_tesis_version_sparse import * # Importa tus funciones actualizadas
K_ini_subprob= K_new_ini_list_cleaned #actualizar K_ini

run = 11 # Versión del experimento  fue 10% + sub probs.
C = 1.0
tol = 1e-6
M = 1e4 # Para lógica interna de la clase (si la usa)
max_iter = 40

# Directorio para guardar los resultados
output_dir = f"resultados_experimentos_v{run}"
os.makedirs(output_dir, exist_ok=True)
print(f"Guardando resultados en: {output_dir}")

# --- Listas de Configuraciones a Probar (REDUCIDAS) ---
tipos = ["convexo"]
pricings_dict = {
    "caja": solve_pricing_problem_caja
    #,"caja_restricto": solve_pricing_problem_combinado_sumar_restricto_2,
}
# --- MODIFICADO: Ahora el valor es solo el código del optimizador ---
solvers_dict = {
    "auto_IPM": 0
    #, "simplex_dual": 2, #el simplex dual
}
M_box_values = [ 1e4]

# Genera todas las combinaciones
combinations = list(product(tipos, pricings_dict.items(), solvers_dict.items(), M_box_values))
print(f"Total de combinaciones a ejecutar: {len(combinations)}")

# --- Bucle Principal de Experimentos ---
results_summary = []

# --- CORREGIDO: Desempaquetado correcto ---
for i, (tipo_master, (pricing_name, pricing_func), (solver_name, solver_code), m_box_val) in enumerate(combinations):

    print(f"\n===== Ejecutando Combinación {i+1}/{len(combinations)} =====")
    print(f"Tipo Master: {tipo_master}")
    print(f"Pricing: {pricing_name}")
    # --- MODIFICADO: Presolve ON para pricing ---
    presolve_pricing = 1 # Forzamos Presolve ON para el pricing
    print(f"Solver Mosek Pricing: {solver_name} (Code {solver_code}, Presolve {presolve_pricing})")
    print(f"M_box: {m_box_val}")
    print("-" * 50)

    filename_base = f"run_{tipo_master}_{pricing_name}_{solver_name}_presolve{presolve_pricing}_M{m_box_val:.0e}"
    pickle_filename = os.path.join(output_dir, filename_base + ".pkl")

    if os.path.exists(pickle_filename):
        print(f"Resultado ya existe ({pickle_filename}), saltando.")
        # Leer resumen si existe y continuar
        if os.path.exists(os.path.join(output_dir, "summary_results.csv")):
             try:
                 df_existing = pd.read_csv(os.path.join(output_dir, "summary_results.csv"))
                 if filename_base in df_existing['filename'].values:
                     existing_row = df_existing[df_existing['filename'] == filename_base].to_dict('records')[0]
                     results_summary.append(existing_row)
                     print("Entrada encontrada en resumen existente.")
                 else:
                     # Si el pickle existe pero no está en el resumen, añadir placeholder
                     results_summary.append({
                        "filename": filename_base, "status": "SKIPPED_PICKLE_EXISTS",
                        "tipo": tipo_master, "pricing": pricing_name, "solver": solver_name,
                         "M_box": m_box_val, "final_obj": None, "iterations": None, "error_msg": None
                     })

             except Exception as read_err:
                 print(f"Advertencia: No se pudo leer el resumen CSV existente: {read_err}")
        continue # Saltar al siguiente

    try:
        # 1. Crear instancia de la clase
        gcg = generacion_columnas(tol=tol, M_box=m_box_val)
        gcg.ingresar_data_(X_train_ct_scaled, y_train_pm)

        # 2. Configurar parámetros (incluyendo la función de pricing)
        gcg.ingresar_parametros_master_pricing(C=C, M=M, K_ini=K_ini_subprob,
                                               tipo=tipo_master, pricing=pricing_func)

        # 3. CREAR PARÁMETROS MOSEK DIFERENCIADOS
        # Master: Presolve ON (level=1), Optimizer Auto (code=0) - ¡Importante tener `mosek_params` en `solve_master_primal_v2`!
        master_mosek_params = mosek_params_from_tol(tol, presolve_level=1, optimizer_code=0)
        # Pricing: Presolve según 'presolve_pricing', Optimizer según 'solver_code'
        pricing_mosek_params = mosek_params_from_tol(tol, presolve_level=presolve_pricing, optimizer_code=solver_code)

        # Guardar los parámetros en la instancia para que los use internamente
        gcg.current_mosek_params = pricing_mosek_params
        # --- CORREGIDO TYPO ---
        gcg.master_mosek_params = master_mosek_params

        # 4. Ejecutar la generación de columnas
        # Ajuste en umbral_theta para evitar división por cero si n_columns es 0 (aunque no debería pasar)
        umbral_theta = 1.0 / (gcg.n_columns * 10.0) if gcg.n_columns > 0 else 1e-8
        n_periodos = max(1, int(max_iter / 10)) # Asegurar int
        frecuencia_check = max(1, int(max_iter / 5)) # Asegurar int

        # Usamos siempre el master_v2 que calcula el gradiente correcto
        gcg.run(max_iter, umbral_theta, n_periodos, frecuencia_check,
                master=solve_master_primal_v2)

        # 5. Guardar el objeto completo
        with open(pickle_filename, 'wb') as f:
            pickle.dump(gcg, f)
        print(f"Resultados guardados en: {pickle_filename}")

        # Extraer métricas clave para un resumen
        final_obj = gcg.opt_val_fin[-1] if gcg.opt_val_fin else None
        results_summary.append({
            "filename": filename_base,
            "tipo": tipo_master,
            "pricing": pricing_name,
            "solver": solver_name,
            "presolve_pricing": presolve_pricing, # Añadido para registro
            "M_box": m_box_val,
            "final_obj": final_obj,
            "iterations": gcg.i,
            "status": gcg.status,
            "error_msg": None # Sin error
        })

    except Exception as e:
        print(f"!!!!!! ERROR en la combinación {i+1} !!!!!!")
        print(f"Error: {e}")
        # Guardar info del error
        results_summary.append({
            "filename": filename_base,
            "tipo": tipo_master,
            "pricing": pricing_name,
            "solver": solver_name,
            "presolve_pricing": presolve_pricing, # Añadido para registro
            "M_box": m_box_val,
            "final_obj": None,
            "iterations": getattr(gcg, 'i', None), # Intentar obtener iteraciones si existen
            "status": "ERROR",
            "error_msg": str(e)
        })

    finally:
        # 6. Limpiar memoria
        if 'gcg' in locals():
            del gcg
        gc.collect()
        print("Memoria limpiada.")

# --- Fin del Bucle ---

# Guardar el resumen en un CSV
try:
    summary_df = pd.DataFrame(results_summary)
    summary_filename = os.path.join(output_dir, "summary_results.csv")
    summary_df.to_csv(summary_filename, index=False)
    print(f"\nResumen de experimentos guardado en: {summary_filename}")
except ImportError:
    print("\nAdvertencia: No se pudo importar pandas. El resumen no se guardará en CSV.")
except Exception as csv_err:
     print(f"\nError al guardar el resumen CSV: {csv_err}")


print("\n===== Fin de todos los experimentos =====")



# In[6]: generacion de columnas 2 




import pandas as pd
import os

run = 10 #10% canonicos
output_dir = f"resultados_experimentos_v{run}"
summary_file = os.path.join(output_dir, "summary_results.csv")

try:
    df = pd.read_csv(summary_file)
    
    # --- INICIO DE LA CORRECCIÓN ---
    # Columnas que esperamos. Si alguna no existe, la creamos con NaN.
    columnas_esperadas = [
        "pricing", "solver", "M_box", 
        "final_obj", "time_sec", "iterations", "status", "error_msg"
    ]
    
    # Lista de columnas que SÍ existen en el CSV
    columnas_existentes = []
    
    for col in columnas_esperadas:
        if col in df.columns:
            columnas_existentes.append(col)
        else:
            print(f"Advertencia: La columna '{col}' no se encontró en el CSV.")
    # --- FIN DE LA CORRECCIÓN ---

    if not columnas_existentes:
        print("El CSV está vacío o no tiene columnas útiles.")
    else:
        # Ordenar por 'final_obj' si existe, si no, no ordenar
        if "final_obj" in columnas_existentes:
            df_sorted = df[columnas_existentes].sort_values(by="final_obj")
        else:
            df_sorted = df[columnas_existentes]

        print("Resumen de Resultados:")
        print(df_sorted.to_markdown(index=False))

except FileNotFoundError:
    print(f"Error: No se encontró el archivo resumen en {summary_file}")
except Exception as e:
    print(f"Error al leer el CSV: {e}")


