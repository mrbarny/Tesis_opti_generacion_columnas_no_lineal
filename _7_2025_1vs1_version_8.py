# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:55:12 2025

@author: matta
"""
#INFO: VERSION QUE USA TODA LA DATA, E INCORPORA TF IDF PARA REVISAR QUE SE PORTE BIEN LA MATRIZ 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # regular expression
import nltk #no me acuerdo lmao 
from funciones_tesis_version_sparse_fast import *
import gc; gc.collect()

# In[1]: Usaremos la data construida en _7_2025_1vs1_version_5.

data=pd.read_csv("texto_blog_limpios_completo.csv",keep_default_na=False, 
    na_values=['']) # 681284,9 tamaño
#sub_data=pd.read_csv("sub_texto_blog_limpios.csv",keep_default_na=False, 
#    na_values=['']) # 17318,9 

#leerlos se demora 1 min.
# In[2]:
#sub_data=data[data["id"].isin([449628,734562,589736,1975546,958176,1476382,470861,780903,665500,1151815])]
# a la hora de construir databases de experimento, se debe separar antes de armar las matrices de vocabulario.  

# los con mas palabras son 1476382,470861,780903,665500,1151815
#los con mas textos 449628,734562,589736,1975546,958176

#si cambias a sub_data, usas menos. 
X=data.clean_data
y=data.clase
X= X.fillna('')


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=2,test_size = 0.2, stratify=y) # este sera enorme, cuidado. 


#13 186 961 tamaño con 1,2 full data. # con sub_data es 1,3 de tamaño 2 196 391

#con filtros de min_df=3 y max_df=0.95 son features de tamaño 2.727.296
# In[3]: #aque se aprenda el idioma
cvect_test=CountVectorizer(ngram_range=(1,2),min_df=3,max_df=0.95) 
cvect_test.fit(X_train) #se demora infinito
#Check the vocablury size
print(len(cvect_test.vocabulary_))



    


# In[3]: Se debe transformar a matriz de vocabulario como matriz sparce. 

X_train_ct=cvect_test.transform(X_train) # 545k x 13187k en full data # sub data es 4k x 2200k 
X_test_ct = cvect_test.transform(X_test) #

y_train_pm = np.where(y_train <= 0, -1, 1)
y_test_pm = np.where(y_test <= 0, -1, 1)
# In[4]: ANALISIS DE TEXTOS TRANSFORMADOS Y PREVIOS


# n      id   gender  ... #palabras  clase
# 10146  1151815    male  ...      13388    -1 ese es en sub data
#len(X_train.iloc[10172]) #el texto de ese indice.  Es el con mas palabras en c_train. Len incluye contar espacios varios 
#cvect_test.get_feature_names_out()[1758830] #te dice la palabra de ese indice del vocabulario, e

# In[3]:
    
    
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_df=0.95) #13 186 961
tfidf_vect.fit(X_train)
print(len(tfidf_vect.vocabulary_)) #13 542 255 para 1,3 0.95 min_df=2.  
#normalizamos por rendimiento. 
X_train_tfidf = tfidf_vect.transform(X_train)
X_test_tfidf =tfidf_vect.transform(X_test) #se demora como 10 min
y_train_pm = np.where(y_train <= 0, -1, 1)
y_test_pm = np.where(y_test <= 0, -1, 1)

#del X_train_ct, X_test_ct #liberar memoria 
del X, X_test, X_train, y, y_test, y_train
del  data
#LA DIFERENCIA ENTRE CVECT Y TFIDF ES QUE EL CVECT CUENTA FRECUENCIA DE PALABRA POR TEXTO, EL OTRO PONDERA POR QUE TAN FRECUENTE ES EN EL DOC

# In[4]: GUARDAR LOS DATOS
import joblib
from scipy.sparse import save_npz


joblib.dump(tfidf_vect, 'tfidf_vectorizer.joblib')
save_npz('X_train_tfidf.npz', X_train_tfidf)
save_npz('X_test_tfidf.npz', X_test_tfidf)
save_npz('X_train_ct.npz', X_train_ct)
save_npz('X_test_ct.npz', X_test_ct)


np.save('y_train_pm.npy', y_train_pm)
np.save('y_test_pm.npy', y_test_pm)
# In[4]: CARGAR TODOS LOS DATOS 
from scipy.sparse import load_npz
import joblib
print("Cargando artefactos pre-procesados...")

# 1. Cargar el vectorizador
# No necesitas 'ajustarlo' (fit), solo 'transformar' si tienes nuevos datos
vectorizer = joblib.load('tfidf_vectorizer.joblib')
#len(vectorizer.vocabulary_)
# 2. Cargar las matrices dispersas X
X_train_tfidf = load_npz('X_train_tfidf.npz') #puedes cambiarlo por _ct en vez de _tfidf
X_test_tfidf = load_npz('X_test_tfidf.npz')

# 3. Cargar las etiquetas Y
y_train_pm= np.load('y_train_pm.npy', allow_pickle=True)
y_test_pm = np.load('y_test_pm.npy', allow_pickle=True) 
# allow_pickle=True es a veces necesario si y era una Serie de Pandas.

print("Carga completada.")
print(f"Forma de X_train: {X_train_loaded.shape}")
print(f"Forma de y_train: {y_train_loaded.shape}")
# In[5]:¿Sera clasificable por sklearn? 
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import time 


start_lsvc=time.time()

skl_sbm_r= skl_svm(X_train_tfidf,y_train_pm,X_test_tfidf,C=1.0, loss="hinge", max_iter=100000, solo_w_b_xi=False)

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
    "tipo": "SKL_TFIDF"
}
resultados_.append(resultado_skl_sbm_r) #

# CON FULL DATA 1,3 EXPLOTA. 10 HORAS PARA TIRAR ERROR Y TENER ACCURACY DE 10% EN LA CLASE 1. 
# In[6]: Ahora optimizacion

R_CONIC= solve_svm_conic(X_train_tfidf,y_train_pm)


# caso sub_Data sin filtro pero tfidf 2 millones columnas
#el valor objetivo es 85.16838930063142
#el valor de la norma de w es 81.16837081734487
#valor de F.O calculada es 85.16838923989549

#NO ME MOLESTARE EN ECHARLO A CORRER. 
# In[6]: Ahora armar los sets iniciales. 


#ESTE ES CON LOS SUB PROBLEMAS

K_ini_subprob,column_sets,df_log= crear_sub_problema(X_train_tfidf,y_train_pm,101,keep_xi=True)



#K_ini_subprob.insert(0,generar_canonico(K_ini_subprob[1][0],-1)) #le agregamos el vector de 0 porque somos cools y ayuda a que no muera. 
#interesante en alrededor de 7000 filas muere x todas columnas
#ESTE ES CON LOS CANONICOS Se debe arreglar porque queda sin memoria. 
#K_ini_canonico= generar_K_canonico(K_ini_subprob[1][0],tamaño=0.1)




# In[6]:  OPCIONAL: Ahora armar los sets iniciales conicamente alterados. 

#EN ESTA PARTE EMPEZAREMOS A TESTEAR EL ALGORITMO. EL MASTER, EL PRICING, EL SET INI, TODO.

#K_ini_subprob,column_sets,df_log= crear_sub_problema(X_train,y_train,11,keep_xi=True)

n_samples, n_features = X_train_tfidf.shape 
print(n_features)

# Creas el K inicial pasándole los enteros
K_ini_canonico_list = generar_K_canonico_sparse(
    n_features=n_features, 
    n_samples=n_samples, 
    tamaño=0.001
)

n_iters=60 

K_ini_end=K_forest(X_train_tfidf,y_train_pm,n_iters, partes=1001, time_max=120, tol=1e-06, keep_xi=False,solapar=True)



# In[6]: generacion de columnas 2 


import pickle
import gc
import os
import time # Asegúrate de importar time
import pandas as pd # Necesario para el DataFrame del resumen
from itertools import product
from funciones_tesis_version_sparse_fast import * # Importa tus funciones actualizadas
K_ini_subprob= K_new_ini_list_cleaned #actualizar K_ini

run = 12 # Versión del experimento  fue 1% + sub probs.
C = 1.0
tol = 1e-6
M = 1e4 # Para lógica interna de la clase (si la usa)
max_iter = 10

# Directorio para guardar los resultados
output_dir = f"resultados_experimentos_tfidf_full_v{run}"
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
        gcg.ingresar_data_(X_train_tfidf, y_train_pm)

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

run = 12 #10% canonicos
#output_dir = f"resultados_experimentos_v{run}" standarizado
#output_dir = f"resultados_experimentos_tfidf_v{run}" sin standarizar
#output_dir = f"resultados_experimentos_tfidf_full_v{run}" con todo el dataset 
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


