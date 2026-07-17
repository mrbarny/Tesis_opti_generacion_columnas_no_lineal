
"""
Script de Generación de Datasets y Conjuntos K Iniciales para Experimentos
"""

import os
import time
import joblib
import numpy as np
from pathlib import Path
from sklearn.datasets import make_classification

# Importar tus funciones personalizadas
from funciones_tesis_version_sparse_fast_2 import K_forest_2, generar_K_canonico_sparse
# NOTA: Asegúrate de tener esta función (o su equivalente) en tu archivo


def crear_data(tamaño=[1e3,1e4], classes=2, info=0.5, redun=0.3, rep=0.2, peso=None,
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



# =============================================================================
# 1. DEFINICIÓN DE DIMENSIONALIDAD Y BENCHMARKS
# =============================================================================
# Reducimos la dimensionalidad a 1000 x 10000 para experimentos más ágiles
tamano_exp = [1000, 10000] 
benchmarks = {}
#pruebas=10
for i in range(3,4):
    benchmarks[f"texto_emb_{i}_1000x10000"]= dict(tamaño=tamano_exp, info=0.1, redun=0.5, rep=0.0, flipy=0.1, disper=0.3, escalar=8.0, desplazar=2.0, hypercubo=False,shufflear=True, semilla=i)



# =============================================================================
# 2. CONFIGURACIÓN DE DIRECTORIOS
# =============================================================================
absolute_path = Path(__file__).resolve() if '__file__' in locals() else Path().resolve()
dir_path = absolute_path.parent  
base_output_dir = dir_path / "Experimentos_K_ini_Resultados"
base_output_dir.mkdir(parents=True, exist_ok=True)

# Parámetros para el K_forest
n_iters_forest = 10 #cuantas veces se repite el proceso de generar sub_problemas haciendo las diviciones al azar. 
tamaño=0.1  # Ajustable según lo necesites #10% canonicos random. 
#n_data deberia ser igual a fixed_sample_size * n_iters_forest. Asi te aseguras en promedio todo el dataset incluido bien. #10x100 = 1000 como minimo
partes=100 #son a dividir los features
#n_features se cubre igual todo gracias a la sub seleccion de features. Solapar es clave.

sample_size_frac= 10 # n_samples/ sample_size_frac

print(f"Iniciando procesamiento de {len(benchmarks)} benchmarks...")
print("-" * 50)


import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler

# =============================================================================
# 3. BUCLE PRINCIPAL DE EXPERIMENTACIÓN
# =============================================================================
for bench_name, bench_params in benchmarks.items():
    gc.collect()
    print(f"\nProcesando Benchmark: {bench_name.upper()}")
    
    # Crear carpeta específica para este benchmark
    bench_dir = base_output_dir / bench_name
    bench_dir.mkdir(parents=True, exist_ok=True)
    
    # A) Generación y Preprocesamiento de Datos
    print("  -> Generando datos y aplicando MaxAbsScaler...")
    X, y = crear_data(**bench_params)
    y = np.where(y <= 0, -1, 1)
    
#    tfidf_vect=TfidfVectorizer(ngram_range=(1, 1),min_df=1) #,max_df=0.95) #Es PA TEXTOS 
#    tfidf_vect.fit(X)
#    X_tfidf = tfidf_vect.transform(X) #transforma textos!!
    scaler = MaxAbsScaler() 
#    X_ms=scaler.fit_transform(X)
    X = scaler.fit_transform(X) #revisar porque no es SPARSE. 
    # Guardar dataset procesado
    data_dict = {
        "X_both": X, 
        "y": y,
        'tipo': 'tfidf and MaxAbsScaler'#,        'X_tfidf':X_tfidf, 'X_ms':X_ms
    }
    joblib.dump(data_dict, bench_dir / "dataset_escalado.joblib")
#    del X_tfidf,X_ms
    n_samples, n_features = X.shape
    
    # B) Conjunto 1: Vectores Canónicos Aleatorios
    print("  -> [1/3] Generando Vectores Canónicos Aleatorios...")
    K_ini_canonico_rand = generar_K_canonico_sparse(
        n_features=n_features, 
        n_samples=n_samples, 
        tamaño=tamaño, # Ajustable según lo necesites #10% canonicos random. 
        ambos_signos=True # Genera pares diametralmente opuestos (+e_i y -e_i)
    )
    joblib.dump(K_ini_canonico_rand, bench_dir / f'K_canonico_aleatorio_{tamaño*100}%.pkl')
    
    # C) Conjunto 2: Solución Forest + Vectores Canónicos (Combinados)
    print("  -> [2/3] Generando Soluciones K_forest...")

    t0 = time.time()
    K_forest = K_forest_2(
        X, y, 
        n_iters=n_iters_forest, 
        partes=partes + 1, #cantidad de partes en que se dividira los features
        time_max=120, 
        tol=1e-5, 
        keep_xi=False, 
        solapar=True,
        fixed_sample_size=int( n_samples/sample_size_frac) # Activara el sub sampling, es decir, armar datasets mas pequeños y manejables que n_data
    )
    print(f"     Tiempo K_forest: {time.time()-t0:.2f}s")
    filas=n_samples/sample_size_frac
    columnas=n_features/partes
    
    joblib.dump(K_forest, bench_dir / f'K_forest_n_iters{n_iters_forest}_{filas}x{columnas}.pkl')
    
    print(f"  ✅ Benchmark '{bench_name}' procesado y guardado en: {bench_dir.name}/")

print("\n" + "=" * 50)
print("PROCESO COMPLETADO EXITOSAMENTE.")
