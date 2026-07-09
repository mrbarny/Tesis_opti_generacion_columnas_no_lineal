# -*- coding: utf-8 -*-
"""
=============================================================================
SCRIPT BATCH DE EXPERIMENTACIÓN: GENERACIÓN DE COLUMNAS ACELERADA (DW)
=============================================================================
Procesa de forma sistemática los experimentos dentro de la carpeta
`Experimentos_K_ini_Resultados`, utilizando el algoritmo de Generación
de Columnas (Dantzig-Wolfe) cónico implementado en `funciones_tesis_version_sparse_fast_2.py`.

Guarda para cada experimento:
- Tiempos de procesamiento por iteración (Master y Pricing) y tiempo total.
- Valores óptimos primales (UB) y duales (LB) por iteración.
- Gráficos de convergencia Primal-Dual, métricas de gap, tiempos y mapas de calor
  de utilización de soluciones (Puntos) y directions extremas (Rayos).
- Resumen global tabular en CSV y Joblib.

Autores: Matías Muñoz Flores & Colaborador
Año: 2026
"""

import os
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import io

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import gc
import time
import argparse
import datetime
from pathlib import Path

# Configurar matplotlib en modo no interactivo ('Agg') para ejecuciones batch en segundo plano
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

# Importar motor core y visualizaciones de tesis
from funciones_tesis_version_sparse_fast_2 import (
    GeneracionColumnasDW,
    solve_pricing_problem_caja,
    solve_svm_conic
)
from test_gurobi_unbd import pricing_gurobi
from visualizaciones_tesis_2 import (
    plot_convergence_paper,
    plot_convergence_metrics,
    plot_convergence_split,
    plot_computation_times,
    plot_heatmap_memoria_robusto
)


def sanear_historial_lb(raw_lb):
    """
    Sanea el historial de cota inferior (Lower Bound) proveniente del Pricing,
    reemplazando valores infinitos o banderas de fallback por cotas progresivas.
    """
    lb_history = np.copy(raw_lb)
    last_valid = -100.0
    for i in range(len(lb_history)):
        val = lb_history[i]
        if np.isnan(val) or val == -np.inf or val == -6742069 or val <= -1e6:
            lb_history[i] = last_valid
        else:
            last_valid = max(last_valid, val)
            lb_history[i] = last_valid
    return lb_history


def cargar_datos_benchmark(bench_dir):
    """
    Carga el dataset escalado e identifica los conjuntos K iniciales en el directorio.
    Soporta formato 'X_both'/'y' o 'X_train'/'y_train'.
    """
    data_path = bench_dir / "dataset_escalado.joblib"
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró dataset_escalado.joblib en {bench_dir}")
        
    data_dict = joblib.load(data_path)
    if "X_both" in data_dict:
        X = data_dict["X_both"]
        y = data_dict["y"]
    elif "X_train" in data_dict:
        X = data_dict["X_train"]
        y = data_dict["y_train"]
    else:
        raise KeyError("El archivo joblib no contiene 'X_both' ni 'X_train'.")

    # Asegurar que las etiquetas y estén en {-1, 1}
    y = np.where(y <= 0, -1, 1)

    # Buscar archivos de K canónicos y K forest
    k_canonicos_files = sorted(bench_dir.glob("K_canonico_*.pkl"))
    k_forest_files = sorted(bench_dir.glob("K_forest_*.pkl"))

    K_ini_rays = []
    if k_canonicos_files:
        K_ini_rays = joblib.load(k_canonicos_files[0])
        print(f"  -> Rayos iniciales cargados desde {k_canonicos_files[0].name}: {len(K_ini_rays)} columnas.")

    K_ini_points = []
    if k_forest_files:
        K_ini_points = joblib.load(k_forest_files[0])
        print(f"  -> Puntos iniciales cargados desde {k_forest_files[0].name}: {len(K_ini_points)} columnas.")

    return X, y, K_ini_points, K_ini_rays


def procesar_experimento(bench_dir, max_iter=200, calc_conic_opt=True):
    """
    Ejecuta el pipeline completo de Generación de Columnas para las 9 combinaciones
    experimentales (3 inputs x 3 pricings) sobre un dataset individual de benchmark.
    """
    bench_name = bench_dir.name
    print("\n" + "=" * 70)
    print(f">>> INICIANDO BENCHMARK (9 EXPERIMENTOS): {bench_name}")
    print("=" * 70)

    # 1. Cargar datos base
    X, y, K_ini_points_forest, K_ini_rays_canonicos = cargar_datos_benchmark(bench_dir)
    n_samples, n_features = X.shape
    print(f"  [Dimensiones] n_samples = {n_samples}, n_features = {n_features}")

    # 2. Configurar tolerancias y caja según lineamientos matemáticos
    tol = max(1e-7, 1.0 / (10.0 * n_features))
    tol_master = tol
    M_box = n_features
    print(f"  [Parámetros Base] tol = {tol:.2e}, M_box = {M_box}, max_iter = {max_iter}")

    # 3. Calcular Óptimo Cónico Global de Referencia (1 sola vez por dataset)
    conic_opt = None
    if calc_conic_opt and n_features <= 100000:
        print("  -> Calculando Óptimo Cónico de Referencia (Mosek)...")
        t0_conic = time.time()
        try:
            res_con = solve_svm_conic(X, y, C=1.0, solo_w_b_xi=False)
            conic_opt = float(res_con[1])
            print(f"  [OK] Óptimo Cónico Teórico: {conic_opt:.6f} (t = {time.time() - t0_conic:.2f}s)")
        except Exception as e:
            print(f"  [WARN] No se pudo obtener el Óptimo Cónico exacto: {e}")

    # 4. Definir las 3 configuraciones de input (K inicial)
    input_configs = [
        ("canonico_puntos", K_ini_rays_canonicos, []),
        ("forest_puntos", K_ini_points_forest, []),
        ("ambos", K_ini_points_forest, K_ini_rays_canonicos)
    ]

    # 5. Definir las 3 configuraciones de pricing
    pricing_configs = [
        ("gurobi_std", pricing_gurobi, False),
        ("gurobi_acc", pricing_gurobi, True),
        ("mosek_caja", solve_pricing_problem_caja, True)
    ]

    rows_experimento = []
    exp_counter = 1

    # 6. Ejecutar las 9 combinaciones
    for input_tag, K_pts, K_rys in input_configs:
        for pricing_tag, pricing_engine, pricing_acc in pricing_configs:
            subexp_name = f"{input_tag}__{pricing_tag}"
            subexp_dir = bench_dir / subexp_name
            subexp_dir.mkdir(parents=True, exist_ok=True)

            print("\n  " + "-" * 60)
            print(f"  [{exp_counter}/9] Ejecutando: {subexp_name}")
            print("  " + "-" * 60)

            gen_col = GeneracionColumnasDW(
                tol=tol,
                tol_master=tol_master,
                M_box=M_box,
                threads=None
            )
            gen_col.ingresar_data(X, y)
            gen_col.ingresar_parametros(
                C=1.0,
                M=M_box,
                K_ini_points=K_pts,
                K_ini_rays=K_rys,
                tipo="convexo",
                pricing=pricing_engine,
                gradient_strategy="full_gradient",
                pricing_acceleration=pricing_acc,
                tijonov=False
            )

            t0_run = time.time()
            res_run = gen_col.run(
                max_iter=max_iter,
                n_periodos=20,
                frecuencia_check=40,
                max_rayos=500
            )
            t_total_seg = time.time() - t0_run
            t_total_min = t_total_seg / 60.0

            print(f"  -> Estado final: {gen_col.status} | Iteraciones: {gen_col.i} | Tiempo: {t_total_min:.2f} min")

            # Guardar visualizaciones e historial en subexp_dir
            try:
                plot_convergence_paper(gen_col, save_path=subexp_dir / "cg_convergencia_paper.png", conic_opt=conic_opt)
            except Exception as e:
                print(f"  [WARN plot_convergence_paper]: {e}")
                plt.close('all')

            try:
                plot_convergence_metrics(gen_col, save_path=subexp_dir / "cg_convergencia_metricas.png", conic_opt=conic_opt)
            except Exception as e:
                print(f"  [WARN plot_convergence_metrics]: {e}")
                plt.close('all')

            try:
                plot_convergence_split(gen_col, save_path=subexp_dir / "cg_convergencia_split.png", conic_opt=conic_opt)
            except Exception as e:
                print(f"  [WARN plot_convergence_split]: {e}")
                plt.close('all')

            try:
                plot_computation_times(gen_col, save_path=subexp_dir / "cg_tiempos_computo.png")
            except Exception as e:
                print(f"  [WARN plot_computation_times]: {e}")
                plt.close('all')

            try:
                plot_heatmap_memoria_robusto(gen_col, save_path=subexp_dir / "cg_heatmap_puntos.png", tipo="puntos")
            except Exception as e:
                print(f"  [WARN plot_heatmap_puntos]: {e}")
                plt.close('all')

            try:
                plot_heatmap_memoria_robusto(gen_col, save_path=subexp_dir / "cg_heatmap_rayos.png", tipo="rayos")
            except Exception as e:
                print(f"  [WARN plot_heatmap_rayos]: {e}")
                plt.close('all')

            ub_history = np.array(gen_col.opt_val_fin, dtype=float)
            lb_history = sanear_historial_lb(gen_col.historial_cotas_inferiores)

            detalle_dict = {
                "benchmark": bench_name,
                "input_mode": input_tag,
                "pricing_mode": pricing_tag,
                "iteraciones": gen_col.i,
                "time_master": gen_col.time_master,
                "time_pricing": gen_col.time_pricing,
                "tiempo_total_seg": t_total_seg,
                "ub_history": ub_history,
                "lb_history": lb_history,
                "conic_opt": conic_opt,
                "status": gen_col.status,
                "n_points": len(gen_col.K_points_dict),
                "n_rays": len(gen_col.K_rays_dict)
            }
            joblib.dump(detalle_dict, subexp_dir / "cg_historial_completo.joblib")

            ub_final = float(ub_history[-1]) if len(ub_history) > 0 else np.nan
            lb_final = float(lb_history[-1]) if len(lb_history) > 0 else np.nan
            gap_final = abs(ub_final - lb_final) if not np.isnan(ub_final) and not np.isnan(lb_final) else np.nan

            rows_experimento.append({
                "benchmark": bench_name,
                "input_mode": input_tag,
                "pricing_mode": pricing_tag,
                "n_samples": int(n_samples),
                "n_features": int(n_features),
                "iteraciones": int(gen_col.i),
                "tiempo_total_min": round(t_total_min, 3),
                "tiempo_total_seg": round(t_total_seg, 2),
                "ub_inicial": float(ub_history[0]) if len(ub_history) > 0 else np.nan,
                "ub_final": round(ub_final, 7),
                "lb_final": round(lb_final, 7),
                "gap_final": round(gap_final, 7),
                "conic_opt": round(conic_opt, 7) if conic_opt is not None else np.nan,
                "n_puntos_final": int(len(gen_col.K_points_dict)),
                "n_rayos_final": int(len(gen_col.K_rays_dict)),
                "status": str(gen_col.status)
            })

            del gen_col
            gc.collect()
            exp_counter += 1

    del X, y
    gc.collect()
    return rows_experimento


def main():
    parser = argparse.ArgumentParser(description="Ejecutor Batch de Experimentos de Generación de Columnas (CG / DW)")
    parser.add_argument("--max-iter", type=int, default=200, help="Límite máximo de iteraciones por experimento (defecto: 200)")
    parser.add_argument("--benchmarks", nargs="*", default=None, help="Lista opcional de nombres de carpetas de benchmark a procesar (defecto: procesa todos)")
    parser.add_argument("--no-conic-opt", action="store_true", help="Desactivar el cálculo del óptimo cónico completo de referencia")
    args = parser.parse_args()

    absolute_path = Path(__file__).resolve()
    dir_path = absolute_path.parent
    base_output_dir = dir_path / "Experimentos_K_ini_Resultados"

    if not base_output_dir.exists():
        raise FileNotFoundError(f"No se encontró el directorio de resultados: {base_output_dir}")

    # Detectar subdirectorios de experimentos
    subdirs = [d for d in sorted(base_output_dir.iterdir()) if d.is_dir() and (d / "dataset_escalado.joblib").exists()]

    if args.benchmarks:
        subdirs = [d for d in subdirs if d.name in args.benchmarks]

    print("=" * 70)
    print(f"PIPELINE BATCH GENERACIÓN DE COLUMNAS: {len(subdirs)} CARPETAS DETECTADAS (9 EXPERIMENTOS C/U)")
    print(f"Directorio Raíz: {base_output_dir}")
    print("=" * 70)

    resumen_list = []
    t_inicio_batch = time.time()

    for idx, bench_dir in enumerate(subdirs, 1):
        print(f"\n[{idx}/{len(subdirs)}] Procesando {bench_dir.name}...")
        try:
            rows_resumen = procesar_experimento(
                bench_dir,
                max_iter=args.max_iter,
                calc_conic_opt=not args.no_conic_opt
            )
            resumen_list.extend(rows_resumen)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [ERROR] Error en experimento {bench_dir.name}: {e}")

        # Guardado incremental de la tabla resumen
        if resumen_list:
            df_resumen = pd.DataFrame(resumen_list)
            csv_path = base_output_dir / "resumen_global_experimentos_CG.csv"
            joblib_path = base_output_dir / "resumen_global_experimentos_CG.joblib"
            df_resumen.to_csv(csv_path, index=False)
            joblib.dump(df_resumen, joblib_path)

    t_total_batch = (time.time() - t_inicio_batch) / 60.0
    print("\n" + "=" * 70)
    print(f"[COMPLETED] PROCESAMIENTO BATCH COMPLETADO EN {t_total_batch:.2f} MINUTOS.")
    print("=" * 70)
    if resumen_list:
        df_resumen = pd.DataFrame(resumen_list)
        print(df_resumen[["benchmark", "input_mode", "pricing_mode", "iteraciones", "tiempo_total_min", "ub_final", "lb_final", "gap_final", "status"]].to_string(index=False))
        print(f"\nResumen global guardado en: {base_output_dir / 'resumen_global_experimentos_CG.csv'}")


if __name__ == "__main__":
    main()
