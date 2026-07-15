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

    # Liberar diccionario intermedio inmediatamente
    del data_dict
    gc.collect()

    # Convertir X estrictamente a matriz dispersa para prevenir MemoryError en matrices 10000x100000
    if not sp.issparse(X):
        print("  -> Convirtiendo X a scipy.sparse.csr_matrix para optimizar consumo de RAM...")
        X = sp.csr_matrix(X)
    elif not isinstance(X, sp.csr_matrix):
        X = X.tocsr()

    # Asegurar que las etiquetas y estén en {-1, 1} de tipo compacto int8
    y = np.where(y <= 0, -1, 1).astype(np.int8)

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


def procesar_experimento(bench_dir, max_iter=200, calc_conic_opt=True, pricing_mode="gurobi_acc"):
    """
    Ejecuta el pipeline completo de Generación de Columnas para las combinaciones
    experimentales sobre un dataset individual de benchmark.
    Por defecto ejecuta los 3 K inputs con Gurobi Acelerado (gurobi_acc, ~90 min total por carpeta).
    """
    bench_name = bench_dir.name
    print("\n" + "=" * 70)
    print(f">>> INICIANDO BENCHMARK: {bench_name} (pricing_mode={pricing_mode})")
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

    # 3. Calcular Óptimo Cónico Global de Referencia (1 sola vez por dataset si n_features < 100000)
    conic_opt = None
    if calc_conic_opt and n_features < 100000:
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

    # 5. Definir configuraciones de pricing disponibles
    all_pricing = [
        ("gurobi_std", pricing_gurobi, False),
        ("gurobi_acc", pricing_gurobi, True),
        ("mosek_caja", solve_pricing_problem_caja, True)
    ]
    if pricing_mode == "all":
        pricing_configs = all_pricing
    else:
        pricing_configs = [p for p in all_pricing if p[0] == pricing_mode]
        if not pricing_configs:
            pricing_configs = [("gurobi_acc", pricing_gurobi, True)]

    total_subexps = len(input_configs) * len(pricing_configs)
    print(f"  -> Total de sub-experimentos programados para esta carpeta: {total_subexps}")

    rows_experimento = []
    exp_counter = 1

    # 6. Ejecutar las 9 combinaciones
    for input_tag, K_pts, K_rys in input_configs:
        for pricing_tag, pricing_engine, pricing_acc in pricing_configs:
            subexp_name = f"{input_tag}__{pricing_tag}"
            subexp_dir = bench_dir / subexp_name
            subexp_dir.mkdir(parents=True, exist_ok=True)

            print("\n  " + "-" * 60)
            print(f"  [{exp_counter}/{total_subexps}] Ejecutando: {subexp_name}")
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
                tipo="afin",
                pricing=pricing_engine,
                gradient_strategy="full_gradient",
                pricing_acceleration=pricing_acc,
                tijonov=False
            )

            t0_run = time.time()
            res_run = gen_col.run(
                max_iter=max_iter,
                n_periodos=20,
                frecuencia_check=20,
                max_rayos=500
            )
            t_total_seg = time.time() - t0_run
            t_total_min = t_total_seg / 60.0

            print(f"  -> Estado final: {gen_col.status} | Iteraciones: {gen_col.i} | Tiempo: {t_total_min:.2f} min")

            # Obtener historiales primal y dual saneados
            ub_history = np.array(gen_col.opt_val_fin, dtype=float)
            lb_raw = getattr(gen_col, "lb_fin", getattr(gen_col, "historial_cotas_inferiores", []))
            lb_history = sanear_historial_lb(lb_raw)

            # Guardar visualizaciones e historial en subexp_dir
            history_dict = {
                'iter': list(range(len(ub_history))),
                'master_value': ub_history,
                'pricing_value': lb_history
            }

            try:
                plot_convergence_paper(history_dict, filename=subexp_dir / "cg_convergencia_paper.png", conic_opt=conic_opt)
            except Exception as e:
                print(f"  [WARN plot_convergence_paper]: {e}")
                plt.close('all')

            try:
                plot_convergence_metrics(history_dict, conic_opt=conic_opt, dataset_size=f"{n_samples}x{n_features}", filename=subexp_dir / "cg_convergencia_metricas.png")
            except Exception as e:
                print(f"  [WARN plot_convergence_metrics]: {e}")
                plt.close('all')

            try:
                plot_convergence_split(history_dict, filename=subexp_dir / "cg_convergencia_split.png", conic_opt=conic_opt)
            except Exception as e:
                print(f"  [WARN plot_convergence_split]: {e}")
                plt.close('all')

            try:
                plot_computation_times(gen_col.time_master, gen_col.time_pricing, filename=subexp_dir / "cg_tiempos_computo.png")
            except Exception as e:
                print(f"  [WARN plot_computation_times]: {e}")
                plt.close('all')

            try:
                plot_heatmap_memoria_robusto(gen_col.memoria_theta, filename=subexp_dir / "cg_heatmap_puntos.png", tipo="Puntos")
            except Exception as e:
                print(f"  [WARN plot_heatmap_puntos]: {e}")
                plt.close('all')

            try:
                plot_heatmap_memoria_robusto(gen_col.memoria_mu, filename=subexp_dir / "cg_heatmap_rayos.png", tipo="Rayos")
            except Exception as e:
                print(f"  [WARN plot_heatmap_rayos]: {e}")
                plt.close('all')

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
    parser.add_argument("--max-iter", type=int, default=50, help="Límite máximo de iteraciones por experimento (defecto: 50)")
    parser.add_argument("--base-dir", type=str, default="Experimentos_K_ini_Resultados", help="Carpeta base de los experimentos (defecto: Experimentos_K_ini_Resultados)")
    parser.add_argument("--benchmarks", nargs="*", default=None, help="Lista opcional de nombres de carpetas de benchmark a procesar (defecto: procesa todos)")
    parser.add_argument("--no-conic-opt", action="store_true", help="Desactivar el cálculo del óptimo cónico completo de referencia")
    parser.add_argument("--pricing-mode", type=str, default="gurobi_acc", choices=["gurobi_acc", "gurobi_std", "mosek_caja", "all"], help="Modo de pricing a ejecutar (defecto: gurobi_acc para 3 experimentos por carpeta)")
    args = parser.parse_args()

    absolute_path = Path(__file__).resolve()
    dir_path = absolute_path.parent
    base_output_dir = dir_path / args.base_dir

    if not base_output_dir.exists():
        raise FileNotFoundError(f"No se encontró el directorio de resultados: {base_output_dir}")

    # Detectar subdirectorios de experimentos y dejar instancias 10000x100000 para el final
    subdirs_all = [d for d in base_output_dir.iterdir() if d.is_dir() and (d / "dataset_escalado.joblib").exists()]
    subdirs_normal = sorted([d for d in subdirs_all if "10000x100000" not in d.name], key=lambda x: x.name)
    subdirs_huge = sorted([d for d in subdirs_all if "10000x100000" in d.name], key=lambda x: x.name)
    subdirs = subdirs_normal + subdirs_huge

    if args.benchmarks:
        subdirs = [d for d in subdirs if d.name in args.benchmarks]

    num_exps_por_carpeta = 9 if args.pricing_mode == "all" else 3
    print("=" * 70)
    print(f"PIPELINE BATCH GENERACIÓN DE COLUMNAS: {len(subdirs)} CARPETAS DETECTADAS ({num_exps_por_carpeta} EXPERIMENTOS C/U)")
    print(f"Directorio Raíz: {base_output_dir} | Modo Pricing: {args.pricing_mode}")
    print("=" * 70)

    reporte_path = base_output_dir / "reporte.txt"
    with open(reporte_path, "a", encoding="utf-8") as f_rep:
        f_rep.write(f"\n--- INICIO LOTE EXPERIMENTOS ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---\n")

    resumen_list = []
    t_inicio_batch = time.time()

    for idx, bench_dir in enumerate(subdirs, 1):
        print(f"\n[{idx}/{len(subdirs)}] Procesando {bench_dir.name}...")
        with open(reporte_path, "a", encoding="utf-8") as f_rep:
            f_rep.write(f"[{idx}/{len(subdirs)}] Iniciando carpeta {bench_dir.name}\n")

        try:
            rows_resumen = procesar_experimento(
                bench_dir,
                max_iter=args.max_iter,
                calc_conic_opt=not args.no_conic_opt,
                pricing_mode=args.pricing_mode
            )
            resumen_list.extend(rows_resumen)
            with open(reporte_path, "a", encoding="utf-8") as f_rep:
                f_rep.write(f"  [OK] Carpeta {bench_dir.name} procesada exitosamente.\n")
        except MemoryError as me:
            print(f"  [WARN] MemoryError detectado en {bench_dir.name}. Limpiando RAM y reintentando una vez...")
            with open(reporte_path, "a", encoding="utf-8") as f_rep:
                f_rep.write(f"  [WARN] MemoryError en {bench_dir.name}. Reintentando tras gc.collect()...\n")
            gc.collect()
            try:
                rows_resumen = procesar_experimento(
                    bench_dir,
                    max_iter=args.max_iter,
                    calc_conic_opt=not args.no_conic_opt,
                    pricing_mode=args.pricing_mode
                )
                resumen_list.extend(rows_resumen)
                with open(reporte_path, "a", encoding="utf-8") as f_rep:
                    f_rep.write(f"  [OK] Carpeta {bench_dir.name} procesada exitosamente tras reintento.\n")
            except Exception as e2:
                print(f"  [SALTADO] Carpeta {bench_dir.name} omitida tras segundo error de memoria: {e2}")
                with open(reporte_path, "a", encoding="utf-8") as f_rep:
                    f_rep.write(f"  [SALTADO] Carpeta {bench_dir.name} omitida: {e2}\n")
                gc.collect()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [ERROR] Error en experimento {bench_dir.name}: {e}")
            with open(reporte_path, "a", encoding="utf-8") as f_rep:
                f_rep.write(f"  [ERROR] {bench_dir.name}: {e}\n")
            gc.collect()

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
