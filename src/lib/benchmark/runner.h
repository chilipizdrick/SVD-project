#pragma once

#include <Eigen/Core>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "matrix_generators.h"
#include "metrics.h"
#include "query.h"
#include "solver_registry.h"
#include "storage.h"
#include "thread_pool.h"
#include "types.h"

namespace SVD_Project::Benchmark {

struct RunOptions {
  std::vector<MatrixSize> sizes{{100, 100}};
  int trials{3};
  uint64_t base_seed{42};
  int jobs{0};  // 0 => hardware_concurrency()
  bool pin_blas_threads{true};
  std::string output_path{"benchmark_results.jsonl"};
  bool resume{false};
  std::string query_str;
  QueryExprPtr filter;
  bool quiet{false};
};

struct RunSummary {
  size_t total_planned{0};
  size_t skipped_resume{0};
  size_t executed_fresh{0};
  size_t failed{0};
  double wall_seconds{0.0};
  std::vector<BenchmarkRecord> records;
};

inline void apply_blas_thread_pinning(bool pin) {
  if (!pin) return;
  Eigen::setNbThreads(1);
#if !defined(_WIN32)
  setenv("OMP_NUM_THREADS", "1", 1);
  setenv("OPENBLAS_NUM_THREADS", "1", 1);
  setenv("MKL_NUM_THREADS", "1", 1);
  setenv("BLIS_NUM_THREADS", "1", 1);
#endif
}

inline RunSummary execute_benchmark_run(const RunOptions& opt) {
  const auto& solvers = SolverRegistry::instance();
  const auto& matrices = MatrixRegistry::instance();

  int hw = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  int workers = (opt.jobs > 0) ? opt.jobs : hw;
  bool pin_blas = opt.pin_blas_threads && (workers > 1);
  if (pin_blas) apply_blas_thread_pinning(true);

  std::string env_fp = compute_env_fingerprint(workers, pin_blas);

  std::set<std::string> existing_keys;
  RunSummary summary;
  if (opt.resume && std::filesystem::exists(opt.output_path)) {
    auto existing = load_jsonl_files({opt.output_path});
    for (const auto& r : existing) {
      existing_keys.insert(r.coord.coord_key());
      summary.records.push_back(r);
    }
  }

  // Group planned tasks by matrix instance (kind, size, trial) so each matrix
  // is generated once and shared read-only across all selected solvers.
  struct MatrixGroup {
    const MatrixFamilyEntry* family{nullptr};
    MatrixSize size;
    int trial{0};
    uint64_t seed{0};
    std::vector<const SolverEntry*> solvers;
  };

  std::vector<MatrixGroup> groups;
  size_t total_to_run = 0;

  for (const auto& fam : matrices.entries()) {
    for (const auto& sz : opt.sizes) {
      for (int t = 0; t < opt.trials; ++t) {
        MatrixGroup grp;
        grp.family = &fam;
        grp.size = sz;
        grp.trial = t;
        grp.seed =
            derive_instance_seed(opt.base_seed, fam.name, sz.rows, sz.cols, t);

        for (const auto& sol : solvers.entries()) {
          TestCoord c{sol.algorithm, sol.diagonalizer, fam.name,
                      sz,            sol.dtype,        t};
          if (opt.filter) {
            auto ctx = QueryContext::from_coord(c, sol.concurrency);
            if (!opt.filter->eval(ctx)) continue;
          }
          ++summary.total_planned;
          if (existing_keys.count(c.coord_key())) {
            ++summary.skipped_resume;
            continue;
          }
          grp.solvers.push_back(&sol);
          ++total_to_run;
        }
        if (!grp.solvers.empty()) {
          groups.push_back(std::move(grp));
        }
      }
    }
  }

  // Ensure output directory exists and open JSONL file
  auto out_parent = std::filesystem::path(opt.output_path).parent_path();
  if (!out_parent.empty()) std::filesystem::create_directories(out_parent);

  std::ios::openmode mode =
      (opt.resume && std::filesystem::exists(opt.output_path))
          ? std::ios::app
          : std::ios::trunc;
  std::ofstream out_file(opt.output_path, mode);
  if (!out_file) {
    throw std::runtime_error("Cannot open output file: " + opt.output_path);
  }

  RunHeader hdr;
  hdr.timestamp = current_iso8601_utc();
  hdr.base_seed = opt.base_seed;
  hdr.jobs = workers;
  hdr.hardware_concurrency = hw;
  hdr.blas_threads_pinned = pin_blas;
  hdr.env_fingerprint = env_fp;
  hdr.query = opt.query_str;
  out_file << serialize_header_jsonl(hdr) << "\n" << std::flush;

  if (total_to_run == 0) {
    if (!opt.quiet) {
      std::cout << "Nothing to execute (planned=" << summary.total_planned
                << ", resumed=" << summary.skipped_resume << ").\n";
    }
    return summary;
  }

  if (!opt.quiet) {
    std::cout << "Running " << total_to_run << " benchmark tasks across "
              << workers << " worker thread(s) (hardware_concurrency=" << hw
              << ", seed=" << opt.base_seed
              << ", blas_pinned=" << (pin_blas ? "true" : "false") << ")...\n"
              << std::flush;
  }

  std::mutex out_mu;
  std::atomic<size_t> completed{0};
  std::atomic<size_t> fresh_runs{0};
  std::atomic<size_t> failed_runs{0};

  auto wall_start = std::chrono::steady_clock::now();

  {
    ThreadPool pool(static_cast<size_t>(workers));

    for (auto& grp : groups) {
      // Lazy thread-safe holder so matrix A is generated once by the first
      // worker that touches this group, shared read-only, and automatically
      // freed when the last solver task in the group finishes.
      struct SharedMatrix {
        std::once_flag flag;
        std::shared_ptr<MatrixInstance> inst;
      };
      auto holder = std::make_shared<SharedMatrix>();
      const MatrixFamilyEntry* fam = grp.family;
      MatrixSize sz = grp.size;
      int trial = grp.trial;
      int total_trials = opt.trials;
      uint64_t seed = grp.seed;

      for (const SolverEntry* sol : grp.solvers) {
        pool.enqueue([&, holder, fam, sz, trial, total_trials, seed, sol]() {
          std::call_once(holder->flag, [&]() {
            holder->inst = std::make_shared<MatrixInstance>(
                fam->generator(sz.rows, sz.cols, trial, total_trials, seed));
          });
          const MatrixInstance& inst = *holder->inst;

          TestCoord coord{sol->algorithm, sol->diagonalizer, fam->name,
                          sz,             sol->dtype,        trial};

          BenchmarkRecord rec;
          rec.coord = coord;
          rec.seed = seed;
          rec.matrix_hash = inst.matrix_hash;
          rec.concurrency = sol->concurrency;
          rec.internally_parallel = is_internally_parallel(sol->concurrency);
          rec.runner_jobs = workers;
          rec.blas_threads_pinned = pin_blas;

          try {
            auto t0 = std::chrono::high_resolution_clock::now();
            SVDOutput svd_out = sol->solve(inst.A);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            evaluate_svd_metrics(inst, svd_out, ms, rec);
          } catch (const std::exception& e) {
            rec.status = RunStatus::Failed;
            rec.error_msg = e.what();
          } catch (...) {
            rec.status = RunStatus::Failed;
            rec.error_msg = "Unknown exception";
          }

          if (rec.status != RunStatus::Ok) {
            ++failed_runs;
          }
          ++fresh_runs;

          size_t done = ++completed;
          {
            std::lock_guard<std::mutex> lock(out_mu);
            out_file << serialize_record_jsonl(rec) << "\n" << std::flush;
            summary.records.push_back(std::move(rec));
            if (!opt.quiet) {
              double pct = 100.0 * static_cast<double>(done) /
                           static_cast<double>(total_to_run);
              std::cout << "\rProgress: " << done << "/" << total_to_run
                        << " (" << std::fixed << std::setprecision(1) << pct
                        << "%, ok=" << (fresh_runs.load() - failed_runs.load())
                        << ", failed=" << failed_runs.load() << ")   "
                        << std::flush;
            }
          }
        });
      }
    }
    pool.wait_all();
  }

  auto wall_end = std::chrono::steady_clock::now();
  summary.wall_seconds =
      std::chrono::duration<double>(wall_end - wall_start).count();
  summary.executed_fresh = fresh_runs.load();
  summary.failed = failed_runs.load();

  if (!opt.quiet) {
    std::cout << "\nCompleted in " << std::fixed << std::setprecision(2)
              << summary.wall_seconds << " s. Saved to " << opt.output_path
              << "\n";
  }
  return summary;
}

}  // namespace SVD_Project::Benchmark
