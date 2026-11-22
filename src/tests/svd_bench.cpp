#include <algorithm>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "lib/benchmark/matrix_generators.h"
#include "lib/benchmark/metrics.h"
#include "lib/benchmark/query.h"
#include "lib/benchmark/runner.h"
#include "lib/benchmark/solver_registry.h"
#include "lib/benchmark/storage.h"
#include "lib/benchmark/tensor_view.h"
#include "lib/benchmark/types.h"

using namespace SVD_Project::Benchmark;

namespace {

std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) out.push_back(item);
  }
  return out;
}

std::string build_in_clause(const std::string& field,
                            const std::vector<std::string>& vals) {
  if (vals.empty()) return "";
  std::ostringstream oss;
  oss << field << " in (";
  for (size_t i = 0; i < vals.size(); ++i) {
    if (i > 0) oss << ",";
    oss << vals[i];
  }
  oss << ")";
  return oss.str();
}

void print_usage() {
  std::cout << R"(svd_bench — Parallel Tensor-Based SVD Benchmark Suite

USAGE:
  svd_bench list [--json]
  svd_bench run  [OPTIONS]
  svd_bench view <file.jsonl> [file2.jsonl ...] [OPTIONS]
  svd_bench rank <file.jsonl> [file2.jsonl ...] [OPTIONS]

RUN OPTIONS:
  -q, --query <expr>         Boolean filter expression, e.g.:
                             "algo=halley and diag in (tridiag_dc,qdwh_eig) and matrix!=hilbert"
  --algo <a,b,...>           Filter algorithms (e.g. halley,dgesdd,dgesvd,dbdsvdx,eigen_jacobi)
  --diag <d1,d2,...>         Filter diagonalizers (tridiag_dc,two_sided_jacobi,cholesky_onesided,qdwh_eig,none)
  --matrix <m1,m2,...>       Filter matrix kinds (e.g. twin_clusters,hilbert,clustered_hdr)
  --size <s1,s2,...>         Matrix sizes to generate (e.g. 100x100,500x500; default: 100x100)
  --trials <N>               Number of deterministic trials per matrix/size (default: 3)
  --seed <uint64>            Base deterministic RNG seed (default: 42)
  -j, --jobs <N>             Worker thread count (default: hardware_concurrency())
  --allow-nested-blas        Do NOT pin BLAS/OpenMP to 1 thread when jobs > 1
  -o, --output <path>        Output JSONL file (default: benchmark_results.jsonl)
  --resume                   Skip coordinates already present in output JSONL
  --quiet                    Suppress progress output

VIEW OPTIONS:
  -q, --query <expr>         Filter loaded records before slicing
  --metric <key>             Metric to display (default: omega_U; ignored if metric is row/col axis)
  --rows <axis>              Row axis: solver|algo|diag|matrix|size|problem|trial|metric (default: solver; solver = algorithm+diagonalizer combined)
  --cols <axis>              Column axis: solver|algo|diag|matrix|size|problem|trial|metric (default: matrix_kind)
  --tables <axis>            Optional 3rd axis for 3D multi-table slice (e.g. diagonalizer, problem)
  --trial-mode <mode>        mean | max | min | <trial_index> (default: mean)
  --broadcast-baselines      Duplicate diag=none baselines into each diagonalizer table
  --format <fmt>             ascii | md | csv | typst | latex (default: ascii)

RANK OPTIONS:
  -q, --query <expr>         Filter loaded records before ranking
  --metric <key>             Metric to rank by (default: r)
  --sort <criterion>         mean_rank | wins | geomean | best | worst (default: mean_rank)
)";
}

int cmd_list(bool as_json) {
  const auto& solvers = SolverRegistry::instance();
  const auto& matrices = MatrixRegistry::instance();
  int hw = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));

  if (as_json) {
    std::cout << "{\"hardware_concurrency\":" << hw << ",\"solvers\":[";
    for (size_t i = 0; i < solvers.entries().size(); ++i) {
      const auto& s = solvers.entries()[i];
      if (i > 0) std::cout << ",";
      std::cout << "{\"algorithm\":\"" << json_escape(s.algorithm)
                << "\",\"diagonalizer\":\"" << json_escape(s.diagonalizer)
                << "\",\"dtype\":\"" << json_escape(s.dtype)
                << "\",\"is_polar\":" << (s.is_polar ? "true" : "false")
                << ",\"concurrency_model\":\"" << to_string(s.concurrency)
                << "\",\"internally_parallel\":"
                << (is_internally_parallel(s.concurrency) ? "true" : "false")
                << ",\"description\":\"" << json_escape(s.description) << "\"}";
    }
    std::cout << "],\"matrices\":[";
    for (size_t i = 0; i < matrices.entries().size(); ++i) {
      const auto& m = matrices.entries()[i];
      if (i > 0) std::cout << ",";
      std::cout << "{\"name\":\"" << json_escape(m.name)
                << "\",\"has_ground_truth\":"
                << (m.has_ground_truth ? "true" : "false")
                << ",\"description\":\"" << json_escape(m.description) << "\"}";
    }
    std::cout << "],\"metrics\":[";
    const auto& mets = all_metrics_info();
    for (size_t i = 0; i < mets.size(); ++i) {
      if (i > 0) std::cout << ",";
      std::cout << "{\"key\":\"" << mets[i].key << "\",\"label\":\""
                << mets[i].label << "\",\"lower_is_better\":"
                << (mets[i].lower_is_better ? "true" : "false")
                << ",\"directional\":"
                << (mets[i].directional ? "true" : "false") << "}";
    }
    std::cout << "]}\n";
    return 0;
  }

  std::cout << "Detected hardware_concurrency: " << hw << " cores\n\n";
  std::cout << "Registered Solvers (dtype=double):\n";
  for (const auto& s : solvers.entries()) {
    std::cout << "  - " << std::left << std::setw(26)
              << (s.algorithm + "/" + s.diagonalizer) << " ["
              << std::setw(15) << to_string(s.concurrency) << "] "
              << s.description << "\n";
  }
  std::cout << "\nRegistered Matrix Kinds:\n";
  for (const auto& m : matrices.entries()) {
    std::cout << "  - " << std::left << std::setw(20) << m.name
              << (m.has_ground_truth ? "[GT] " : "[--] ") << m.description
              << "\n";
  }
  std::cout << "\nMetrics:\n  ";
  for (size_t i = 0; i < all_metrics_info().size(); ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << all_metrics_info()[i].key;
  }
  std::cout << "\n";
  return 0;
}

int cmd_run(const std::vector<std::string>& args) {
  RunOptions opt;
  std::vector<std::string> extra_clauses;

  for (size_t i = 0; i < args.size(); ++i) {
    const std::string& a = args[i];
    auto need = [&](const std::string& flag) -> std::string {
      if (i + 1 >= args.size()) {
        throw std::invalid_argument("Missing value for " + flag);
      }
      return args[++i];
    };

    if (a == "-q" || a == "--query") {
      opt.query_str = need(a);
    } else if (a == "--algo") {
      extra_clauses.push_back(build_in_clause("algorithm", split_csv(need(a))));
    } else if (a == "--diag") {
      extra_clauses.push_back(
          build_in_clause("diagonalizer", split_csv(need(a))));
    } else if (a == "--matrix") {
      extra_clauses.push_back(
          build_in_clause("matrix_kind", split_csv(need(a))));
    } else if (a == "--size") {
      opt.sizes.clear();
      for (const auto& tok : split_csv(need(a))) {
        auto sz = parse_matrix_size(tok);
        if (!sz) throw std::invalid_argument("Invalid matrix size: " + tok);
        opt.sizes.push_back(*sz);
      }
    } else if (a == "--trials") {
      opt.trials = std::stoi(need(a));
      if (opt.trials <= 0)
        throw std::invalid_argument("--trials must be >= 1");
    } else if (a == "--seed") {
      opt.base_seed = std::stoull(need(a));
    } else if (a == "-j" || a == "--jobs") {
      opt.jobs = std::stoi(need(a));
    } else if (a == "--allow-nested-blas") {
      opt.pin_blas_threads = false;
    } else if (a == "-o" || a == "--output") {
      opt.output_path = need(a);
    } else if (a == "--resume") {
      opt.resume = true;
    } else if (a == "--quiet") {
      opt.quiet = true;
    } else {
      throw std::invalid_argument("Unknown option for run: " + a);
    }
  }

  QueryExprPtr filter = parse_query(opt.query_str);
  for (const auto& cl : extra_clauses) {
    if (!cl.empty()) {
      filter = combine_and(filter, parse_query(cl));
      if (!opt.query_str.empty()) opt.query_str += " and ";
      opt.query_str += cl;
    }
  }
  opt.filter = filter;

  execute_benchmark_run(opt);
  return 0;
}

int cmd_view(const std::vector<std::string>& args) {
  std::vector<std::string> files;
  SliceOptions opt;
  std::string qstr;

  for (size_t i = 0; i < args.size(); ++i) {
    const std::string& a = args[i];
    auto need = [&](const std::string& flag) -> std::string {
      if (i + 1 >= args.size())
        throw std::invalid_argument("Missing value for " + flag);
      return args[++i];
    };
    if (a == "-q" || a == "--query") {
      qstr = need(a);
    } else if (a == "--metric") {
      opt.metric = need(a);
    } else if (a == "--rows") {
      opt.row_axis = need(a);
    } else if (a == "--cols") {
      opt.col_axis = need(a);
    } else if (a == "--tables") {
      opt.table_axis = need(a);
    } else if (a == "--trial-mode") {
      std::string tm = to_lower_copy(need(a));
      if (tm == "mean") opt.trial_mode = TrialMode::Mean;
      else if (tm == "max") opt.trial_mode = TrialMode::Max;
      else if (tm == "min") opt.trial_mode = TrialMode::Min;
      else {
        opt.trial_mode = TrialMode::Exact;
        opt.exact_trial = std::stoi(tm);
      }
    } else if (a == "--broadcast-baselines") {
      opt.broadcast_baselines = true;
    } else if (a == "--format") {
      opt.format = to_lower_copy(need(a));
    } else if (!a.empty() && a[0] == '-') {
      throw std::invalid_argument("Unknown option for view: " + a);
    } else {
      files.push_back(a);
    }
  }

  if (files.empty()) {
    throw std::invalid_argument("view requires at least one .jsonl file");
  }
  opt.filter = parse_query(qstr);

  auto records = load_jsonl_files(files);
  auto tables = build_slice_tables(records, opt);
  render_tables(std::cout, tables, opt);
  return 0;
}

int cmd_rank(const std::vector<std::string>& args) {
  std::vector<std::string> files;
  std::string metric = "r";
  std::string sort_by = "mean_rank";
  std::string qstr;

  for (size_t i = 0; i < args.size(); ++i) {
    const std::string& a = args[i];
    auto need = [&](const std::string& flag) -> std::string {
      if (i + 1 >= args.size())
        throw std::invalid_argument("Missing value for " + flag);
      return args[++i];
    };
    if (a == "-q" || a == "--query") {
      qstr = need(a);
    } else if (a == "--metric") {
      metric = need(a);
    } else if (a == "--sort") {
      sort_by = to_lower_copy(need(a));
    } else if (!a.empty() && a[0] == '-') {
      throw std::invalid_argument("Unknown option for rank: " + a);
    } else {
      files.push_back(a);
    }
  }

  if (files.empty()) {
    throw std::invalid_argument("rank requires at least one .jsonl file");
  }

  auto records = load_jsonl_files(files);
  auto rep = compute_ranking(records, metric, parse_query(qstr), sort_by);
  print_ranking_report(std::cout, rep);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage();
    return 1;
  }
  std::string cmd = argv[1];
  std::vector<std::string> rest;
  for (int i = 2; i < argc; ++i) rest.emplace_back(argv[i]);

  try {
    if (cmd == "-h" || cmd == "--help" || cmd == "help") {
      print_usage();
      return 0;
    }
    if (cmd == "list") {
      bool as_json = (!rest.empty() && rest[0] == "--json");
      return cmd_list(as_json);
    }
    if (cmd == "run") return cmd_run(rest);
    if (cmd == "view") return cmd_view(rest);
    if (cmd == "rank") return cmd_rank(rest);
    std::cerr << "Unknown subcommand: " << cmd << "\n";
    print_usage();
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
