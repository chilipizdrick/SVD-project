#pragma once

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "query.h"
#include "types.h"

namespace SVD_Project::Benchmark {

inline std::string get_axis_value(const BenchmarkRecord& r,
                                  const std::string& axis) {
  std::string a = to_lower_copy(axis);
  if (a == "algo" || a == "algorithm") return r.coord.algorithm;
  if (a == "diag_solver" || a == "solver") return r.coord.solver_key();
  if (a == "diag" || a == "diagonalizer") return r.coord.diagonalizer;
  if (a == "matrix" || a == "matrix_kind" || a == "kind")
    return r.coord.matrix_kind;
  if (a == "size" || a == "matrix_size") return r.coord.size.label();
  if (a == "problem") return r.coord.problem_key();
  if (a == "dtype") return r.coord.dtype;
  if (a == "trial") return std::to_string(r.coord.trial);
  if (a == "concurrency") return to_string(r.concurrency);
  return "";
}

enum class TrialMode { Exact, Mean, Max, Min };

struct SliceOptions {
  std::string metric{"omega_U"};
  std::string row_axis{"solver"};
  std::string col_axis{"matrix_kind"};
  std::string table_axis;  // empty for 2D slice; e.g. "diagonalizer" for 3D
  TrialMode trial_mode{TrialMode::Mean};
  int exact_trial{0};
  bool broadcast_baselines{false};
  std::string format{"ascii"};  // ascii | md | csv | typst
  QueryExprPtr filter;
};

struct CellStats {
  double value{std::numeric_limits<double>::quiet_NaN()};
  int count{0};
  bool is_broadcast{false};
  bool internally_parallel{false};
  ConcurrencyModel concurrency{ConcurrencyModel::Serial};
  int rank{0};  // 1-based rank within its column (problem)
};

struct SliceTable2D {
  std::string title;
  std::vector<std::string> rows;
  std::vector<std::string> cols;
  std::vector<std::vector<CellStats>> cells;
};

inline double aggregate_trials(const std::vector<double>& vals, TrialMode mode) {
  std::vector<double> valid;
  for (double v : vals) {
    if (std::isfinite(v)) valid.push_back(v);
  }
  if (valid.empty()) return std::numeric_limits<double>::quiet_NaN();
  if (mode == TrialMode::Max) {
    return *std::max_element(valid.begin(), valid.end());
  }
  if (mode == TrialMode::Min) {
    return *std::min_element(valid.begin(), valid.end());
  }
  double sum = std::accumulate(valid.begin(), valid.end(), 0.0);
  return sum / static_cast<double>(valid.size());
}

inline std::vector<BenchmarkRecord> filter_records(
    const std::vector<BenchmarkRecord>& all, const QueryExprPtr& filter) {
  if (!filter) return all;
  std::vector<BenchmarkRecord> out;
  for (const auto& r : all) {
    if (filter->eval(QueryContext::from_record(r))) out.push_back(r);
  }
  return out;
}

inline std::vector<SliceTable2D> build_slice_tables(
    const std::vector<BenchmarkRecord>& records_in, const SliceOptions& opt) {
  auto filtered = filter_records(records_in, opt.filter);
  if (opt.trial_mode == TrialMode::Exact) {
    std::vector<BenchmarkRecord> exact;
    for (const auto& r : filtered) {
      if (r.coord.trial == opt.exact_trial) exact.push_back(r);
    }
    filtered = std::move(exact);
  }

  std::vector<std::string> table_keys;
  if (opt.table_axis.empty()) {
    table_keys.push_back("");
  } else {
    std::set<std::string> seen;
    for (const auto& r : filtered) {
      std::string tk = get_axis_value(r, opt.table_axis);
      if (opt.broadcast_baselines &&
          (opt.table_axis == "diag" || opt.table_axis == "diagonalizer") &&
          tk == "none") {
        continue;
      }
      if (seen.insert(tk).second) table_keys.push_back(tk);
    }
    if (table_keys.empty()) table_keys.push_back("none");
  }

  bool split_by_diag =
      (opt.table_axis == "diag" || opt.table_axis == "diagonalizer");

  std::vector<SliceTable2D> tables;
  const bool row_is_metric = (to_lower_copy(opt.row_axis) == "metric");
  const bool col_is_metric = (to_lower_copy(opt.col_axis) == "metric");
  const auto& metric_defs = all_metrics_info();

  for (const auto& tk : table_keys) {
    SliceTable2D tbl;
    if (row_is_metric || col_is_metric) {
      tbl.title = opt.table_axis.empty()
                      ? "All Metrics"
                      : (opt.table_axis + " = " + tk);
    } else {
      tbl.title = opt.table_axis.empty()
                      ? ("metric = " + opt.metric)
                      : (opt.table_axis + " = " + tk + " | metric = " + opt.metric);
    }

    std::vector<std::string> row_order;
    std::vector<std::string> col_order;
    std::set<std::string> row_set, col_set;

    struct Bucket {
      std::vector<double> vals;
      bool broadcast{false};
      ConcurrencyModel cm{ConcurrencyModel::Serial};
    };
    std::map<std::pair<std::string, std::string>, Bucket> buckets;

    if (row_is_metric) {
      for (const auto& m : metric_defs) {
        row_set.insert(m.key);
        row_order.push_back(m.key);
      }
    }
    if (col_is_metric) {
      for (const auto& m : metric_defs) {
        col_set.insert(m.key);
        col_order.push_back(m.key);
      }
    }

    for (const auto& r : filtered) {
      bool is_bc = false;
      if (!opt.table_axis.empty()) {
        std::string rtk = get_axis_value(r, opt.table_axis);
        if (rtk != tk) {
          if (opt.broadcast_baselines && split_by_diag && rtk == "none") {
            is_bc = true;
          } else {
            continue;
          }
        }
      }
      if (col_is_metric && !row_is_metric) {
        std::string rk = get_axis_value(r, opt.row_axis);
        if (is_bc) rk += " [baseline]";
        if (row_set.insert(rk).second) row_order.push_back(rk);
        for (const auto& m : metric_defs) {
          auto& b = buckets[{rk, m.key}];
          b.vals.push_back(r.metrics.get(m.key));
          b.broadcast = b.broadcast || is_bc;
          b.cm = r.concurrency;
        }
      } else if (row_is_metric && !col_is_metric) {
        std::string ck = get_axis_value(r, opt.col_axis);
        if (is_bc) ck += " [baseline]";
        if (col_set.insert(ck).second) col_order.push_back(ck);
        for (const auto& m : metric_defs) {
          auto& b = buckets[{m.key, ck}];
          b.vals.push_back(r.metrics.get(m.key));
          b.broadcast = b.broadcast || is_bc;
          b.cm = r.concurrency;
        }
      } else {
        std::string rk = get_axis_value(r, opt.row_axis);
        if (is_bc) rk += " [baseline]";
        std::string ck = get_axis_value(r, opt.col_axis);
        if (row_set.insert(rk).second) row_order.push_back(rk);
        if (col_set.insert(ck).second) col_order.push_back(ck);

        auto& b = buckets[{rk, ck}];
        b.vals.push_back(r.metrics.get(opt.metric));
        b.broadcast = b.broadcast || is_bc;
        b.cm = r.concurrency;
      }
    }

    tbl.rows = std::move(row_order);
    tbl.cols = std::move(col_order);
    tbl.cells.assign(tbl.rows.size(), std::vector<CellStats>(tbl.cols.size()));

    for (size_t ri = 0; ri < tbl.rows.size(); ++ri) {
      for (size_t ci = 0; ci < tbl.cols.size(); ++ci) {
        auto it = buckets.find({tbl.rows[ri], tbl.cols[ci]});
        if (it != buckets.end()) {
          CellStats cs;
          cs.value = aggregate_trials(it->second.vals, opt.trial_mode);
          cs.count = static_cast<int>(it->second.vals.size());
          cs.is_broadcast = it->second.broadcast;
          cs.concurrency = it->second.cm;
          cs.internally_parallel = is_internally_parallel(cs.concurrency);
          tbl.cells[ri][ci] = cs;
        }
      }
    }

    if (row_is_metric && !col_is_metric) {
      // Rank columns within each directional metric row
      for (size_t ri = 0; ri < tbl.rows.size(); ++ri) {
        auto mi = find_metric_info(tbl.rows[ri]);
        if (mi && !mi->directional) continue;
        std::vector<std::pair<double, size_t>> order;
        for (size_t ci = 0; ci < tbl.cols.size(); ++ci) {
          if (std::isfinite(tbl.cells[ri][ci].value)) {
            order.emplace_back(tbl.cells[ri][ci].value, ci);
          }
        }
        std::sort(order.begin(), order.end());
        for (size_t k = 0; k < order.size(); ++k) {
          tbl.cells[ri][order[k].second].rank = static_cast<int>(k + 1);
        }
      }
    } else {
      // Rank rows within each column (skipping non-directional metrics if col_is_metric)
      for (size_t ci = 0; ci < tbl.cols.size(); ++ci) {
        if (col_is_metric) {
          auto mi = find_metric_info(tbl.cols[ci]);
          if (mi && !mi->directional) continue;
        }
        std::vector<std::pair<double, size_t>> order;
        for (size_t ri = 0; ri < tbl.rows.size(); ++ri) {
          if (std::isfinite(tbl.cells[ri][ci].value)) {
            order.emplace_back(tbl.cells[ri][ci].value, ri);
          }
        }
        std::sort(order.begin(), order.end());
        for (size_t k = 0; k < order.size(); ++k) {
          tbl.cells[order[k].second][ci].rank = static_cast<int>(k + 1);
        }
      }
    }

    tables.push_back(std::move(tbl));
  }
  return tables;
}

inline std::string format_metric_value(const std::string& metric, double v) {
  if (!std::isfinite(v)) return "N/A";
  std::ostringstream oss;
  if (metric == "time_ms") {
    oss << std::fixed << std::setprecision(2) << v;
  } else {
    oss << std::scientific << std::setprecision(2) << v;
  }
  return oss.str();
}

inline std::string latex_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '\\': out += "\\textbackslash{}"; break;
      case '_': out += "\\_"; break;
      case '%': out += "\\%"; break;
      case '&': out += "\\&"; break;
      case '#': out += "\\#"; break;
      case '$': out += "\\$"; break;
      case '{': out += "\\{"; break;
      case '}': out += "\\}"; break;
      case '~': out += "\\textasciitilde{}"; break;
      case '^': out += "\\textasciicircum{}"; break;
      default: out.push_back(c); break;
    }
  }
  return out;
}

inline void render_tables(std::ostream& out,
                          const std::vector<SliceTable2D>& tables,
                          const SliceOptions& opt) {
  const bool row_is_metric = (to_lower_copy(opt.row_axis) == "metric");
  const bool col_is_metric = (to_lower_copy(opt.col_axis) == "metric");
  auto cell_metric = [&](const SliceTable2D& tbl, size_t r, size_t c) -> std::string {
    if (col_is_metric) return tbl.cols[c];
    if (row_is_metric) return tbl.rows[r];
    return opt.metric;
  };

  for (const auto& tbl : tables) {
    if (opt.format == "csv") {
      out << "# " << tbl.title << "\n";
      out << opt.row_axis;
      for (const auto& c : tbl.cols) out << "," << c;
      out << "\n";
      for (size_t r = 0; r < tbl.rows.size(); ++r) {
        out << tbl.rows[r];
        for (size_t c = 0; c < tbl.cols.size(); ++c) {
          out << "," << format_metric_value(cell_metric(tbl, r, c), tbl.cells[r][c].value);
        }
        out << "\n";
      }
      out << "\n";
    } else if (opt.format == "md") {
      out << "### " << tbl.title << "\n\n";
      out << "| " << opt.row_axis;
      for (const auto& c : tbl.cols) out << " | " << c;
      out << " |\n| :---";
      for (size_t c = 0; c < tbl.cols.size(); ++c) out << " | ---:";
      out << " |\n";
      for (size_t r = 0; r < tbl.rows.size(); ++r) {
        out << "| " << tbl.rows[r];
        for (size_t c = 0; c < tbl.cols.size(); ++c) {
          const auto& cs = tbl.cells[r][c];
          std::string s = format_metric_value(cell_metric(tbl, r, c), cs.value);
          if (cs.rank == 1) s = "**" + s + "**";
          out << " | " << s;
        }
        out << " |\n";
      }
      out << "\n";
    } else if (opt.format == "typst") {
      out << "// " << tbl.title << "\n";
      out << "#table(\n  columns: " << (tbl.cols.size() + 1) << ",\n  table.header([*"
          << opt.row_axis << "*]";
      for (const auto& c : tbl.cols) out << ", [*" << c << "*]";
      out << "),\n";
      for (size_t r = 0; r < tbl.rows.size(); ++r) {
        out << "  [" << tbl.rows[r] << "]";
        for (size_t c = 0; c < tbl.cols.size(); ++c) {
          const auto& cs = tbl.cells[r][c];
          std::string s = format_metric_value(cell_metric(tbl, r, c), cs.value);
          if (cs.rank == 1) out << ", [*" << s << "*]";
          else out << ", [" << s << "]";
        }
        out << ",\n";
      }
      out << ")\n\n";
    } else if (opt.format == "latex" || opt.format == "tex") {
      out << "% " << tbl.title << "\n";
      out << "\\begin{table}[htbp]\n\\centering\n";
      out << "\\caption{" << latex_escape(tbl.title) << "}\n";
      out << "\\begin{tabular}{l";
      for (size_t c = 0; c < tbl.cols.size(); ++c) out << " r";
      out << "}\n\\toprule\n";
      out << "\\textbf{" << latex_escape(opt.row_axis) << "}";
      for (const auto& c : tbl.cols) out << " & \\textbf{" << latex_escape(c) << "}";
      out << " \\\\\n\\midrule\n";
      for (size_t r = 0; r < tbl.rows.size(); ++r) {
        out << latex_escape(tbl.rows[r]);
        for (size_t c = 0; c < tbl.cols.size(); ++c) {
          const auto& cs = tbl.cells[r][c];
          std::string s = format_metric_value(cell_metric(tbl, r, c), cs.value);
          if (cs.rank == 1) out << " & \\textbf{" << latex_escape(s) << "}";
          else out << " & " << latex_escape(s);
        }
        out << " \\\\\n";
      }
      out << "\\bottomrule\n\\end{tabular}\n\\end{table}\n\n";
    } else {
      // Pretty ASCII
      std::vector<size_t> widths(tbl.cols.size() + 1, 10);
      widths[0] = std::max<size_t>(opt.row_axis.size(), 16);
      for (const auto& r : tbl.rows) widths[0] = std::max(widths[0], r.size());
      for (size_t c = 0; c < tbl.cols.size(); ++c) {
        widths[c + 1] = std::max<size_t>(tbl.cols[c].size(), 12);
      }
      size_t total_w = widths[0] + 2;
      for (size_t c = 0; c < tbl.cols.size(); ++c) total_w += widths[c + 1] + 2;

      out << "\n" << std::string(total_w, '=') << "\n";
      out << "  " << tbl.title << "\n";
      out << std::string(total_w, '=') << "\n";
      out << std::left << std::setw(static_cast<int>(widths[0]) + 2)
          << opt.row_axis;
      for (size_t c = 0; c < tbl.cols.size(); ++c) {
        out << std::right << std::setw(static_cast<int>(widths[c + 1]) + 2)
            << tbl.cols[c];
      }
      out << "\n" << std::string(total_w, '-') << "\n";
      for (size_t r = 0; r < tbl.rows.size(); ++r) {
        out << std::left << std::setw(static_cast<int>(widths[0]) + 2)
            << tbl.rows[r];
        for (size_t c = 0; c < tbl.cols.size(); ++c) {
          const auto& cs = tbl.cells[r][c];
          std::string s = format_metric_value(cell_metric(tbl, r, c), cs.value);
          if (cs.rank == 1) s = "*" + s;
          out << std::right << std::setw(static_cast<int>(widths[c + 1]) + 2)
              << s;
        }
        out << "\n";
      }
      out << std::string(total_w, '=') << "\n";
      out << "  * = best in group\n";
    }
  }
}

// ============================================================================
// Algorithm Ranking & Best-Cell Discovery (Deduplicated — Never Double-Counts)
// ============================================================================
struct SolverRankRow {
  std::string solver_key;
  ConcurrencyModel concurrency{ConcurrencyModel::Serial};
  int problems_evaluated{0};
  int wins{0};
  double mean_rank{0.0};
  double geomean_ratio{1.0};
  double best_val{std::numeric_limits<double>::infinity()};
  double worst_val{-std::numeric_limits<double>::infinity()};
};

struct RankingReport {
  std::string metric;
  bool has_best{false};
  BenchmarkRecord best_record;
  bool has_worst{false};
  BenchmarkRecord worst_record;
  std::vector<SolverRankRow> leaderboard;
};

inline RankingReport compute_ranking(
    const std::vector<BenchmarkRecord>& records_in, const std::string& metric,
    const QueryExprPtr& filter, const std::string& sort_by = "mean_rank") {
  auto filtered = filter_records(records_in, filter);
  RankingReport rep;
  rep.metric = metric;

  // 1. Absolute best & worst individual records
  double best_v = std::numeric_limits<double>::infinity();
  double worst_v = -std::numeric_limits<double>::infinity();

  // Group by problem = (matrix_kind, size) -> solver_key -> mean across trials
  struct SolverAccum {
    std::vector<double> vals;
    ConcurrencyModel cm{ConcurrencyModel::Serial};
  };
  std::map<std::string, std::map<std::string, SolverAccum>> by_problem;

  for (const auto& r : filtered) {
    if (r.status != RunStatus::Ok) continue;
    double v = r.metrics.get(metric);
    if (!std::isfinite(v)) continue;
    if (v < best_v) {
      best_v = v;
      rep.best_record = r;
      rep.has_best = true;
    }
    if (v > worst_v) {
      worst_v = v;
      rep.worst_record = r;
      rep.has_worst = true;
    }
    auto& acc = by_problem[r.coord.problem_key()][r.coord.solver_key()];
    acc.vals.push_back(v);
    acc.cm = r.concurrency;
  }

  // 2. Per-problem fair ranking
  struct Totals {
    ConcurrencyModel cm{ConcurrencyModel::Serial};
    int problems{0};
    int wins{0};
    double rank_sum{0.0};
    double log_ratio_sum{0.0};
    int ratio_count{0};
    double best{std::numeric_limits<double>::infinity()};
    double worst{-std::numeric_limits<double>::infinity()};
  };
  std::map<std::string, Totals> totals;

  for (const auto& [prob, solver_map] : by_problem) {
    std::vector<std::pair<double, std::string>> ranked;
    for (const auto& [skey, acc] : solver_map) {
      double m = aggregate_trials(acc.vals, TrialMode::Mean);
      if (std::isfinite(m)) {
        ranked.emplace_back(m, skey);
        auto& tot = totals[skey];
        tot.cm = acc.cm;
        if (m < tot.best) tot.best = m;
        if (m > tot.worst) tot.worst = m;
      }
    }
    if (ranked.empty()) continue;
    std::sort(ranked.begin(), ranked.end());
    double best_in_prob = std::max(ranked.front().first, 1e-30);
    for (size_t i = 0; i < ranked.size(); ++i) {
      auto& tot = totals[ranked[i].second];
      tot.problems += 1;
      tot.rank_sum += static_cast<double>(i + 1);
      if (i == 0) tot.wins += 1;
      if (ranked[i].first >= 0.0) {
        double ratio = std::max(ranked[i].first, 1e-30) / best_in_prob;
        tot.log_ratio_sum += std::log(std::max(1.0, ratio));
        tot.ratio_count += 1;
      }
    }
  }

  for (const auto& [skey, tot] : totals) {
    SolverRankRow row;
    row.solver_key = skey;
    row.concurrency = tot.cm;
    row.problems_evaluated = tot.problems;
    row.wins = tot.wins;
    row.mean_rank =
        tot.problems > 0 ? tot.rank_sum / static_cast<double>(tot.problems) : 0.0;
    row.geomean_ratio =
        tot.ratio_count > 0
            ? std::exp(tot.log_ratio_sum / static_cast<double>(tot.ratio_count))
            : 1.0;
    row.best_val = tot.best;
    row.worst_val = tot.worst;
    rep.leaderboard.push_back(std::move(row));
  }

  std::sort(rep.leaderboard.begin(), rep.leaderboard.end(),
            [&](const SolverRankRow& a, const SolverRankRow& b) {
              if (sort_by == "wins") {
                if (a.wins != b.wins) return a.wins > b.wins;
                return a.mean_rank < b.mean_rank;
              }
              if (sort_by == "geomean") return a.geomean_ratio < b.geomean_ratio;
              if (sort_by == "best") return a.best_val < b.best_val;
              if (sort_by == "worst") return a.worst_val < b.worst_val;
              if (std::abs(a.mean_rank - b.mean_rank) > 1e-9)
                return a.mean_rank < b.mean_rank;
              return a.geomean_ratio < b.geomean_ratio;
            });

  return rep;
}

inline void print_ranking_report(std::ostream& out, const RankingReport& rep) {
  out << "\n========================================================================================================\n";
  out << "  ALGORITHM RANKING & EXTREMUM REPORT | Metric: " << rep.metric << "\n";
  out << "========================================================================================================\n";
  if (rep.has_best) {
    out << "  [ABSOLUTE BEST]  "
        << format_metric_value(rep.metric, rep.best_record.metrics.get(rep.metric))
        << "  | " << rep.best_record.coord.solver_key() << " on "
        << rep.best_record.coord.problem_key()
        << " (trial=" << rep.best_record.coord.trial << ", "
        << to_string(rep.best_record.concurrency) << ")\n";
  }
  if (rep.has_worst) {
    out << "  [ABSOLUTE WORST] "
        << format_metric_value(rep.metric,
                               rep.worst_record.metrics.get(rep.metric))
        << "  | " << rep.worst_record.coord.solver_key() << " on "
        << rep.worst_record.coord.problem_key()
        << " (trial=" << rep.worst_record.coord.trial << ", "
        << to_string(rep.worst_record.concurrency) << ")\n";
  }
  out << "--------------------------------------------------------------------------------------------------------\n";
  out << std::left << std::setw(6) << "Rank" << std::setw(28) << "solver (algo+diag)"
      << std::setw(16) << "Concurrency" << std::right << std::setw(8)
      << "Wins" << std::setw(11) << "MeanRank" << std::setw(12) << "GeoRatio"
      << std::setw(13) << "Best" << std::setw(13) << "Worst" << "\n";
  out << "--------------------------------------------------------------------------------------------------------\n";
  for (size_t i = 0; i < rep.leaderboard.size(); ++i) {
    const auto& row = rep.leaderboard[i];
    out << std::left << std::setw(6) << ("#" + std::to_string(i + 1))
        << std::setw(28) << row.solver_key << std::setw(16)
        << to_string(row.concurrency) << std::right << std::setw(8)
        << (std::to_string(row.wins) + "/" +
            std::to_string(row.problems_evaluated))
        << std::fixed << std::setprecision(2) << std::setw(11) << row.mean_rank
        << std::setw(11) << row.geomean_ratio << "x" << std::setw(13)
        << format_metric_value(rep.metric, row.best_val) << std::setw(13)
        << format_metric_value(rep.metric, row.worst_val) << "\n";
  }
  out << "========================================================================================================\n";
}

}  // namespace SVD_Project::Benchmark
