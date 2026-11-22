#pragma once

#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "types.h"

namespace SVD_Project::Benchmark {

inline std::string current_iso8601_utc() {
  auto now = std::chrono::system_clock::now();
  std::time_t tt = std::chrono::system_clock::to_time_t(now);
  std::tm utc_tm{};
#if defined(_WIN32)
  gmtime_s(&utc_tm, &tt);
#else
  gmtime_r(&tt, &utc_tm);
#endif
  std::ostringstream oss;
  oss << std::put_time(&utc_tm, "%Y-%m-%dT%H:%M:%SZ");
  return oss.str();
}

inline std::string compute_env_fingerprint(int runner_jobs,
                                           bool blas_pinned) {
  std::ostringstream oss;
#if defined(__clang__)
  oss << "clang-" << __clang_major__ << "." << __clang_minor__ << "."
      << __clang_patchlevel__;
#elif defined(__GNUC__)
  oss << "gcc-" << __GNUC__ << "." << __GNUC_MINOR__ << "."
      << __GNUC_PATCHLEVEL__;
#else
  oss << "unknown-cc";
#endif
  oss << "|eigen-" << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "."
      << EIGEN_MINOR_VERSION;
  oss << "|jobs-" << runner_jobs << "|pin-" << (blas_pinned ? "1" : "0");
  return oss.str();
}

inline std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          std::ostringstream hex;
          hex << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(static_cast<unsigned char>(c));
          out += hex.str();
        } else {
          out.push_back(c);
        }
    }
  }
  return out;
}

inline std::string json_double(double v) {
  if (!std::isfinite(v)) return "null";
  std::ostringstream oss;
  oss << std::setprecision(17) << v;
  return oss.str();
}

struct RunHeader {
  int schema_version{kSchemaVersion};
  int metrics_schema_version{kMetricsSchemaVersion};
  std::string timestamp;
  uint64_t base_seed{42};
  int jobs{1};
  int hardware_concurrency{1};
  bool blas_threads_pinned{true};
  std::string env_fingerprint;
  std::string query;
};

inline std::string serialize_header_jsonl(const RunHeader& h) {
  std::ostringstream oss;
  oss << "{"
      << "\"record_type\":\"run_header\","
      << "\"schema_version\":" << h.schema_version << ","
      << "\"metrics_schema_version\":" << h.metrics_schema_version << ","
      << "\"timestamp\":\"" << json_escape(h.timestamp) << "\","
      << "\"seed\":" << h.base_seed << ","
      << "\"jobs\":" << h.jobs << ","
      << "\"hardware_concurrency\":" << h.hardware_concurrency << ","
      << "\"blas_threads_pinned\":" << (h.blas_threads_pinned ? "true" : "false")
      << ","
      << "\"env_fingerprint\":\"" << json_escape(h.env_fingerprint) << "\","
      << "\"query\":\"" << json_escape(h.query) << "\""
      << "}";
  return oss.str();
}

inline std::string serialize_record_jsonl(const BenchmarkRecord& r) {
  std::ostringstream oss;
  oss << "{"
      << "\"record_type\":\"result\","
      << "\"algorithm\":\"" << json_escape(r.coord.algorithm) << "\","
      << "\"diagonalizer\":\"" << json_escape(r.coord.diagonalizer) << "\","
      << "\"matrix_kind\":\"" << json_escape(r.coord.matrix_kind) << "\","
      << "\"matrix_size\":\"" << json_escape(r.coord.size.label()) << "\","
      << "\"rows\":" << r.coord.size.rows << ","
      << "\"cols\":" << r.coord.size.cols << ","
      << "\"dtype\":\"" << json_escape(r.coord.dtype) << "\","
      << "\"trial\":" << r.coord.trial << ","
      << "\"seed\":" << r.seed << ","
      << "\"matrix_hash\":\"" << json_escape(r.matrix_hash) << "\","
      << "\"concurrency_model\":\"" << to_string(r.concurrency) << "\","
      << "\"internally_parallel\":"
      << (r.internally_parallel ? "true" : "false") << ","
      << "\"runner_jobs\":" << r.runner_jobs << ","
      << "\"blas_threads_pinned\":" << (r.blas_threads_pinned ? "true" : "false")
      << ","
      << "\"status\":\"" << to_string(r.status) << "\","
      << "\"error_msg\":\"" << json_escape(r.error_msg) << "\","
      << "\"has_ground_truth\":" << (r.has_ground_truth ? "true" : "false")
      << ","
      << "\"metrics\":{"
      << "\"time_ms\":" << json_double(r.metrics.time_ms) << ","
      << "\"omega_U\":" << json_double(r.metrics.omega_U) << ","
      << "\"omega_V\":" << json_double(r.metrics.omega_V) << ","
      << "\"r\":" << json_double(r.metrics.r) << ","
      << "\"eta\":" << json_double(r.metrics.eta) << ","
      << "\"rho\":" << json_double(r.metrics.rho) << ","
      << "\"kappa\":" << json_double(r.metrics.kappa) << ","
      << "\"delta\":" << json_double(r.metrics.delta) << ","
      << "\"Delta\":" << json_double(r.metrics.Delta) << ","
      << "\"log_det\":" << json_double(r.metrics.log_det) << ","
      << "\"sigma_min\":" << json_double(r.metrics.sigma_min) << ","
      << "\"sigma_max\":" << json_double(r.metrics.sigma_max)
      << "}}";
  return oss.str();
}

// Minimal, fast JSON object parser tailored to our flat/1-level-nested JSONL records
namespace json_detail {

inline void skip_ws(const std::string& s, size_t& i) {
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n'))
    ++i;
}

inline std::string parse_string(const std::string& s, size_t& i) {
  std::string out;
  if (i >= s.size() || s[i] != '"') return out;
  ++i;
  while (i < s.size()) {
    char c = s[i++];
    if (c == '"') break;
    if (c == '\\' && i < s.size()) {
      char e = s[i++];
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        default: out.push_back(e); break;
      }
    } else {
      out.push_back(c);
    }
  }
  return out;
}

inline std::string parse_raw_token(const std::string& s, size_t& i) {
  skip_ws(s, i);
  if (i < s.size() && s[i] == '"') return parse_string(s, i);
  if (i < s.size() && s[i] == '{') {
    size_t start = i;
    int depth = 0;
    bool in_str = false;
    while (i < s.size()) {
      char c = s[i++];
      if (c == '"' && (i < 2 || s[i - 2] != '\\')) in_str = !in_str;
      if (!in_str) {
        if (c == '{') ++depth;
        else if (c == '}') {
          --depth;
          if (depth == 0) break;
        }
      }
    }
    return s.substr(start, i - start);
  }
  size_t start = i;
  while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ' ' &&
         s[i] != '\t' && s[i] != '\r' && s[i] != '\n') {
    ++i;
  }
  return s.substr(start, i - start);
}

inline std::map<std::string, std::string> parse_flat_object(const std::string& s) {
  std::map<std::string, std::string> kv;
  size_t i = 0;
  skip_ws(s, i);
  if (i >= s.size() || s[i] != '{') return kv;
  ++i;
  while (i < s.size()) {
    skip_ws(s, i);
    if (i >= s.size() || s[i] == '}') break;
    if (s[i] != '"') break;
    std::string key = parse_string(s, i);
    skip_ws(s, i);
    if (i < s.size() && s[i] == ':') ++i;
    std::string val = parse_raw_token(s, i);
    kv[key] = val;
    skip_ws(s, i);
    if (i < s.size() && s[i] == ',') ++i;
  }
  return kv;
}

inline double parse_double_or_nan(const std::string& s) {
  if (s.empty() || s == "null" || s == "NaN" || s == "nan") {
    return std::numeric_limits<double>::quiet_NaN();
  }
  try {
    return std::stod(s);
  } catch (...) {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

}  // namespace json_detail

inline std::optional<BenchmarkRecord> parse_record_jsonl(const std::string& line) {
  auto kv = json_detail::parse_flat_object(line);
  auto rt = kv.find("record_type");
  if (rt == kv.end() || rt->second != "result") return std::nullopt;

  BenchmarkRecord r;
  r.coord.algorithm = kv["algorithm"];
  r.coord.diagonalizer = kv["diagonalizer"];
  r.coord.matrix_kind = kv["matrix_kind"];
  r.coord.size.rows = kv.count("rows") ? std::stoi(kv["rows"]) : 0;
  r.coord.size.cols = kv.count("cols") ? std::stoi(kv["cols"]) : 0;
  if (r.coord.size.rows == 0 && kv.count("matrix_size")) {
    if (auto sz = parse_matrix_size(kv["matrix_size"])) r.coord.size = *sz;
  }
  r.coord.dtype = kv.count("dtype") ? kv["dtype"] : "double";
  r.coord.trial = kv.count("trial") ? std::stoi(kv["trial"]) : 0;
  if (kv.count("seed") && !kv["seed"].empty()) {
    try { r.seed = std::stoull(kv["seed"]); } catch (...) {}
  }
  r.matrix_hash = kv["matrix_hash"];
  r.concurrency = concurrency_from_string(kv["concurrency_model"]);
  r.internally_parallel = (kv["internally_parallel"] == "true");
  r.runner_jobs = kv.count("runner_jobs") ? std::stoi(kv["runner_jobs"]) : 1;
  r.blas_threads_pinned = (kv["blas_threads_pinned"] == "true");
  r.status = status_from_string(kv["status"]);
  r.error_msg = kv["error_msg"];
  r.has_ground_truth = (kv["has_ground_truth"] == "true");

  if (kv.count("metrics")) {
    auto m = json_detail::parse_flat_object(kv["metrics"]);
    for (const auto& [k, v] : m) {
      r.metrics.set(k, json_detail::parse_double_or_nan(v));
    }
  }
  return r;
}

inline std::vector<BenchmarkRecord> load_jsonl_files(
    const std::vector<std::string>& paths) {
  // Merge by coord_key (latest record wins)
  std::map<std::string, size_t> seen;
  std::vector<BenchmarkRecord> records;
  for (const auto& path : paths) {
    std::ifstream in(path);
    if (!in) {
      throw std::runtime_error("Cannot open JSONL file: " + path);
    }
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      if (auto rec = parse_record_jsonl(line)) {
        std::string key = rec->coord.coord_key();
        auto it = seen.find(key);
        if (it != seen.end()) {
          records[it->second] = std::move(*rec);
        } else {
          seen[key] = records.size();
          records.push_back(std::move(*rec));
        }
      }
    }
  }
  return records;
}

}  // namespace SVD_Project::Benchmark
