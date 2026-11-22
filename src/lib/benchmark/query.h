#pragma once

#include <algorithm>
#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "types.h"

namespace SVD_Project::Benchmark {

struct QueryContext {
  std::string algorithm;
  std::string diagonalizer;
  std::string matrix_kind;
  std::string matrix_size;
  int rows{0};
  int cols{0};
  std::string dtype{"double"};
  int trial{0};
  std::string concurrency;
  bool internally_parallel{false};
  std::string status{"ok"};

  static QueryContext from_coord(const TestCoord& c, ConcurrencyModel cm,
                                 RunStatus st = RunStatus::Ok) {
    QueryContext ctx;
    ctx.algorithm = c.algorithm;
    ctx.diagonalizer = c.diagonalizer;
    ctx.matrix_kind = c.matrix_kind;
    ctx.matrix_size = c.size.label();
    ctx.rows = c.size.rows;
    ctx.cols = c.size.cols;
    ctx.dtype = c.dtype;
    ctx.trial = c.trial;
    ctx.concurrency = to_string(cm);
    ctx.internally_parallel = is_internally_parallel(cm);
    ctx.status = to_string(st);
    return ctx;
  }

  static QueryContext from_record(const BenchmarkRecord& r) {
    return from_coord(r.coord, r.concurrency, r.status);
  }
};

inline bool glob_match(const std::string& pattern, const std::string& text) {
  size_t p = 0, t = 0, star = std::string::npos, match = 0;
  while (t < text.size()) {
    if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == text[t])) {
      ++p;
      ++t;
    } else if (p < pattern.size() && pattern[p] == '*') {
      star = p++;
      match = t;
    } else if (star != std::string::npos) {
      p = star + 1;
      t = ++match;
    } else {
      return false;
    }
  }
  while (p < pattern.size() && pattern[p] == '*') ++p;
  return p == pattern.size();
}

inline std::string to_lower_copy(std::string s) {
  for (char& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return s;
}

class QueryExpr {
 public:
  virtual ~QueryExpr() = default;
  virtual bool eval(const QueryContext& ctx) const = 0;
};

using QueryExprPtr = std::shared_ptr<const QueryExpr>;

namespace query_detail {

enum class TokKind {
  End,
  Ident,
  LParen,
  RParen,
  Comma,
  Eq,
  Neq,
  Lt,
  Le,
  Gt,
  Ge,
  And,
  Or,
  Not,
  In
};

struct Token {
  TokKind kind{TokKind::End};
  std::string text;
};

inline std::vector<Token> tokenize(const std::string& input) {
  std::vector<Token> toks;
  size_t i = 0;
  const size_t n = input.size();
  while (i < n) {
    if (std::isspace(static_cast<unsigned char>(input[i]))) {
      ++i;
      continue;
    }
    char c = input[i];
    if (c == '(') {
      toks.push_back({TokKind::LParen, "("});
      ++i;
    } else if (c == ')') {
      toks.push_back({TokKind::RParen, ")"});
      ++i;
    } else if (c == ',') {
      toks.push_back({TokKind::Comma, ","});
      ++i;
    } else if (c == '&') {
      if (i + 1 < n && input[i + 1] == '&') ++i;
      toks.push_back({TokKind::And, "&"});
      ++i;
    } else if (c == '|') {
      if (i + 1 < n && input[i + 1] == '|') ++i;
      toks.push_back({TokKind::Or, "|"});
      ++i;
    } else if (c == '!' && i + 1 < n && input[i + 1] == '=') {
      toks.push_back({TokKind::Neq, "!="});
      i += 2;
    } else if (c == '!') {
      toks.push_back({TokKind::Not, "!"});
      ++i;
    } else if (c == '=') {
      if (i + 1 < n && input[i + 1] == '=') ++i;
      toks.push_back({TokKind::Eq, "="});
      ++i;
    } else if (c == '<') {
      if (i + 1 < n && input[i + 1] == '=') {
        toks.push_back({TokKind::Le, "<="});
        i += 2;
      } else {
        toks.push_back({TokKind::Lt, "<"});
        ++i;
      }
    } else if (c == '>') {
      if (i + 1 < n && input[i + 1] == '=') {
        toks.push_back({TokKind::Ge, ">="});
        i += 2;
      } else {
        toks.push_back({TokKind::Gt, ">"});
        ++i;
      }
    } else if (c == '\'' || c == '"') {
      char quote = c;
      ++i;
      std::string val;
      while (i < n && input[i] != quote) val.push_back(input[i++]);
      if (i < n) ++i;
      toks.push_back({TokKind::Ident, val});
    } else {
      std::string val;
      while (i < n && !std::isspace(static_cast<unsigned char>(input[i])) &&
             std::string("(),=!<>|&").find(input[i]) == std::string::npos) {
        val.push_back(input[i++]);
      }
      std::string low = to_lower_copy(val);
      if (low == "and") toks.push_back({TokKind::And, val});
      else if (low == "or") toks.push_back({TokKind::Or, val});
      else if (low == "not") toks.push_back({TokKind::Not, val});
      else if (low == "in") toks.push_back({TokKind::In, val});
      else toks.push_back({TokKind::Ident, val});
    }
  }
  toks.push_back({TokKind::End, ""});
  return toks;
}

struct TrueExpr : QueryExpr {
  bool eval(const QueryContext&) const override { return true; }
};

struct AndExpr : QueryExpr {
  QueryExprPtr lhs, rhs;
  AndExpr(QueryExprPtr a, QueryExprPtr b)
      : lhs(std::move(a)), rhs(std::move(b)) {}
  bool eval(const QueryContext& ctx) const override {
    return lhs->eval(ctx) && rhs->eval(ctx);
  }
};

struct OrExpr : QueryExpr {
  QueryExprPtr lhs, rhs;
  OrExpr(QueryExprPtr a, QueryExprPtr b)
      : lhs(std::move(a)), rhs(std::move(b)) {}
  bool eval(const QueryContext& ctx) const override {
    return lhs->eval(ctx) || rhs->eval(ctx);
  }
};

struct NotExpr : QueryExpr {
  QueryExprPtr sub;
  explicit NotExpr(QueryExprPtr s) : sub(std::move(s)) {}
  bool eval(const QueryContext& ctx) const override { return !sub->eval(ctx); }
};

inline std::string canonical_field(const std::string& raw) {
  std::string f = to_lower_copy(raw);
  if (f == "algo" || f == "algorithm") return "algorithm";
  if (f == "diag" || f == "diagonalizer" || f == "diagonalization")
    return "diagonalizer";
  if (f == "matrix" || f == "matrix_kind" || f == "kind" || f == "family")
    return "matrix_kind";
  if (f == "size" || f == "matrix_size") return "matrix_size";
  if (f == "rows" || f == "m") return "rows";
  if (f == "cols" || f == "n") return "cols";
  if (f == "dtype" || f == "type") return "dtype";
  if (f == "trial") return "trial";
  if (f == "concurrency" || f == "concurrency_model") return "concurrency";
  if (f == "parallel" || f == "internally_parallel") return "internally_parallel";
  if (f == "status") return "status";
  throw std::invalid_argument("Unknown query field: '" + raw + "'");
}

inline std::string extract_str(const QueryContext& ctx, const std::string& f) {
  if (f == "algorithm") return ctx.algorithm;
  if (f == "diagonalizer") return ctx.diagonalizer;
  if (f == "matrix_kind") return ctx.matrix_kind;
  if (f == "matrix_size") return ctx.matrix_size;
  if (f == "rows") return std::to_string(ctx.rows);
  if (f == "cols") return std::to_string(ctx.cols);
  if (f == "dtype") return ctx.dtype;
  if (f == "trial") return std::to_string(ctx.trial);
  if (f == "concurrency") return ctx.concurrency;
  if (f == "internally_parallel") return ctx.internally_parallel ? "true" : "false";
  if (f == "status") return ctx.status;
  return "";
}

inline int extract_int(const QueryContext& ctx, const std::string& f) {
  if (f == "rows") return ctx.rows;
  if (f == "cols") return ctx.cols;
  if (f == "trial") return ctx.trial;
  throw std::invalid_argument("Numeric comparison not supported on field: " + f);
}

struct CompareExpr : QueryExpr {
  std::string field;
  TokKind op;
  std::vector<std::string> values;
  bool negate{false};

  bool eval(const QueryContext& ctx) const override {
    bool res = false;
    if (op == TokKind::Lt || op == TokKind::Le || op == TokKind::Gt ||
        op == TokKind::Ge) {
      int lhs = extract_int(ctx, field);
      int rhs = std::stoi(values[0]);
      if (op == TokKind::Lt) res = lhs < rhs;
      else if (op == TokKind::Le) res = lhs <= rhs;
      else if (op == TokKind::Gt) res = lhs > rhs;
      else if (op == TokKind::Ge) res = lhs >= rhs;
    } else if (op == TokKind::In) {
      std::string actual = to_lower_copy(extract_str(ctx, field));
      for (const auto& v : values) {
        if (glob_match(to_lower_copy(v), actual)) {
          res = true;
          break;
        }
      }
    } else if (op == TokKind::Eq || op == TokKind::Neq) {
      std::string actual = to_lower_copy(extract_str(ctx, field));
      std::string target = to_lower_copy(values[0]);
      if (field == "matrix_size" && target.find('x') == std::string::npos &&
          target.find('*') == std::string::npos) {
        target = target + "x" + target;
      }
      bool matched = glob_match(target, actual);
      res = (op == TokKind::Eq) ? matched : !matched;
    }
    return negate ? !res : res;
  }
};

class Parser {
 public:
  explicit Parser(std::vector<Token> toks) : toks_(std::move(toks)) {}

  QueryExprPtr parse() {
    if (peek().kind == TokKind::End) return std::make_shared<TrueExpr>();
    auto expr = parse_or();
    if (peek().kind != TokKind::End) {
      throw std::invalid_argument("Unexpected token in query: '" + peek().text + "'");
    }
    return expr;
  }

 private:
  std::vector<Token> toks_;
  size_t pos_{0};

  const Token& peek() const { return toks_[pos_]; }
  const Token& advance() { return toks_[pos_++]; }

  QueryExprPtr parse_or() {
    auto lhs = parse_and();
    while (peek().kind == TokKind::Or) {
      advance();
      lhs = std::make_shared<OrExpr>(std::move(lhs), parse_and());
    }
    return lhs;
  }

  QueryExprPtr parse_and() {
    auto lhs = parse_not();
    while (peek().kind == TokKind::And) {
      advance();
      lhs = std::make_shared<AndExpr>(std::move(lhs), parse_not());
    }
    return lhs;
  }

  QueryExprPtr parse_not() {
    if (peek().kind == TokKind::Not) {
      advance();
      return std::make_shared<NotExpr>(parse_not());
    }
    return parse_primary();
  }

  QueryExprPtr parse_primary() {
    if (peek().kind == TokKind::LParen) {
      advance();
      auto expr = parse_or();
      if (peek().kind != TokKind::RParen) {
        throw std::invalid_argument("Missing closing ')' in query");
      }
      advance();
      return expr;
    }
    if (peek().kind != TokKind::Ident) {
      throw std::invalid_argument("Expected field name in query, got '" +
                                  peek().text + "'");
    }
    std::string field = canonical_field(advance().text);
    auto cmp = std::make_shared<CompareExpr>();
    cmp->field = field;

    if (peek().kind == TokKind::Not) {
      advance();
      if (peek().kind != TokKind::In) {
        throw std::invalid_argument("Expected 'in' after 'not' in query");
      }
      advance();
      cmp->op = TokKind::In;
      cmp->negate = true;
      cmp->values = parse_list();
      return cmp;
    }
    if (peek().kind == TokKind::In) {
      advance();
      cmp->op = TokKind::In;
      cmp->values = parse_list();
      return cmp;
    }
    TokKind op = peek().kind;
    if (op != TokKind::Eq && op != TokKind::Neq && op != TokKind::Lt &&
        op != TokKind::Le && op != TokKind::Gt && op != TokKind::Ge) {
      throw std::invalid_argument("Expected comparison operator after '" +
                                  field + "'");
    }
    advance();
    cmp->op = op;
    if (peek().kind != TokKind::Ident) {
      throw std::invalid_argument("Expected value after operator in query");
    }
    cmp->values.push_back(advance().text);
    return cmp;
  }

  std::vector<std::string> parse_list() {
    if (peek().kind != TokKind::LParen) {
      throw std::invalid_argument("Expected '(' after 'in'");
    }
    advance();
    std::vector<std::string> vals;
    while (peek().kind != TokKind::RParen && peek().kind != TokKind::End) {
      if (peek().kind != TokKind::Ident) {
        throw std::invalid_argument("Expected value inside 'in (...)'");
      }
      vals.push_back(advance().text);
      if (peek().kind == TokKind::Comma) advance();
    }
    if (peek().kind != TokKind::RParen) {
      throw std::invalid_argument("Missing ')' in 'in (...)'");
    }
    advance();
    return vals;
  }
};

}  // namespace query_detail

inline QueryExprPtr parse_query(const std::string& expr) {
  if (expr.empty()) return std::make_shared<query_detail::TrueExpr>();
  auto toks = query_detail::tokenize(expr);
  query_detail::Parser p(std::move(toks));
  return p.parse();
}

inline QueryExprPtr combine_and(QueryExprPtr a, QueryExprPtr b) {
  if (!a) return b;
  if (!b) return a;
  return std::make_shared<query_detail::AndExpr>(std::move(a), std::move(b));
}

}  // namespace SVD_Project::Benchmark
