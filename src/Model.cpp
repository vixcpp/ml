/**
 * @file Model.cpp
 * @author Gaspard Kirira
 *
 * Copyright 2025, Gaspard Kirira. All rights reserved.
 * https://github.com/vixcpp/vix
 * Use of this source code is governed by a MIT license that can be found in the License file.
 *
 * Vix.cpp
 */
#include <vix/ai/ml/Model.hpp>

#include <stdexcept>
#include <string>

namespace vix::ai::ml
{

  // =============================================================================
  // Protected static helpers
  // =============================================================================

  /**
   * @brief Validate that @p X is non-empty and that every row has the same
   *        number of columns as the first row.
   *
   * Complexity: O(n_rows) — one pass to check widths.
   *
   * @throws std::invalid_argument
   *   - "… X must not be empty"           — when nrows(X) == 0
   *   - "… X must not contain empty rows" — when ncols(X) == 0
   *   - "… row i has width w …"           — on a width mismatch
   */
  void Model::validate_matrix(const Mat &X, const std::string &context)
  {
    if (X.empty())
    {
      throw std::invalid_argument(
          context + ": X must not be empty.");
    }

    const std::size_t expected_cols = X[0].size();

    if (expected_cols == 0)
    {
      throw std::invalid_argument(
          context + ": X must not contain empty rows (ncols == 0).");
    }

    for (std::size_t i = 1; i < X.size(); ++i)
    {
      if (X[i].size() != expected_cols)
      {
        throw std::invalid_argument(
            context + ": row " + std::to_string(i) +
            " has " + std::to_string(X[i].size()) +
            " column(s); expected " + std::to_string(expected_cols) + ".");
      }
    }
  }

  /**
   * @brief Validate that @p X is a valid matrix and that @p y has exactly as
   *        many elements as @p X has rows.
   *
   * Delegates the matrix check to `validate_matrix`, then performs the
   * label-alignment check.
   *
   * @throws std::invalid_argument (see validate_matrix), or:
   *   - "… y.size() (m) != nrows(X) (n)" — on a size mismatch
   */
  void Model::validate_supervised(const Mat &X,
                                  const Vec &y,
                                  const std::string &context)
  {
    // Reuse matrix checks first (emptiness, uniform widths).
    validate_matrix(X, context);

    if (y.size() != X.size())
    {
      throw std::invalid_argument(
          context + ": y.size() (" + std::to_string(y.size()) +
          ") != nrows(X) (" + std::to_string(X.size()) + ").");
    }
  }

  // =============================================================================
  // Training — default (validated no-ops)
  // =============================================================================

  /**
   * Default supervised fit: validates inputs and returns without modifying state.
   * Derived supervised models must override this.
   */
  void Model::fit(const Mat &X, const Vec &y)
  {
    validate_supervised(X, y, "Model::fit(X, y)");
    // No-op: base class holds no parameters.
  }

  /**
   * Default unsupervised fit: validates the matrix and returns without
   * modifying state.  Derived unsupervised models must override this.
   */
  void Model::fit(const Mat &X)
  {
    validate_matrix(X, "Model::fit(X)");
    // No-op: base class holds no parameters.
  }

  // =============================================================================
  // Inference
  // =============================================================================

  /**
   * Default single-sample prediction.
   * Returns 0.0 unconditionally — override in derived models.
   */
  double Model::predict_one(const Vec &x) const
  {
    (void)x;
    return 0.0;
  }

  /**
   * Batch prediction via repeated calls to predict_one().
   *
   * Iterates over every row of @p X and collects results into a contiguous
   * output vector.  The output is pre-allocated with `reserve` to avoid
   * reallocation.
   *
   * Derived models may override for a vectorised/SIMD implementation while
   * keeping this as a correct, portable fallback.
   *
   * @note No matrix validation is performed here intentionally; callers are
   *       expected to have a fitted model and to supply well-formed input.
   *       Derived overrides should add validation if they require it.
   */
  Vec Model::predict(const Mat &X) const
  {
    Vec out;
    out.reserve(nrows(X));
    for (const auto &row : X)
      out.push_back(predict_one(row));
    return out;
  }

  // =============================================================================
  // Serialisation — default (no-ops)
  // =============================================================================

  /**
   * Default save: writes nothing.
   * Override in derived models to persist learned parameters.
   */
  void Model::save(std::ostream &os) const
  {
    (void)os;
  }

  /**
   * Default load: reads nothing.
   * Override in derived models to restore parameters written by save().
   */
  void Model::load(std::istream &is)
  {
    (void)is;
  }

} // namespace vix::ai::ml
