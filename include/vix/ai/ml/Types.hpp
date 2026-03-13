/**
 * @file Types.hpp
 * @author Gaspard Kirira
 *
 * Copyright 2025, Gaspard Kirira. All rights reserved.
 * https://github.com/vixcpp/vix
 * Use of this source code is governed by a MIT license that can be found in the License file.
 *
 * Vix.cpp
 */
#ifndef VIX_AI_ML_TYPES_HPP
#define VIX_AI_ML_TYPES_HPP

#include <vector>
#include <cstddef>

namespace vix::ai::ml
{
  /**
   * @file Types.hpp
   * @brief Core type aliases and lightweight matrix helpers for Vix.AI.ML.
   *
   * This header defines fundamental data structures used throughout the
   * Vix.AI.ML module, including:
   * - Basic numeric containers (`Vec`, `Mat`)
   * - Index arrays (`Idxs`)
   * - Convenience functions for accessing matrix dimensions
   *
   * These abstractions are intentionally simple and header-only to maintain
   * compatibility with both standard STL-based implementations and potential
   * future backends (e.g., Eigen, Blaze, or custom GPU arrays).
   *
   * ### Example:
   * @code
   * Mat X = {{1.0, 2.0}, {3.0, 4.0}};
   * Vec y = {5.0, 6.0};
   * std::cout << "rows=" << nrows(X) << " cols=" << ncols(X);
   * @endcode
   */

  /**
   * @brief Vector of doubles representing a single data row or feature vector.
   */
  using Vec = std::vector<double>;

  /**
   * @brief Matrix of doubles represented as a vector of rows.
   *
   * Conceptually, `Mat[row][col]` gives the value of feature `col`
   * in sample `row`. Each inner vector represents one data sample.
   */
  using Mat = std::vector<Vec>;

  /**
   * @brief Index vector (commonly used for cluster assignments or sample indices).
   */
  using Idxs = std::vector<std::size_t>;

  /**
   * @brief Get the number of rows (samples) in a matrix.
   * @param m Matrix.
   * @return Number of rows.
   *
   * Equivalent to `m.size()`.
   */
  inline std::size_t nrows(const Mat &m) { return m.size(); }

  /**
   * @brief Get the number of columns (features) in a matrix.
   * @param m Matrix.
   * @return Number of columns, or 0 if the matrix is empty.
   *
   * Equivalent to `m.empty() ? 0 : m[0].size()`.
   */
  inline std::size_t ncols(const Mat &m) { return m.empty() ? 0 : m[0].size(); }

} // namespace vix::ai::ml

#endif
