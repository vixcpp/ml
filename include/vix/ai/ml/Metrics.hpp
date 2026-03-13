/**
 * @file Metrics.hpp
 * @author Gaspard Kirira
 *
 * Copyright 2025, Gaspard Kirira. All rights reserved.
 * https://github.com/vixcpp/vix
 * Use of this source code is governed by a MIT license that can be found in the License file.
 *
 * Vix.cpp
 */
#ifndef VIX_AI_ML_METRICS_HPP
#define VIX_AI_ML_METRICS_HPP

#include <vix/ai/ml/Types.hpp>
#include <cmath>
#include <cstddef>
#include <limits>

namespace vix::ai::ml
{
  /**
   * @file Metrics.hpp
   * @brief Evaluation metrics for supervised learning models.
   *
   * This header defines lightweight utility functions to compute common
   * performance metrics such as Mean Squared Error (MSE) for regression
   * and binary accuracy for classification.
   *
   * These metrics are intentionally implemented as simple, inline, header-only
   * functions for portability and performance.
   *
   * ### Example:
   * @code
   * double err = mse(y_true, y_pred);
   * double acc = accuracy01(y_true, y_pred);
   * @endcode
   */

  /**
   * @brief Compute the Mean Squared Error (MSE) between predictions and targets.
   *
   * The MSE is defined as:
   * \f[
   *   \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
   * \f]
   *
   * @param y     Ground truth vector (true values).
   * @param yhat  Predicted values vector.
   * @return Mean squared error (MSE). Returns +∞ if vectors are empty or size mismatch.
   *
   * ### Example:
   * @code
   * double loss = mse(y, y_pred);
   * @endcode
   */
  inline double mse(const Vec &y, const Vec &yhat)
  {
    const std::size_t n = y.size();
    if (n == 0 || yhat.size() != n)
      return std::numeric_limits<double>::infinity();

    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
      const double e = yhat[i] - y[i];
      s += e * e;
    }
    return s / static_cast<double>(n);
  }

  /**
   * @brief Compute binary classification accuracy for y ∈ {0, 1} and ŷ ∈ [0, 1].
   *
   * Each predicted probability is thresholded at 0.5:
   * - If ŷ ≥ 0.5 → predicted = 1
   * - Otherwise → predicted = 0
   *
   * The metric returns the fraction of correctly predicted labels.
   *
   * @param y     Ground truth labels (0 or 1).
   * @param yhat  Predicted probabilities or scores.
   * @return Accuracy between 0.0 and 1.0 (0 if invalid input).
   *
   * ### Example:
   * @code
   * double acc = accuracy01(y_true, y_pred);
   * @endcode
   */
  inline double accuracy01(const Vec &y, const Vec &yhat)
  {
    const std::size_t n = y.size();
    if (n == 0 || yhat.size() != n)
      return 0.0;

    std::size_t ok = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
      const int p = yhat[i] >= 0.5 ? 1 : 0;
      ok += (p == static_cast<int>(y[i]));
    }
    return static_cast<double>(ok) / static_cast<double>(n);
  }

} // namespace vix::ai::ml

#endif
