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
#include <numeric>

namespace vix::ai::ml
{

  /**
   * @brief Evaluation metrics for regression and binary classification models.
   *
   * All functions are `inline`, header-only, and make no dynamic allocations.
   * Invalid inputs (empty vectors, size mismatches) are handled consistently:
   * - Regression metrics return `+∞` or `NaN` as documented per function.
   * - Classification metrics return `0.0` on invalid input.
   *
   * ### Quick reference
   * | Function         | Task           | Returns on error |
   * |------------------|----------------|-----------------|
   * | `mse`            | Regression     | `+∞`            |
   * | `mae`            | Regression     | `+∞`            |
   * | `rmse`           | Regression     | `+∞`            |
   * | `r2_score`       | Regression     | `NaN`           |
   * | `accuracy01`     | Classification | `0.0`           |
   * | `precision01`    | Classification | `0.0`           |
   * | `recall01`       | Classification | `0.0`           |
   * | `f1_score01`     | Classification | `0.0`           |
   *
   * ### Example
   * @code
   * double loss = vix::ai::ml::mse(y_true, y_pred);
   * double acc  = vix::ai::ml::accuracy01(y_true, y_pred);
   * @endcode
   */

  namespace detail
  {

    /// Sentinel returned by regression metrics on invalid input.
    inline constexpr double INF = std::numeric_limits<double>::infinity();
    /// Sentinel returned by r2_score on invalid input.
    inline double NAN_VAL() noexcept { return std::numeric_limits<double>::quiet_NaN(); }

    /// True when @p y and @p yhat have the same positive size.
    inline bool valid_pair(const Vec &y, const Vec &yhat) noexcept
    {
      return !y.empty() && y.size() == yhat.size();
    }

    /// Threshold a probability to a binary class label (threshold = 0.5).
    inline int threshold(double p) noexcept { return p >= 0.5 ? 1 : 0; }

  } // namespace detail

  /**
   * @brief Mean Squared Error.
   *
   * @code
   *   MSE = (1/n) Σ (yᵢ − ŷᵢ)²
   * @endcode
   *
   * @param y     Ground-truth values.
   * @param yhat  Predicted values.
   * @return MSE ≥ 0, or `+∞` when input is empty or sizes mismatch.
   */
  inline double mse(const Vec &y, const Vec &yhat)
  {
    if (!detail::valid_pair(y, yhat))
      return detail::INF;

    const std::size_t n = y.size();
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
      const double e = yhat[i] - y[i];
      s += e * e;
    }
    return s / static_cast<double>(n);
  }

  /**
   * @brief Mean Absolute Error.
   *
   * @code
   *   MAE = (1/n) Σ |yᵢ − ŷᵢ|
   * @endcode
   *
   * Less sensitive to large outliers than MSE.
   *
   * @param y     Ground-truth values.
   * @param yhat  Predicted values.
   * @return MAE ≥ 0, or `+∞` when input is invalid.
   */
  inline double mae(const Vec &y, const Vec &yhat)
  {
    if (!detail::valid_pair(y, yhat))
      return detail::INF;

    const std::size_t n = y.size();
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i)
      s += std::abs(yhat[i] - y[i]);
    return s / static_cast<double>(n);
  }

  /**
   * @brief Root Mean Squared Error.
   *
   * @code
   *   RMSE = √MSE
   * @endcode
   *
   * Expressed in the same units as the target variable.
   *
   * @param y     Ground-truth values.
   * @param yhat  Predicted values.
   * @return RMSE ≥ 0, or `+∞` when input is invalid.
   */
  inline double rmse(const Vec &y, const Vec &yhat)
  {
    return std::sqrt(mse(y, yhat)); // √+∞ = +∞ — correct sentinel propagation
  }

  /**
   * @brief R² — Coefficient of Determination.
   *
   * Measures the proportion of variance in @p y explained by @p yhat:
   * @code
   *   SS_res = Σ (yᵢ − ŷᵢ)²
   *   SS_tot = Σ (yᵢ − ȳ)²
   *   R²     = 1 − SS_res / SS_tot
   * @endcode
   *
   * A perfect model gives R² = 1.  A baseline model that always predicts the
   * mean gives R² = 0.  Negative values indicate worse-than-baseline performance.
   *
   * **Edge cases:**
   * - Invalid input (empty / size mismatch) → `NaN`.
   * - Zero variance in @p y (constant target) → `0.0` (the model cannot
   *   do better than the constant prediction regardless of its output).
   *
   * @param y     Ground-truth values.
   * @param yhat  Predicted values.
   * @return R² ∈ (−∞, 1], or `NaN` on invalid input.
   */
  inline double r2_score(const Vec &y, const Vec &yhat)
  {
    if (!detail::valid_pair(y, yhat))
      return detail::NAN_VAL();

    const std::size_t n = y.size();

    // Compute mean of y in a single pass
    double mean_y = 0.0;
    for (std::size_t i = 0; i < n; ++i)
      mean_y += y[i];
    mean_y /= static_cast<double>(n);

    double ss_res = 0.0;
    double ss_tot = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
      const double res = yhat[i] - y[i];
      const double dev = y[i] - mean_y;
      ss_res += res * res;
      ss_tot += dev * dev;
    }

    // Zero-variance target: no meaningful R² — return 0 by convention
    if (ss_tot == 0.0)
      return 0.0;

    return 1.0 - ss_res / ss_tot;
  }

  //
  // Conventions:
  //   y    ∈ {0, 1}  — ground-truth class labels
  //   yhat ∈ [0, 1]  — predicted probabilities
  //   threshold = 0.5: ŷ ≥ 0.5 → class 1, else class 0
  //
  // Confusion-matrix counts:
  //   TP  predicted 1, true 1
  //   TN  predicted 0, true 0
  //   FP  predicted 1, true 0
  //   FN  predicted 0, true 1

  /**
   * @brief Binary classification accuracy.
   *
   * @code
   *   accuracy = (TP + TN) / n
   * @endcode
   *
   * @param y     Ground-truth labels ∈ {0, 1}.
   * @param yhat  Predicted probabilities ∈ [0, 1].
   * @return Fraction of correct predictions in [0, 1], or `0.0` on invalid input.
   */
  inline double accuracy01(const Vec &y, const Vec &yhat)
  {
    if (!detail::valid_pair(y, yhat))
      return 0.0;

    const std::size_t n = y.size();
    std::size_t correct = 0;
    for (std::size_t i = 0; i < n; ++i)
      correct += (detail::threshold(yhat[i]) == static_cast<int>(y[i]));
    return static_cast<double>(correct) / static_cast<double>(n);
  }

  /**
   * @brief Binary classification precision.
   *
   * @code
   *   precision = TP / (TP + FP)
   * @endcode
   *
   * Answers: "Of all samples predicted positive, how many were actually positive?"
   *
   * @param y     Ground-truth labels ∈ {0, 1}.
   * @param yhat  Predicted probabilities ∈ [0, 1].
   * @return Precision ∈ [0, 1], or `0.0` when there are no positive predictions
   *         or input is invalid.
   */
  inline double precision01(const Vec &y, const Vec &yhat)
  {
    if (!detail::valid_pair(y, yhat))
      return 0.0;

    const std::size_t n = y.size();
    std::size_t tp = 0, fp = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
      const int pred = detail::threshold(yhat[i]);
      const int truth = static_cast<int>(y[i]);
      if (pred == 1)
      {
        if (truth == 1)
          ++tp;
        else
          ++fp;
      }
    }
    const std::size_t denom = tp + fp;
    return denom == 0 ? 0.0 : static_cast<double>(tp) / static_cast<double>(denom);
  }

  /**
   * @brief Binary classification recall (sensitivity / true-positive rate).
   *
   * @code
   *   recall = TP / (TP + FN)
   * @endcode
   *
   * Answers: "Of all actually positive samples, how many were correctly predicted?"
   *
   * @param y     Ground-truth labels ∈ {0, 1}.
   * @param yhat  Predicted probabilities ∈ [0, 1].
   * @return Recall ∈ [0, 1], or `0.0` when there are no positive ground-truth
   *         samples or input is invalid.
   */
  inline double recall01(const Vec &y, const Vec &yhat)
  {
    if (!detail::valid_pair(y, yhat))
      return 0.0;

    const std::size_t n = y.size();
    std::size_t tp = 0, fn = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
      const int pred = detail::threshold(yhat[i]);
      const int truth = static_cast<int>(y[i]);
      if (truth == 1)
      {
        if (pred == 1)
          ++tp;
        else
          ++fn;
      }
    }
    const std::size_t denom = tp + fn;
    return denom == 0 ? 0.0 : static_cast<double>(tp) / static_cast<double>(denom);
  }

  /**
   * @brief Binary F1 score — harmonic mean of precision and recall.
   *
   * @code
   *   F1 = 2 × (precision × recall) / (precision + recall)
   * @endcode
   *
   * Balances precision and recall into a single scalar.  Useful when the class
   * distribution is skewed.
   *
   * @param y     Ground-truth labels ∈ {0, 1}.
   * @param yhat  Predicted probabilities ∈ [0, 1].
   * @return F1 ∈ [0, 1], or `0.0` when precision + recall = 0 or input is invalid.
   *
   * @note F1 is computed directly from TP/FP/FN counts in a single pass to
   *       avoid recomputing them separately in `precision01` and `recall01`.
   */
  inline double f1_score01(const Vec &y, const Vec &yhat)
  {
    if (!detail::valid_pair(y, yhat))
      return 0.0;

    const std::size_t n = y.size();
    std::size_t tp = 0, fp = 0, fn = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
      const int pred = detail::threshold(yhat[i]);
      const int truth = static_cast<int>(y[i]);

      if (pred == 1 && truth == 1)
        ++tp;
      else if (pred == 1 && truth == 0)
        ++fp;
      else if (pred == 0 && truth == 1)
        ++fn;
      // TN — not needed for F1
    }

    // F1 = 2TP / (2TP + FP + FN)  — equivalent form, avoids a division
    const std::size_t denom = 2 * tp + fp + fn;
    return denom == 0 ? 0.0
                      : (2.0 * static_cast<double>(tp)) / static_cast<double>(denom);
  }

} // namespace vix::ai::ml

#endif // VIX_AI_ML_METRICS_HPP
