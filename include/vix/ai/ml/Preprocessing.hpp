/**
 * @file Preprocessing.hpp
 * @author Gaspard Kirira
 *
 * Copyright 2025, Gaspard Kirira. All rights reserved.
 * https://github.com/vixcpp/vix
 * Use of this source code is governed by a MIT license that can be found in the License file.
 *
 * Vix.cpp
 */
#ifndef VIX_AI_ML_PREPROCESSING_HPP
#define VIX_AI_ML_PREPROCESSING_HPP

#include <vix/ai/ml/Types.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>

namespace vix::ai::ml
{
  namespace detail
  {

    /// Throw std::invalid_argument if @p X is empty or has zero-width rows.
    inline void require_nonempty(const Mat &X, const char *context)
    {
      if (X.empty() || X[0].empty())
        throw std::invalid_argument(
            std::string(context) + ": X must be a non-empty matrix.");
    }

    /// Throw if the number of columns in @p X does not match @p expected_cols.
    inline void require_cols(const Mat &X, std::size_t expected_cols,
                             const char *context)
    {
      if (ncols(X) != expected_cols)
        throw std::invalid_argument(
            std::string(context) +
            ": X has " + std::to_string(ncols(X)) +
            " column(s); scaler was fitted on " +
            std::to_string(expected_cols) + ".");
    }

  } // namespace detail

  /**
   * @brief Standardise features by removing the mean and scaling to unit variance.
   *
   * Per-column transformation:
   * @code
   *   z = (x - μ) / σ
   * @endcode
   * where μ and σ are computed over the training set.
   * Constant columns (σ = 0) are left with σ = 1 to avoid division by zero.
   *
   * ### Usage
   * @code
   * StandardScaler sc;
   * Mat X_train_scaled = sc.fit_transform(X_train);
   * Mat X_test_scaled  = sc.transform(X_test);
   * @endcode
   */
  struct StandardScaler
  {
    /// Per-feature means (size = n_features after fit).
    Vec mean;
    /// Per-feature standard deviations (size = n_features after fit).
    Vec std;

    /**
     * @brief Compute per-feature mean and standard deviation from @p X.
     *
     * Uses sample standard deviation (denominator = m − 1, or 1 when m = 1).
     *
     * @param X Feature matrix (n_samples × n_features).
     * @throws std::invalid_argument if @p X is empty.
     */
    void fit(const Mat &X)
    {
      detail::require_nonempty(X, "StandardScaler::fit");

      const std::size_t m = nrows(X);
      const std::size_t d = ncols(X);

      mean.assign(d, 0.0);
      std.assign(d, 0.0);

      // Column means
      for (std::size_t j = 0; j < d; ++j)
      {
        double s = 0.0;
        for (std::size_t i = 0; i < m; ++i)
          s += X[i][j];
        mean[j] = s / static_cast<double>(m);
      }

      // Sample standard deviation
      const double denom = static_cast<double>(m > 1 ? m - 1 : 1);
      for (std::size_t j = 0; j < d; ++j)
      {
        double v = 0.0;
        for (std::size_t i = 0; i < m; ++i)
        {
          const double z = X[i][j] - mean[j];
          v += z * z;
        }
        std[j] = std::sqrt(v / denom);
        if (std[j] == 0.0)
          std[j] = 1.0; // constant feature — no rescaling
      }
    }

    /**
     * @brief Apply standardisation using the statistics computed by `fit`.
     *
     * @param X Feature matrix (n_samples × n_features).
     * @return Standardised copy of @p X.
     * @throws std::invalid_argument if column count differs from fit data.
     */
    Mat transform(const Mat &X) const
    {
      detail::require_nonempty(X, "StandardScaler::transform");
      detail::require_cols(X, mean.size(), "StandardScaler::transform");

      const std::size_t m = nrows(X);
      const std::size_t d = ncols(X);

      Mat Z = X;
      for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < d; ++j)
          Z[i][j] = (X[i][j] - mean[j]) / std[j];
      return Z;
    }

    /**
     * @brief Convenience: `fit(X)` then `transform(X)`.
     * @param X Feature matrix (n_samples × n_features).
     * @return Standardised copy of @p X.
     */
    Mat fit_transform(const Mat &X)
    {
      fit(X);
      return transform(X);
    }
  };

  /**
   * @brief Scale each feature to the range [0, 1].
   *
   * Per-column transformation:
   * @code
   *   x' = (x - min) / (max - min)
   * @endcode
   * Constant columns (max = min) are mapped to 0.0 throughout.
   *
   * ### Usage
   * @code
   * MinMaxScaler sc;
   * Mat X_scaled = sc.fit_transform(X_train);
   * @endcode
   */
  struct MinMaxScaler
  {
    /// Per-feature minimum values (size = n_features after fit).
    Vec min;
    /// Per-feature maximum values (size = n_features after fit).
    Vec max;

    /**
     * @brief Compute per-feature min and max from @p X.
     *
     * @param X Feature matrix (n_samples × n_features).
     * @throws std::invalid_argument if @p X is empty.
     */
    void fit(const Mat &X)
    {
      detail::require_nonempty(X, "MinMaxScaler::fit");

      const std::size_t m = nrows(X);
      const std::size_t d = ncols(X);

      min.assign(d, std::numeric_limits<double>::infinity());
      max.assign(d, -std::numeric_limits<double>::infinity());

      for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < d; ++j)
        {
          if (X[i][j] < min[j])
            min[j] = X[i][j];
          if (X[i][j] > max[j])
            max[j] = X[i][j];
        }
    }

    /**
     * @brief Scale @p X to [0, 1] using the statistics computed by `fit`.
     *
     * Constant columns are mapped to 0.0 (range = 0 → no information).
     *
     * @param X Feature matrix (n_samples × n_features).
     * @return Scaled copy of @p X.
     * @throws std::invalid_argument if column count differs from fit data.
     */
    Mat transform(const Mat &X) const
    {
      detail::require_nonempty(X, "MinMaxScaler::transform");
      detail::require_cols(X, min.size(), "MinMaxScaler::transform");

      const std::size_t m = nrows(X);
      const std::size_t d = ncols(X);

      Mat Z = X;
      for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < d; ++j)
        {
          const double range = max[j] - min[j];
          Z[i][j] = (range == 0.0) ? 0.0
                                   : (X[i][j] - min[j]) / range;
        }
      return Z;
    }

    /**
     * @brief Convenience: `fit(X)` then `transform(X)`.
     * @param X Feature matrix (n_samples × n_features).
     * @return Scaled copy of @p X.
     */
    Mat fit_transform(const Mat &X)
    {
      fit(X);
      return transform(X);
    }
  };

  /**
   * @brief Scale each feature by its maximum absolute value.
   *
   * Per-column transformation:
   * @code
   *   x' = x / max(|x|)
   * @endcode
   * After scaling each feature lies in [−1, 1].
   * Columns whose max absolute value is zero are left unchanged (mapped to 0).
   * This scaler does **not** shift the data, preserving sparsity.
   *
   * ### Usage
   * @code
   * MaxAbsScaler sc;
   * Mat X_scaled = sc.fit_transform(X_train);
   * @endcode
   */
  struct MaxAbsScaler
  {
    /// Per-feature maximum absolute values (size = n_features after fit).
    Vec maxabs;

    /**
     * @brief Compute the maximum absolute value of each feature column.
     *
     * @param X Feature matrix (n_samples × n_features).
     * @throws std::invalid_argument if @p X is empty.
     */
    void fit(const Mat &X)
    {
      detail::require_nonempty(X, "MaxAbsScaler::fit");

      const std::size_t m = nrows(X);
      const std::size_t d = ncols(X);

      maxabs.assign(d, 0.0);

      for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < d; ++j)
        {
          const double a = std::abs(X[i][j]);
          if (a > maxabs[j])
            maxabs[j] = a;
        }
    }

    /**
     * @brief Divide each feature by its max absolute value.
     *
     * Columns with max absolute value == 0 are left as 0.
     *
     * @param X Feature matrix (n_samples × n_features).
     * @return Scaled copy of @p X, values in [−1, 1].
     * @throws std::invalid_argument if column count differs from fit data.
     */
    Mat transform(const Mat &X) const
    {
      detail::require_nonempty(X, "MaxAbsScaler::transform");
      detail::require_cols(X, maxabs.size(), "MaxAbsScaler::transform");

      const std::size_t m = nrows(X);
      const std::size_t d = ncols(X);

      Mat Z = X;
      for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < d; ++j)
          Z[i][j] = (maxabs[j] == 0.0) ? 0.0
                                       : X[i][j] / maxabs[j];
      return Z;
    }

    /**
     * @brief Convenience: `fit(X)` then `transform(X)`.
     * @param X Feature matrix (n_samples × n_features).
     * @return Scaled copy of @p X.
     */
    Mat fit_transform(const Mat &X)
    {
      fit(X);
      return transform(X);
    }
  };

  /**
   * @brief Split arrays into random train and test subsets.
   *
   * Rows of @p X and the corresponding elements of @p y are kept aligned
   * throughout the split.  The split point is determined by @p test_ratio.
   *
   * @param X          Feature matrix (n_samples × n_features).
   * @param y          Target vector  (n_samples).
   * @param test_ratio Fraction of the dataset to hold out for testing (0 < ratio < 1).
   * @param shuffle    Shuffle row indices before splitting when `true`.
   * @param seed       RNG seed used when @p shuffle is `true`.
   * @return `std::tuple<Mat, Mat, Vec, Vec>` — (X_train, X_test, y_train, y_test).
   *
   * @throws std::invalid_argument if @p X is empty, @p y.size() != nrows(X), or
   *         @p test_ratio is outside (0, 1).
   *
   * ### Example
   * @code
   * auto [X_tr, X_te, y_tr, y_te] = train_test_split(X, y, 0.2);
   * @endcode
   */
  inline std::tuple<Mat, Mat, Vec, Vec>
  train_test_split(
      const Mat &X,
      const Vec &y,
      double test_ratio = 0.2,
      bool shuffle = true,
      unsigned seed = 42)
  {
    if (X.empty())
      throw std::invalid_argument("train_test_split: X must not be empty.");
    if (y.size() != nrows(X))
      throw std::invalid_argument(
          "train_test_split: y.size() (" + std::to_string(y.size()) +
          ") != nrows(X) (" + std::to_string(nrows(X)) + ").");
    if (test_ratio <= 0.0 || test_ratio >= 1.0)
      throw std::invalid_argument(
          "train_test_split: test_ratio must be in (0, 1).");

    const std::size_t m = nrows(X);

    // Build index array
    std::vector<std::size_t> idx(m);
    std::iota(idx.begin(), idx.end(), std::size_t{0});

    if (shuffle)
    {
      std::mt19937 rng(seed);
      std::shuffle(idx.begin(), idx.end(), rng);
    }

    // Determine split boundary
    const std::size_t n_test = std::max(std::size_t{1},
                                        static_cast<std::size_t>(std::round(static_cast<double>(m) * test_ratio)));
    const std::size_t n_train = m - n_test;

    // Allocate output containers
    Mat X_train(n_train), X_test(n_test);
    Vec y_train(n_train), y_test(n_test);

    for (std::size_t k = 0; k < n_train; ++k)
    {
      X_train[k] = X[idx[k]];
      y_train[k] = y[idx[k]];
    }
    for (std::size_t k = 0; k < n_test; ++k)
    {
      X_test[k] = X[idx[n_train + k]];
      y_test[k] = y[idx[n_train + k]];
    }

    return {std::move(X_train), std::move(X_test),
            std::move(y_train), std::move(y_test)};
  }

  /**
   * @brief Shuffle samples in-place while keeping @p X and @p y aligned.
   *
   * Applies the same random permutation to both the rows of @p X and the
   * elements of @p y, so the i-th row of X still corresponds to y[i] after
   * shuffling.
   *
   * @param X    Feature matrix (n_samples × n_features). Modified in-place.
   * @param y    Target vector  (n_samples).              Modified in-place.
   * @param seed RNG seed for reproducibility.
   *
   * @throws std::invalid_argument if @p X is empty or y.size() != nrows(X).
   *
   * ### Example
   * @code
   * shuffle_dataset(X, y, 123);
   * @endcode
   */
  inline void shuffle_dataset(Mat &X, Vec &y, unsigned seed = 42)
  {
    if (X.empty())
      throw std::invalid_argument("shuffle_dataset: X must not be empty.");
    if (y.size() != nrows(X))
      throw std::invalid_argument(
          "shuffle_dataset: y.size() (" + std::to_string(y.size()) +
          ") != nrows(X) (" + std::to_string(nrows(X)) + ").");

    const std::size_t m = nrows(X);

    std::vector<std::size_t> idx(m);
    std::iota(idx.begin(), idx.end(), std::size_t{0});

    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);

    // Apply the permutation with O(m) auxiliary storage
    Mat X_tmp(m);
    Vec y_tmp(m);
    for (std::size_t i = 0; i < m; ++i)
    {
      X_tmp[i] = std::move(X[idx[i]]);
      y_tmp[i] = y[idx[i]];
    }

    X = std::move(X_tmp);
    y = std::move(y_tmp);
  }

  /**
   * @brief L2-normalise each row of @p X to unit Euclidean length.
   *
   * Per-row transformation:
   * @code
   *   x' = x / ||x||₂
   * @endcode
   * Rows with zero norm are left unchanged (all zeros stay all zeros).
   *
   * Useful before computing cosine similarities or when working with
   * text/embedding representations.
   *
   * @param X Feature matrix (n_samples × n_features).
   * @return Row-normalised copy of @p X.
   *
   * @throws std::invalid_argument if @p X is empty.
   *
   * ### Example
   * @code
   * Mat X_norm = normalize_l2(X);
   * @endcode
   */
  inline Mat normalize_l2(const Mat &X)
  {
    detail::require_nonempty(X, "normalize_l2");

    const std::size_t m = nrows(X);
    const std::size_t d = ncols(X);

    Mat Z = X;
    for (std::size_t i = 0; i < m; ++i)
    {
      // Compute L2 norm of this row
      double norm2 = 0.0;
      for (std::size_t j = 0; j < d; ++j)
        norm2 += X[i][j] * X[i][j];

      const double norm = std::sqrt(norm2);
      if (norm == 0.0)
        continue; // zero-row — leave unchanged

      for (std::size_t j = 0; j < d; ++j)
        Z[i][j] = X[i][j] / norm;
    }
    return Z;
  }

} // namespace vix::ai::ml

#endif // VIX_AI_ML_PREPROCESSING_HPP
