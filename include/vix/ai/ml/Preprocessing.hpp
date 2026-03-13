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
#include <tuple>
#include <algorithm>
#include <cmath>

namespace vix::ai::ml
{
  /**
   * @file Preprocessing.hpp
   * @brief Basic data preprocessing utilities for Vix.AI.ML.
   *
   * This header provides standard data normalization and scaling utilities
   * used prior to training machine learning models.
   *
   * Currently implemented:
   *  - **StandardScaler:** mean-centering and variance scaling (z-score normalization)
   *
   * These utilities are designed to work directly on in-memory matrices (Mat)
   * and operate with minimal dependencies.
   *
   * ### Example:
   * @code
   * StandardScaler scaler;
   * scaler.fit(X_train);
   * Mat X_scaled = scaler.transform(X_test);
   * @endcode
   */

  /**
   * @brief Standardize features by removing the mean and scaling to unit variance.
   *
   * Each feature column in the dataset is standardized independently as:
   * \f[
   *   z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}
   * \f]
   * where \f$\mu_j\f$ and \f$\sigma_j\f$ are the mean and standard deviation
   * of feature j over all samples.
   *
   * This process improves the convergence of many gradient-based algorithms
   * and ensures each feature contributes equally to the model.
   *
   * ### Features:
   * - Avoids division by zero by replacing zero-variance features with σ = 1.
   * - Provides fit(), transform(), and fit_transform() methods.
   *
   * ### Usage:
   * @code
   * StandardScaler scaler;
   * auto X_scaled = scaler.fit_transform(X);
   * @endcode
   */
  struct StandardScaler
  {
    // Feature-wise mean values (size = n_features).
    Vec mean;

    // Feature-wise standard deviations (size = n_features).
    Vec std;

    /**
     * @brief Compute per-feature mean and standard deviation.
     * @param X Input feature matrix (n_samples × n_features).
     *
     * Populates `mean` and `std` based on the provided data.
     * If any feature has zero variance, its std is replaced with 1.0.
     */
    void fit(const Mat &X)
    {
      const auto m = nrows(X), d = ncols(X);
      mean.assign(d, 0.0);
      std.assign(d, 0.0);
      if (m == 0 || d == 0)
        return;

      // Compute mean
      for (std::size_t j = 0; j < d; ++j)
      {
        double s = 0;
        for (std::size_t i = 0; i < m; ++i)
          s += X[i][j];
        mean[j] = s / static_cast<double>(m);

        // Compute variance
        double v = 0;
        for (std::size_t i = 0; i < m; ++i)
        {
          const double z = X[i][j] - mean[j];
          v += z * z;
        }

        std[j] = std::sqrt(v / static_cast<double>(m > 1 ? m - 1 : 1));
        if (std[j] == 0)
          std[j] = 1.0; // avoid division by zero (constant feature)
      }
    }

    /**
     * @brief Apply standardization to the given data using precomputed statistics.
     * @param X Input feature matrix (n_samples × n_features).
     * @return Transformed matrix (standardized features).
     *
     * Assumes that `fit()` has already been called. Does not modify input X.
     */
    Mat transform(const Mat &X) const
    {
      Mat Z = X;
      const auto m = nrows(X), d = ncols(X);
      for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < d; ++j)
          Z[i][j] = (X[i][j] - mean[j]) / std[j];
      return Z;
    }

    /**
     * @brief Fit the scaler on X, then transform X.
     * @param X Input feature matrix (n_samples × n_features).
     * @return Standardized version of X.
     *
     * Equivalent to calling fit(X) followed by transform(X).
     */
    Mat fit_transform(const Mat &X)
    {
      fit(X);
      return transform(X);
    }
  };

} // namespace vix::ai::ml

#endif
