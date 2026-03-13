/**
 * @file Dataset.hpp
 * @author Gaspard Kirira
 *
 * Copyright 2025, Gaspard Kirira. All rights reserved.
 * https://github.com/vixcpp/vix
 * Use of this source code is governed by a MIT license that can be found in the License file.
 *
 * Vix.cpp
 */
#ifndef VIX_AI_ML_DATASET_HPP
#define VIX_AI_ML_DATASET_HPP

#include <vix/ai/ml/Types.hpp>
#include <string>
#include <optional>

namespace vix::ai::ml
{
  /**
   * @file Dataset.hpp
   * @brief Defines the Dataset structure used across the Vix.AI.ML module.
   *
   * The Dataset structure provides a minimal abstraction for handling
   * supervised and unsupervised learning data in memory.
   * It supports convenient introspection (sample/feature counts)
   * and static loading from CSV files.
   *
   * ### Supported modes:
   * - **Supervised:** (X, y) — feature matrix and target vector.
   * - **Unsupervised:** (U) — unlabeled feature matrix.
   *
   * @note This struct is intentionally lightweight and has no dynamic preprocessing;
   *       transformations (normalization, shuffling, etc.) should be applied externally
   *       via `Preprocessing` utilities.
   */
  struct Dataset
  {
    // Features for supervised learning (n_samples × n_features).
    Mat X;

    // Target values for supervised learning (n_samples).
    Vec y;

    // Unlabeled data for unsupervised learning (n_samples × n_features).
    Mat U;

    /**
     * @brief Get the number of supervised samples (rows in X).
     * @return Number of rows in X.
     */
    std::size_t size_supervised() const noexcept { return X.size(); }

    /**
     * @brief Get the number of unsupervised samples (rows in U).
     * @return Number of rows in U.
     */
    std::size_t size_unsupervised() const noexcept { return U.size(); }

    /**
     * @brief Get the number of features (columns) per sample in X.
     * @return Number of features, or 0 if X is empty.
     */
    std::size_t n_features() const noexcept { return X.empty() ? 0 : X[0].size(); }

    /**
     * @brief Load a dataset from a CSV file.
     *
     * @param path        Path to the CSV file on disk.
     * @param has_header  Whether the CSV file includes a header row (default = true).
     * @param target_col  Column index of the target variable (for supervised learning).
     *                    Use -1 for unsupervised datasets (no y column).
     *
     * @return std::optional<Dataset> containing the loaded data,
     *         or std::nullopt if the file could not be parsed.
     *
     * ### Example:
     * @code
     * auto ds = Dataset::from_csv("data.csv", true, 3);
     * if (ds) {
     *     std::cout << "Loaded " << ds->size_supervised() << " samples\n";
     * }
     * @endcode
     */
    static std::optional<Dataset> from_csv(
        const std::string &path,
        bool has_header = true,
        int target_col = -1 // -1 → no target column (unsupervised)
    );
  };

} // namespace vix::ai::ml

#endif
