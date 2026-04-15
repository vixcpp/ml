/**
 *
 *  @file Dataset.hpp
 *  @author Gaspard Kirira
 *
 *  Copyright 2025, Gaspard Kirira.
 *  All rights reserved.
 *  https://github.com/vixcpp/vix
 *
 *  Use of this source code is governed by a MIT license
 *  that can be found in the License file.
 *
 *  Vix.cpp
 */
#ifndef VIX_AI_ML_DATASET_HPP
#define VIX_AI_ML_DATASET_HPP

#include <vix/ai/ml/Types.hpp>

#include <optional>
#include <string>
#include <tuple>

namespace vix::ai::ml
{

  /**
   * @brief Lightweight dataset container for supervised and unsupervised ML workflows.
   *
   * A `Dataset` holds data in one of two modes:
   *
   * | Mode           | Active members | Populated by               |
   * |----------------|----------------|----------------------------|
   * | Supervised     | `X`, `y`       | `from_csv(..., target_col≥0)` |
   * | Unsupervised   | `U`            | `from_csv(..., target_col=-1)` |
   *
   * The two modes are mutually exclusive: a supervised dataset has non-empty `X`
   * and `y` while `U` is empty, and vice versa.
   *
   * ### Typical workflow
   * @code
   * // Load
   * auto ds = Dataset::from_csv("iris.csv", true, 4);
   * if (!ds) { std::cerr << "load failed\n"; return 1; }
   *
   * // Inspect
   * std::cout << ds->size_supervised() << " samples, "
   *           << ds->n_features()      << " features\n";
   *
   * // Split
   * auto [train, test] = ds->train_test_split(0.2);
   *
   * // Persist
   * train.to_csv("train.csv");
   * @endcode
   *
   * @note No preprocessing is applied during loading.  Use the `Preprocessing`
   *       utilities (StandardScaler, MinMaxScaler, …) on `X` or `U` after loading.
   */
  struct Dataset
  {
    /// Feature matrix for supervised learning  (n_samples × n_features).
    Mat X;
    /// Target vector for supervised learning   (n_samples).
    Vec y;
    /// Feature matrix for unsupervised learning (n_samples × n_features).
    Mat U;

    /// Number of supervised samples (rows in X).
    std::size_t size_supervised() const noexcept { return X.size(); }
    /// Number of unsupervised samples (rows in U).
    std::size_t size_unsupervised() const noexcept { return U.size(); }

    /**
     * @brief Number of features per sample.
     * @return `X[0].size()` for supervised datasets, `U[0].size()` for
     *         unsupervised datasets, or 0 if both are empty.
     */
    std::size_t n_features() const noexcept;

    /// True when supervised data (X, y) is present and non-empty.
    bool is_supervised() const noexcept { return !X.empty() && !y.empty(); }
    /// True when unsupervised data (U) is present and non-empty.
    bool is_unsupervised() const noexcept { return !U.empty(); }

    /**
     * @brief Parse a numeric CSV file into a Dataset.
     *
     * @param path        Path to the CSV file.
     * @param has_header  Skip the first line if `true` (default).
     * @param target_col  Column index of the target variable.
     *                    - `>= 0`: supervised — that column goes into `y`,
     *                      the rest into `X`.
     *                    - `-1`:   unsupervised — all columns go into `U`.
     *
     * @return A populated `Dataset`, or `std::nullopt` when the file cannot
     *         be opened.  Rows with parse errors or inconsistent column counts
     *         are silently skipped.
     *
     * ### Limitations
     * - Numeric values only (`std::stod` parsing).
     * - No quoted fields or escaped commas.
     * - No missing-value handling.
     */
    static std::optional<Dataset> from_csv(
        const std::string &path,
        bool has_header = true,
        int target_col = -1);

    /**
     * @brief Serialise the dataset to a CSV file.
     *
     * For supervised datasets each row is written as:
     * @code
     *   x0,x1,...,xN,y
     * @endcode
     * The target is always appended as the last column.
     *
     * For unsupervised datasets each row of `U` is written as-is.
     *
     * @param path Output file path.
     * @return `true` on success, `false` if the file could not be opened.
     */
    bool to_csv(const std::string &path) const;

    /**
     * @brief Return a contiguous subset of rows [start, end).
     *
     * Works for both supervised and unsupervised datasets.
     *
     * @param start First row index (inclusive).
     * @param end   One-past-last row index (exclusive).
     * @return A new Dataset containing the selected rows.
     * @throws std::out_of_range if indices are invalid.
     */
    Dataset slice(std::size_t start, std::size_t end) const;

    /**
     * @brief Return a copy with rows shuffled in random order.
     *
     * The correspondence between `X` rows and `y` elements is preserved.
     *
     * @param seed RNG seed for reproducibility (default 42).
     * @return Shuffled copy of this dataset.
     */
    Dataset shuffle(unsigned seed = 42) const;

    /**
     * @brief Split the dataset into train and test subsets.
     *
     * Rows are shuffled before splitting to avoid ordering bias.
     *
     * @param test_ratio Fraction of samples reserved for the test set
     *                   (must be in (0, 1); default 0.2).
     * @param seed       RNG seed (default 42).
     * @return `{train_dataset, test_dataset}`.
     * @throws std::invalid_argument if `test_ratio` is outside (0, 1).
     */
    std::tuple<Dataset, Dataset>
    train_test_split(double test_ratio = 0.2, unsigned seed = 42) const;
  };

} // namespace vix::ai::ml

#endif // VIX_AI_ML_DATASET_HPP
