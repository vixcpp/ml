/**
 * @file Dataset.cpp
 * @brief Implementation of CSV-based dataset loading utilities for Vix.AI.ML.
 *
 * This file implements the static method `Dataset::from_csv()` which allows
 * lightweight parsing of supervised or unsupervised datasets from CSV files.
 *
 * The implementation avoids external dependencies for simplicity and portability.
 * It is suitable for small to medium datasets, educational purposes, or quick prototyping.
 *
 * ### Example:
 * @code
 * auto ds = Dataset::from_csv("data.csv", true, 3);
 * if (ds) {
 *     std::cout << "Loaded " << ds->size_supervised() << " samples\n";
 * }
 * @endcode
 */

#include <vix/ai/ml/Dataset.hpp>
#include <fstream>
#include <sstream>

namespace vix::ai::ml
{
  /**
   * @brief Parse a single CSV line into string tokens.
   * @param line The input CSV line as a string.
   * @param out  Output vector where tokens will be stored.
   * @return True if at least one token was extracted, false otherwise.
   *
   * This helper splits the input string by commas without quoting support.
   * It is intended for simple, well-formed numeric CSV files.
   */
  static bool parse_line(const std::string &line, std::vector<std::string> &out)
  {
    out.clear();
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ','))
      out.push_back(cell);
    return !out.empty();
  }

  /**
   * @brief Load a dataset from a CSV file.
   *
   * @param path        Path to the CSV file.
   * @param has_header  Whether the CSV file includes a header row (default = true).
   * @param target_col  Index of the target column (for supervised datasets).
   *                    Use -1 for unsupervised datasets (no target column).
   *
   * @return std::optional<Dataset> containing the parsed dataset,
   *         or std::nullopt if the file could not be opened or parsed.
   *
   * ### Behavior:
   * - If `target_col >= 0`: The dataset is treated as supervised.
   *   - All columns except `target_col` are features (X)
   *   - The `target_col` is stored in y
   * - If `target_col == -1`: The dataset is treated as unsupervised.
   *   - All columns are stored in U
   *
   * ### Limitations:
   * - Does not support quoted strings, escaped commas, or missing values.
   * - All values are parsed as doubles via `std::stod()`.
   * - Designed for small datasets loaded entirely into memory.
   */
  std::optional<Dataset> Dataset::from_csv(const std::string &path, bool has_header, int target_col)
  {
    std::ifstream f(path);
    if (!f.is_open())
      return std::nullopt;

    Dataset ds{};
    std::string line;
    std::vector<std::string> cells;

    // Skip header row if requested
    if (has_header && std::getline(f, line))
    {
      /* header skipped */
    }

    // Parse line by line
    while (std::getline(f, line))
    {
      if (!parse_line(line, cells))
        continue;

      // Convert tokens to doubles
      std::vector<double> row;
      row.reserve(cells.size());
      for (const auto &c : cells)
        row.push_back(std::stod(c));

      if (target_col >= 0)
      {
        // Supervised mode
        std::vector<double> feat;
        feat.reserve(row.size() - 1);
        for (int j = 0; j < static_cast<int>(row.size()); ++j)
        {
          if (j == target_col)
            ds.y.push_back(row[j]);
          else
            feat.push_back(row[j]);
        }
        ds.X.push_back(std::move(feat));
      }
      else
      {
        // Unsupervised mode
        ds.U.push_back(std::move(row));
      }
    }

    return ds;
  }

} // namespace vix::ai::ml
