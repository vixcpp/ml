/**
 * @file Dataset.cpp
 * @author Gaspard Kirira
 *
 * Copyright 2025, Gaspard Kirira. All rights reserved.
 * https://github.com/vixcpp/vix
 * Use of this source code is governed by a MIT license that can be found in the License file.
 *
 * Vix.cpp
 */
#include <vix/ai/ml/Dataset.hpp>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace vix::ai::ml
{
  namespace
  {

    /**
     * @brief Split a CSV line into string tokens on commas.
     *
     * Simple split — no quoting, no escape handling, pure numeric CSV assumed.
     *
     * @param line  Raw text line from the file.
     * @param[out] tokens  Populated with one token per comma-delimited field.
     * @return `true` when at least one non-empty token was produced.
     */
    bool split_csv_line(const std::string &line, std::vector<std::string> &tokens)
    {
      tokens.clear();
      if (line.empty())
        return false;

      std::istringstream ss(line);
      std::string cell;
      while (std::getline(ss, cell, ','))
        tokens.push_back(cell);

      return !tokens.empty();
    }

    /**
     * @brief Convert a vector of string tokens to doubles.
     *
     * @param tokens  String tokens from `split_csv_line`.
     * @param[out] row  Output numeric row.
     * @return `true` on complete success; `false` if any token fails `std::stod`.
     */
    bool tokens_to_doubles(const std::vector<std::string> &tokens,
                           std::vector<double> &row)
    {
      row.clear();
      row.reserve(tokens.size());
      for (const auto &t : tokens)
      {
        try
        {
          std::size_t consumed = 0;
          const double v = std::stod(t, &consumed);
          // Reject tokens with trailing non-numeric garbage
          if (consumed != t.size())
            return false;
          row.push_back(v);
        }
        catch (const std::invalid_argument &)
        {
          return false;
        }
        catch (const std::out_of_range &)
        {
          return false;
        }
      }
      return !row.empty();
    }

    /**
     * @brief Strip leading and trailing whitespace (spaces, CR, LF, TAB) in-place.
     *
     * Windows-style line endings leave a trailing `\r` when `std::getline` is used
     * on a file opened in text mode; this helper removes it.
     */
    void trim(std::string &s)
    {
      const auto not_space = [](unsigned char c)
      { return c != ' ' && c != '\t' && c != '\r' && c != '\n'; };
      s.erase(s.begin(),
              std::find_if(s.begin(), s.end(), not_space));
      s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(),
              s.end());
    }

  } // namespace (anonymous)

  std::size_t Dataset::n_features() const noexcept
  {
    if (!X.empty())
      return X[0].size();
    if (!U.empty())
      return U[0].size();
    return 0;
  }

  std::optional<Dataset> Dataset::from_csv(
      const std::string &path,
      bool has_header,
      int target_col)
  {
    std::ifstream file(path);
    if (!file.is_open())
      return std::nullopt;

    Dataset ds;
    std::string line;
    std::vector<std::string> tokens;
    std::vector<double> row;

    // skip / read header
    if (has_header)
    {
      if (!std::getline(file, line))
        return ds; // empty file after header → valid but empty dataset
    }

    // Reference column count — set on the first successfully parsed data row.
    // All subsequent rows must match; mismatches are silently skipped.
    std::size_t expected_cols = 0;
    bool first_row = true;

    while (std::getline(file, line))
    {
      trim(line);
      if (line.empty())
        continue; // blank line
      if (!split_csv_line(line, tokens))
        continue; // parse failure
      if (!tokens_to_doubles(tokens, row))
        continue; // conversion error

      // Establish or verify column count
      if (first_row)
      {
        expected_cols = row.size();
        first_row = false;
      }
      else if (row.size() != expected_cols)
      {
        continue; // inconsistent width — skip
      }

      // supervised mode
      if (target_col >= 0)
      {
        const auto tcol = static_cast<std::size_t>(target_col);

        // target_col must be a valid column index
        if (tcol >= row.size())
          continue;

        Vec feat;
        feat.reserve(row.size() - 1);
        for (std::size_t j = 0; j < row.size(); ++j)
        {
          if (j == tcol)
            ds.y.push_back(row[j]);
          else
            feat.push_back(row[j]);
        }
        ds.X.push_back(std::move(feat));
      }
      // unsupervised mode
      else
      {
        ds.U.push_back(row);
      }
    }

    return ds;
  }

  bool Dataset::to_csv(const std::string &path) const
  {
    std::ofstream file(path);
    if (!file.is_open())
      return false;

    file << std::setprecision(17);

    if (is_supervised())
    {
      // Each row: x0,x1,...,xN-1,y
      const std::size_t m = X.size();
      for (std::size_t i = 0; i < m; ++i)
      {
        for (std::size_t j = 0; j < X[i].size(); ++j)
        {
          if (j > 0)
            file << ',';
          file << X[i][j];
        }
        // Append target as last column (guard against y being shorter)
        if (i < y.size())
          file << ',' << y[i];
        file << '\n';
      }
    }
    else
    {
      // Each row of U written as-is
      for (const auto &row : U)
      {
        for (std::size_t j = 0; j < row.size(); ++j)
        {
          if (j > 0)
            file << ',';
          file << row[j];
        }
        file << '\n';
      }
    }

    return file.good();
  }

  Dataset Dataset::slice(std::size_t start, std::size_t end) const
  {
    // Determine total sample count for bounds checking
    const std::size_t total = is_supervised() ? X.size() : U.size();

    if (start > end)
      throw std::out_of_range(
          "Dataset::slice: start (" + std::to_string(start) +
          ") > end (" + std::to_string(end) + ").");
    if (end > total)
      throw std::out_of_range(
          "Dataset::slice: end (" + std::to_string(end) +
          ") exceeds dataset size (" + std::to_string(total) + ").");

    Dataset out;

    if (is_supervised())
    {
      out.X.assign(X.begin() + static_cast<std::ptrdiff_t>(start),
                   X.begin() + static_cast<std::ptrdiff_t>(end));
      out.y.assign(y.begin() + static_cast<std::ptrdiff_t>(start),
                   y.begin() + static_cast<std::ptrdiff_t>(end));
    }
    else
    {
      out.U.assign(U.begin() + static_cast<std::ptrdiff_t>(start),
                   U.begin() + static_cast<std::ptrdiff_t>(end));
    }

    return out;
  }

  Dataset Dataset::shuffle(unsigned seed) const
  {
    const std::size_t m = is_supervised() ? X.size() : U.size();
    if (m == 0)
      return *this;

    // Build a shuffled index permutation
    std::vector<std::size_t> idx(m);
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);

    Dataset out;

    if (is_supervised())
    {
      out.X.resize(m);
      out.y.resize(m);
      for (std::size_t i = 0; i < m; ++i)
      {
        out.X[i] = X[idx[i]];
        out.y[i] = y[idx[i]];
      }
    }
    else
    {
      out.U.resize(m);
      for (std::size_t i = 0; i < m; ++i)
        out.U[i] = U[idx[i]];
    }

    return out;
  }

  std::tuple<Dataset, Dataset>
  Dataset::train_test_split(double test_ratio, unsigned seed) const
  {
    if (test_ratio <= 0.0 || test_ratio >= 1.0)
      throw std::invalid_argument(
          "Dataset::train_test_split: test_ratio must be in (0, 1); got " +
          std::to_string(test_ratio) + ".");

    const std::size_t m = is_supervised() ? X.size() : U.size();
    if (m == 0)
      return {Dataset{}, Dataset{}};

    // Shuffle first to avoid ordering bias, then split
    const Dataset shuffled = this->shuffle(seed);

    const std::size_t n_test = std::max(std::size_t{1},
                                        static_cast<std::size_t>(std::round(static_cast<double>(m) * test_ratio)));
    const std::size_t n_train = m - n_test;

    return {shuffled.slice(0, n_train), shuffled.slice(n_train, m)};
  }

} // namespace vix::ai::ml
