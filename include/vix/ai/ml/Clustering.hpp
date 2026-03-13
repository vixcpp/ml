/**
 * @file CLustering.hpp
 * @author Gaspard Kirira
 *
 * Copyright 2025, Gaspard Kirira. All rights reserved.
 * https://github.com/vixcpp/vix
 * Use of this source code is governed by a MIT license that can be found in the License file.
 *
 * Vix.cpp
 */
#ifndef VIX_AI_NLP_CLUSTERING_HP
#define VIX_AI_NLP_CLUSTERING_HPP

#include <vix/ai/ml/Model.hpp>
#include <random>
#include <limits>

namespace vix::ai::ml
{
  /**
   * @brief K-Means clustering algorithm (Lloyd's algorithm).
   *
   * This implementation performs unsupervised partitioning of a dataset
   * into k clusters by minimizing the within-cluster sum of squares (WCSS).
   * It alternates between assignment (E-step) and centroid update (M-step)
   * until convergence or a fixed number of iterations.
   *
   * ### Features:
   * - Deterministic initialization (fixed RNG seed for reproducibility)
   * - Handles empty clusters gracefully (keeps previous centers)
   * - Compatible with the Model interface (predict_one returns cluster ID)
   *
   * ### Usage example:
   * @code
   * KMeans km(3);
   * km.fit(X);                       // X: n_samples × n_features
   * auto label = km.predict_label(x); // cluster index for one sample
   * const auto& centers = km.centers();
   * @endcode
   *
   * ### Complexity:
   * O(max_iters × n_samples × k × n_features)
   *
   * @note This is a minimal in-memory version suitable for small to medium datasets.
   *       It is not optimized for streaming or mini-batching.
   */
  class KMeans final : public Model
  {
  public:
    /**
     * @brief Construct a KMeans instance.
     * @param k          Number of clusters to form.
     * @param max_iters  Maximum number of iterations (default = 100).
     */
    explicit KMeans(std::size_t k = 2, std::size_t max_iters = 100)
        : k_(k), max_iters_(max_iters) {}

    /**
     * @brief Fit the model to the input data X.
     * @param X Input feature matrix (n_samples × n_features).
     *
     * Performs iterative assignment and centroid recomputation until
     * convergence or reaching max_iters_. Cluster centers are initialized
     * randomly from existing samples.
     */
    void fit(const Mat &X) override
    {
      const auto m = nrows(X), d = ncols(X);
      if (m == 0 || d == 0 || k_ == 0)
        return;

      // Initialize k random centroids from the dataset
      std::mt19937 rng(42);
      std::uniform_int_distribution<std::size_t> dist(0, m - 1);
      centers_.assign(k_, Vec(d, 0.0));
      for (std::size_t c = 0; c < k_; ++c)
        centers_[c] = X[dist(rng)];

      Idxs assign(m, 0);
      for (std::size_t it = 0; it < max_iters_; ++it)
      {
        bool changed = false;

        // E-step: assign each sample to nearest centroid
        for (std::size_t i = 0; i < m; ++i)
        {
          std::size_t best = 0;
          double bestd = std::numeric_limits<double>::infinity();
          for (std::size_t c = 0; c < k_; ++c)
          {
            double d2 = sqdist(X[i], centers_[c]);
            if (d2 < bestd)
            {
              bestd = d2;
              best = c;
            }
          }
          if (assign[i] != best)
          {
            assign[i] = best;
            changed = true;
          }
        }

        // M-step: recompute centroids
        Mat newC(k_, Vec(d, 0.0));
        Vec cnt(k_, 0.0);
        for (std::size_t i = 0; i < m; ++i)
        {
          const auto c = assign[i];
          cnt[c] += 1.0;
          for (std::size_t j = 0; j < d; ++j)
            newC[c][j] += X[i][j];
        }

        for (std::size_t c = 0; c < k_; ++c)
        {
          if (cnt[c] > 0)
          {
            for (std::size_t j = 0; j < d; ++j)
              newC[c][j] /= cnt[c];
          }
          else
          {
            // Handle empty cluster: keep previous center
            newC[c] = centers_[c];
          }
        }

        centers_.swap(newC);
        if (!changed)
          break;
      }
    }

    /**
     * @brief Predict the cluster label (0..k-1) for a given input vector.
     * @param x Single feature vector (size = n_features).
     * @return Index of the nearest cluster center.
     */
    std::size_t predict_label(const Vec &x) const
    {
      std::size_t best = 0;
      double bestd = std::numeric_limits<double>::infinity();
      for (std::size_t c = 0; c < k_; ++c)
      {
        double d2 = sqdist(x, centers_[c]);
        if (d2 < bestd)
        {
          bestd = d2;
          best = c;
        }
      }
      return best;
    }

    /**
     * @brief Predict cluster ID as a numeric value (Model interface).
     * @param x Feature vector.
     * @return Cluster index as a double.
     */
    double predict_one(const Vec &x) const override
    {
      return static_cast<double>(predict_label(x));
    }

    /**
     * @brief Get cluster centers.
     * @return Matrix (k × n_features) of current cluster centroids.
     */
    const Mat &centers() const noexcept { return centers_; }

    /**
     * @brief Get the number of clusters (k).
     */
    std::size_t k() const noexcept { return k_; }

  private:
    static double sqdist(const Vec &a, const Vec &b)
    {
      double s = 0;
      for (std::size_t j = 0; j < a.size(); ++j)
      {
        const double z = a[j] - b[j];
        s += z * z;
      }
      return s;
    }

    // Number of clusters
    std::size_t k_;

    // Maximum number of iterations
    std::size_t max_iters_;

    // Centroid matrix (k × n_features)
    Mat centers_;
  };

} // namespace vix::ai::ml

#endif
