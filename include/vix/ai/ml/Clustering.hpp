/**
 * @file Clustering.hpp
 * @author Gaspard Kirira
 *
 * Copyright 2025, Gaspard Kirira. All rights reserved.
 * https://github.com/vixcpp/vix
 * Use of this source code is governed by a MIT license that can be found in the License file.
 *
 * Vix.cpp
 */
#ifndef VIX_AI_ML_CLUSTERING_HPP
#define VIX_AI_ML_CLUSTERING_HPP

#include <vix/ai/ml/Model.hpp>

#include <algorithm>
#include <iomanip>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace vix::ai::ml
{
  /**
   * @brief K-Means clustering via Lloyd's algorithm with optional k-means++
   *        initialisation.
   *
   * Partitions @p n_samples rows of a feature matrix into @p k clusters by
   * minimising the within-cluster sum of squares (WCSS / inertia):
   * @code
   *   WCSS = Σᵢ ‖xᵢ − μ_{aᵢ}‖²
   * @endcode
   *
   * ### Initialisation modes
   * | `use_kmeanspp_` | Strategy |
   * |---|---|
   * | `false` (default) | Pick @p k distinct rows at random. |
   * | `true`            | K-Means++ — seeds first center uniformly, then each
   *                       subsequent center with probability ∝ D²(x). |
   *
   * K-Means++ typically converges faster and to a lower inertia than random
   * initialisation.
   *
   * ### Usage
   * @code
   * KMeans km(3, 300, 42, true);   // k=3, iters=300, seed=42, k-means++
   * km.fit(X);
   * Idxs labels = km.predict_labels(X);
   * double wcss  = km.inertia(X);
   * @endcode
   *
   * ### Complexity
   * O(max_iters × n_samples × k × n_features) — same as standard Lloyd's.
   */
  class KMeans final : public Model
  {
  public:
    /**
     * @param k             Number of clusters (must be ≥ 1).
     * @param max_iters     Maximum Lloyd's iterations (default 300).
     * @param seed          RNG seed for reproducibility (default 42).
     * @param use_kmeanspp  Use k-means++ seeding when `true` (default false).
     */
    explicit KMeans(std::size_t k = 2,
                    std::size_t max_iters = 300,
                    unsigned seed = 42,
                    bool use_kmeanspp = false)
        : k_(k), max_iters_(max_iters), seed_(seed),
          use_kmeanspp_(use_kmeanspp)
    {
    }

    /**
     * @brief Fit cluster centres to the data.
     *
     * Alternates between:
     * - **E-step**: assign every sample to its nearest centroid.
     * - **M-step**: recompute each centroid as the mean of assigned samples.
     *
     * Stops early when no assignment changes between two consecutive iterations.
     * Empty clusters retain their previous centre.
     *
     * @param X Feature matrix (n_samples × n_features).
     * @throws std::invalid_argument if @p X is empty.
     */
    void fit(const Mat &X) override
    {
      const std::size_t m = nrows(X);
      const std::size_t d = ncols(X);

      if (m == 0 || d == 0)
        throw std::invalid_argument("KMeans::fit: X must not be empty.");
      if (k_ == 0)
        throw std::invalid_argument("KMeans::fit: k must be >= 1.");

      // Clamp k to the number of available samples
      const std::size_t k_eff = std::min(k_, m);

      std::mt19937 rng(seed_);

      // Initialise centres
      centers_ = use_kmeanspp_
                     ? init_kmeanspp(X, k_eff, rng)
                     : init_random(X, k_eff, rng);

      Idxs assign(m, 0);

      for (std::size_t it = 0; it < max_iters_; ++it)
      {
        bool changed = false;

        // ---- E-step: assign samples to nearest centre ----
        for (std::size_t i = 0; i < m; ++i)
        {
          const std::size_t best = nearest_center(X[i]);
          if (assign[i] != best)
          {
            assign[i] = best;
            changed = true;
          }
        }

        if (!changed)
          break; // Converged

        // ---- M-step: recompute centres ----
        Mat new_centers(k_eff, Vec(d, 0.0));
        Vec counts(k_eff, 0.0);

        for (std::size_t i = 0; i < m; ++i)
        {
          const std::size_t c = assign[i];
          counts[c] += 1.0;
          for (std::size_t j = 0; j < d; ++j)
            new_centers[c][j] += X[i][j];
        }

        for (std::size_t c = 0; c < k_eff; ++c)
        {
          if (counts[c] > 0.0)
          {
            for (std::size_t j = 0; j < d; ++j)
              new_centers[c][j] /= counts[c];
          }
          else
          {
            // Empty cluster — keep the previous centre
            new_centers[c] = centers_[c];
          }
        }

        centers_ = std::move(new_centers);
      }
    }

    /**
     * @brief Return the index of the nearest cluster centre for @p x.
     * @param x Feature vector (n_features).
     */
    std::size_t predict_label(const Vec &x) const
    {
      return nearest_center(x);
    }

    /**
     * @brief Model-API wrapper — returns cluster index as `double`.
     * @param x Feature vector (n_features).
     */
    double predict_one(const Vec &x) const override
    {
      return static_cast<double>(predict_label(x));
    }

    /**
     * @brief Predict cluster labels for every row of @p X.
     * @param X Feature matrix (n_samples × n_features).
     * @return Index vector (n_samples).
     */
    Idxs predict_labels(const Mat &X) const
    {
      Idxs out;
      out.reserve(nrows(X));
      for (const auto &row : X)
        out.push_back(predict_label(row));
      return out;
    }

    /**
     * @brief Batch predict via Model API — returns cluster IDs as doubles.
     * @param X Feature matrix (n_samples × n_features).
     * @return Vec of cluster IDs.
     */
    Vec predict(const Mat &X) const override
    {
      Vec out;
      out.reserve(nrows(X));
      for (const auto &row : X)
        out.push_back(predict_one(row));
      return out;
    }

    /**
     * @brief Compute within-cluster sum of squares (inertia).
     *
     * @code
     *   inertia = Σᵢ ‖xᵢ − μ_{aᵢ}‖²
     * @endcode
     *
     * Lower inertia indicates tighter clusters.  Use this to evaluate
     * different values of @p k (elbow method).
     *
     * @param X Feature matrix (n_samples × n_features).
     * @return Total WCSS over all samples.
     * @throws std::invalid_argument if centres have not been fitted yet.
     */
    double inertia(const Mat &X) const
    {
      if (centers_.empty())
        throw std::invalid_argument(
            "KMeans::inertia: model has not been fitted.");

      double total = 0.0;
      for (const auto &row : X)
        total += sqdist(row, centers_[nearest_center(row)]);
      return total;
    }

    /**
     * @brief Write cluster centres to a text stream.
     *
     * Format:
     * @code
     * <k> <n_features>
     * <c[0][0]> <c[0][1]> ...
     * ...
     * <c[k-1][0]> ...
     * @endcode
     */
    void save(std::ostream &os) const override
    {
      const std::size_t k = centers_.size();
      const std::size_t d = k > 0 ? centers_[0].size() : 0;
      os << std::setprecision(17) << k << " " << d << "\n";
      for (const auto &c : centers_)
      {
        for (double v : c)
          os << v << " ";
        os << "\n";
      }
    }

    /**
     * @brief Restore cluster centres from a text stream written by `save`.
     */
    void load(std::istream &is) override
    {
      std::size_t k = 0, d = 0;
      is >> k >> d;
      centers_.assign(k, Vec(d, 0.0));
      for (std::size_t c = 0; c < k; ++c)
        for (std::size_t j = 0; j < d; ++j)
          is >> centers_[c][j];
      k_ = k;
    }

    /// Cluster centroid matrix (k × n_features).
    const Mat &centers() const noexcept { return centers_; }
    /// Number of clusters this model was constructed with.
    std::size_t k() const noexcept { return k_; }

  private:
    /// Pick @p k distinct rows uniformly at random.
    static Mat init_random(const Mat &X, std::size_t k, std::mt19937 &rng)
    {
      const std::size_t m = nrows(X);
      // Sample k distinct indices
      std::vector<std::size_t> pool(m);
      std::iota(pool.begin(), pool.end(), std::size_t{0});
      std::shuffle(pool.begin(), pool.end(), rng);

      Mat centers(k);
      for (std::size_t c = 0; c < k; ++c)
        centers[c] = X[pool[c]];
      return centers;
    }

    /**
     * @brief K-Means++ seeding.
     *
     * 1. Choose the first centre uniformly at random.
     * 2. For each subsequent centre, choose sample @p x with probability
     *    proportional to D²(x) = min squared distance to any already-chosen
     *    centre.
     *
     * This gives an O(log k) approximation guarantee on the expected inertia
     * compared to the optimal clustering.
     */
    static Mat init_kmeanspp(const Mat &X, std::size_t k, std::mt19937 &rng)
    {
      const std::size_t m = nrows(X);
      const std::size_t d = ncols(X);

      Mat centers;
      centers.reserve(k);

      // First centre: uniform random
      {
        std::uniform_int_distribution<std::size_t> dist(0, m - 1);
        centers.push_back(X[dist(rng)]);
      }

      // Squared distances from each sample to its nearest chosen centre
      Vec D2(m, std::numeric_limits<double>::infinity());

      for (std::size_t c = 1; c < k; ++c)
      {
        // Update D² with the most recently added centre
        const Vec &last = centers.back();
        for (std::size_t i = 0; i < m; ++i)
        {
          const double d2 = sqdist(X[i], last);
          if (d2 < D2[i])
            D2[i] = d2;
        }

        // Weighted random selection proportional to D²
        std::discrete_distribution<std::size_t> wdist(D2.begin(), D2.end());
        centers.push_back(X[wdist(rng)]);
      }

      return centers;
    }

    /// Squared Euclidean distance between two equal-length vectors.
    static double sqdist(const Vec &a, const Vec &b)
    {
      double s = 0.0;
      for (std::size_t j = 0; j < a.size(); ++j)
      {
        const double z = a[j] - b[j];
        s += z * z;
      }
      return s;
    }

    /// Return the index of the centre nearest to @p x.
    std::size_t nearest_center(const Vec &x) const
    {
      std::size_t best = 0;
      double bestd = std::numeric_limits<double>::infinity();
      for (std::size_t c = 0; c < centers_.size(); ++c)
      {
        const double d2 = sqdist(x, centers_[c]);
        if (d2 < bestd)
        {
          bestd = d2;
          best = c;
        }
      }
      return best;
    }

    std::size_t k_;
    std::size_t max_iters_;
    unsigned seed_;
    bool use_kmeanspp_;
    Mat centers_;
  };

  /**
   * @brief K-Means with mini-batch stochastic centroid updates.
   *
   * Instead of using the entire dataset in each M-step, a random subset
   * (mini-batch) of @p batch_size samples is drawn each iteration.  This gives
   * sub-linear convergence in exchange for a significant speed-up on large
   * datasets.
   *
   * ### Algorithm (per iteration)
   * 1. Draw a mini-batch **B** of `batch_size` samples uniformly at random.
   * 2. **E-step**: assign each sample in **B** to its nearest centre.
   * 3. **M-step (online)**: for each assigned centre, update it with a running
   *    mean weighted by how many samples have been assigned to it in total:
   *    @code
   *      count[c]  += 1
   *      eta        = 1 / count[c]          // learning rate decays as 1/n
   *      center[c] += eta * (x - center[c]) // incremental mean
   *    @endcode
   *
   * Because the learning rate decays as 1/count, centres converge even though
   * each update only sees a fraction of the data.
   *
   * ### Usage
   * @code
   * MiniBatchKMeans mb(5, 256, 200, 42);
   * mb.fit(X);
   * auto labels = mb.predict_labels(X);
   * @endcode
   */
  class MiniBatchKMeans final : public Model
  {
  public:
    /**
     * @param k          Number of clusters (must be ≥ 1).
     * @param batch_size Number of samples per mini-batch (0 = full batch,
     *                   behaving like standard KMeans with random init).
     * @param max_iters  Maximum mini-batch iterations (default 100).
     * @param seed       RNG seed (default 42).
     */
    explicit MiniBatchKMeans(std::size_t k = 2,
                             std::size_t batch_size = 256,
                             std::size_t max_iters = 100,
                             unsigned seed = 42)
        : k_(k), batch_size_(batch_size), max_iters_(max_iters), seed_(seed)
    {
    }

    /**
     * @brief Fit cluster centres using mini-batch stochastic updates.
     *
     * @param X Feature matrix (n_samples × n_features).
     * @throws std::invalid_argument if @p X is empty.
     */
    void fit(const Mat &X) override
    {
      const std::size_t m = nrows(X);
      const std::size_t d = ncols(X);

      if (m == 0 || d == 0)
        throw std::invalid_argument(
            "MiniBatchKMeans::fit: X must not be empty.");
      if (k_ == 0)
        throw std::invalid_argument(
            "MiniBatchKMeans::fit: k must be >= 1.");

      const std::size_t k_eff = std::min(k_, m);
      const std::size_t B = (batch_size_ == 0 || batch_size_ > m)
                                ? m
                                : batch_size_;

      std::mt19937 rng(seed_);

      // Initialise centres by picking k_eff distinct samples at random
      centers_ = init_random(X, k_eff, rng);

      // Per-centre running sample count (for online mean updates)
      Vec counts(k_eff, 0.0);

      std::vector<std::size_t> pool(m);
      std::iota(pool.begin(), pool.end(), std::size_t{0});

      for (std::size_t it = 0; it < max_iters_; ++it)
      {
        // Draw a mini-batch without replacement
        std::shuffle(pool.begin(), pool.end(), rng);

        for (std::size_t bi = 0; bi < B; ++bi)
        {
          const Vec &x = X[pool[bi]];

          // E-step: nearest centre
          const std::size_t c = nearest_center(x);

          // M-step: incremental mean update  η = 1 / ++count[c]
          counts[c] += 1.0;
          const double eta = 1.0 / counts[c];
          for (std::size_t j = 0; j < d; ++j)
            centers_[c][j] += eta * (x[j] - centers_[c][j]);
        }
      }
    }

    /// Return the nearest cluster index for @p x.
    std::size_t predict_label(const Vec &x) const
    {
      return nearest_center(x);
    }

    /// Model-API wrapper — cluster index as `double`.
    double predict_one(const Vec &x) const override
    {
      return static_cast<double>(predict_label(x));
    }

    /// Batch cluster assignment.
    Idxs predict_labels(const Mat &X) const
    {
      Idxs out;
      out.reserve(nrows(X));
      for (const auto &row : X)
        out.push_back(predict_label(row));
      return out;
    }

    /// Batch predict via Model API.
    Vec predict(const Mat &X) const override
    {
      Vec out;
      out.reserve(nrows(X));
      for (const auto &row : X)
        out.push_back(predict_one(row));
      return out;
    }

    /**
     * @brief Within-cluster sum of squares.
     * @param X Feature matrix.
     * @return Total WCSS.
     * @throws std::invalid_argument if the model has not been fitted.
     */
    double inertia(const Mat &X) const
    {
      if (centers_.empty())
        throw std::invalid_argument(
            "MiniBatchKMeans::inertia: model has not been fitted.");

      double total = 0.0;
      for (const auto &row : X)
        total += sqdist(row, centers_[nearest_center(row)]);
      return total;
    }

    /**
     * @brief Serialise centres to a text stream (same format as KMeans).
     */
    void save(std::ostream &os) const override
    {
      const std::size_t k = centers_.size();
      const std::size_t d = k > 0 ? centers_[0].size() : 0;
      os << std::setprecision(17) << k << " " << d << "\n";
      for (const auto &c : centers_)
      {
        for (double v : c)
          os << v << " ";
        os << "\n";
      }
    }

    /** @brief Restore centres from a text stream written by `save`. */
    void load(std::istream &is) override
    {
      std::size_t k = 0, d = 0;
      is >> k >> d;
      centers_.assign(k, Vec(d, 0.0));
      for (std::size_t c = 0; c < k; ++c)
        for (std::size_t j = 0; j < d; ++j)
          is >> centers_[c][j];
      k_ = k;
    }

    const Mat &centers() const noexcept { return centers_; }
    std::size_t k() const noexcept { return k_; }

  private:
    static Mat init_random(const Mat &X, std::size_t k, std::mt19937 &rng)
    {
      const std::size_t m = nrows(X);
      std::vector<std::size_t> pool(m);
      std::iota(pool.begin(), pool.end(), std::size_t{0});
      std::shuffle(pool.begin(), pool.end(), rng);

      Mat centers(k);
      for (std::size_t c = 0; c < k; ++c)
        centers[c] = X[pool[c]];
      return centers;
    }

    static double sqdist(const Vec &a, const Vec &b)
    {
      double s = 0.0;
      for (std::size_t j = 0; j < a.size(); ++j)
      {
        const double z = a[j] - b[j];
        s += z * z;
      }
      return s;
    }

    std::size_t nearest_center(const Vec &x) const
    {
      std::size_t best = 0;
      double bestd = std::numeric_limits<double>::infinity();
      for (std::size_t c = 0; c < centers_.size(); ++c)
      {
        const double d2 = sqdist(x, centers_[c]);
        if (d2 < bestd)
        {
          bestd = d2;
          best = c;
        }
      }
      return best;
    }

    std::size_t k_;
    std::size_t batch_size_;
    std::size_t max_iters_;
    unsigned seed_;
    Mat centers_;
  };

} // namespace vix::ai::ml

#endif // VIX_AI_ML_CLUSTERING_HPP
