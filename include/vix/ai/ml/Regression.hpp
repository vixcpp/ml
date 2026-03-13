/**
 * @file Regression.hpp
 * @author Gaspard Kirira
 *
 * Copyright 2025, Gaspard Kirira. All rights reserved.
 * https://github.com/vixcpp/vix
 * Use of this source code is governed by a MIT license that can be found in the License file.
 *
 * Vix.cpp
 */
#ifndef VIX_AI_ML_REGRESSION_HPP
#define VIX_AI_ML_REGRESSION_HPP

#include <vix/ai/ml/Model.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace vix::ai::ml
{
  /**
   * @brief Ordinary least-squares linear regression with optional L2
   *        regularisation (Ridge).
   *
   * Two training modes are available:
   * - **Gradient descent** (`fit`): supports mini-batches, early stopping, and
   *   L2 regularisation.  Suitable for large datasets.
   * - **Closed-form / Normal Equation** (`fit_closed_form`): exact solution via
   *   Gaussian elimination.  Fast for small-to-medium datasets.
   *
   * Model: ŷ = w · x + b
   *
   * ### Example
   * @code
   * LinearRegression lr;
   * lr.set_hyperparams(0.01, 1000, 32, 1e-4);
   * lr.fit(X_train, y_train);
   * Vec preds = lr.predict(X_test);
   * @endcode
   */
  class LinearRegression final : public Model
  {
  public:
    LinearRegression() = default;

    /// Convenience constructor for 1-D problems: ŷ = a·x + b.
    LinearRegression(double a, double b)
    {
      w_.assign(1, a);
      b_ = b;
    }

    /**
     * @brief Set all gradient-descent hyperparameters in one call.
     *
     * @param lr            Learning rate (step size).
     * @param iters         Maximum number of full passes over the dataset.
     * @param batch_size    Mini-batch size (0 = full-batch gradient descent).
     * @param l2            L2 (Ridge) regularisation coefficient.
     * @param shuffle       Shuffle sample order each epoch when true.
     * @param tol           Minimum loss improvement to reset the patience counter.
     * @param patience      Stop training after this many non-improving epochs.
     * @param verbose_every Print loss every N iterations (0 = silent).
     */
    void set_hyperparams(
        double lr,
        std::size_t iters,
        std::size_t batch_size = 0,
        double l2 = 0.0,
        bool shuffle = true,
        double tol = 1e-8,
        std::size_t patience = 20,
        std::size_t verbose_every = 0)
    {
      learning_rate_ = lr;
      max_iters_ = iters;
      batch_size_ = batch_size;
      l2_ = l2;
      shuffle_ = shuffle;
      tol_ = tol;
      patience_ = patience;
      verbose_every_ = verbose_every;
    }

    /**
     * @brief Train via mini-batch gradient descent.
     *
     * Minimises MSE + optional L2 penalty:
     *   L = (1/m) Σ (ŷᵢ - yᵢ)² + λ‖w‖²
     *
     * @param X Feature matrix (n_samples × n_features).
     * @param y Target vector  (n_samples).
     */
    void fit(const Mat &X, const Vec &y) override
    {
      validate_supervised(X, y, "LinearRegression::fit");

      const std::size_t m = nrows(X);
      const std::size_t d = ncols(X);

      w_.assign(d, 0.0);
      b_ = 0.0;

      std::vector<std::size_t> idx(m);
      std::iota(idx.begin(), idx.end(), std::size_t{0});

      std::mt19937 rng(42);

      const std::size_t B =
          (batch_size_ == 0 || batch_size_ > m) ? m : batch_size_;

      double best_loss = std::numeric_limits<double>::infinity();
      std::size_t bad_rounds = 0;

      Mat batchX;
      Vec batchY;

      for (std::size_t it = 0; it < max_iters_; ++it)
      {
        if (shuffle_)
          std::shuffle(idx.begin(), idx.end(), rng);

        // ---- mini-batch loop ----
        for (std::size_t start = 0; start < m; start += B)
        {
          const std::size_t end = std::min(start + B, m);
          const std::size_t curB = end - start;

          // Gather batch
          batchX.assign(curB, Vec(d, 0.0));
          batchY.assign(curB, 0.0);
          for (std::size_t r = 0; r < curB; ++r)
          {
            const auto i = idx[start + r];
            batchX[r] = X[i];
            batchY[r] = y[i];
          }

          // Accumulate gradients
          Vec gw(d, 0.0);
          double gb = 0.0;

          for (std::size_t r = 0; r < curB; ++r)
          {
            const double yhat = dot(batchX[r], w_) + b_;
            const double err = yhat - batchY[r];
            for (std::size_t j = 0; j < d; ++j)
              gw[j] += err * batchX[r][j];
            gb += err;
          }

          const double invB = 1.0 / static_cast<double>(curB);

          // L2 gradient contribution (scaled back to per-sample)
          if (l2_ > 0.0)
          {
            for (std::size_t j = 0; j < d; ++j)
              gw[j] += l2_ * w_[j] * static_cast<double>(curB);
          }

          // Parameter update
          for (std::size_t j = 0; j < d; ++j)
            w_[j] -= learning_rate_ * (gw[j] * invB);
          b_ -= learning_rate_ * (gb * invB);
        }

        // ---- early stopping ----
        const double cur_loss = loss_with_l2(X, y);

        if (verbose_every_ && (it % verbose_every_ == 0))
          std::cout << "[LinearRegression it=" << it
                    << "] loss=" << cur_loss << "\n";

        if (best_loss - cur_loss > tol_)
        {
          best_loss = cur_loss;
          bad_rounds = 0;
        }
        else
        {
          if (++bad_rounds >= patience_)
            break;
        }
      }
    }

    /**
     * @brief Exact solution via the Normal Equation with optional Ridge penalty.
     *
     * Solves:  argmin ‖Xw + b - y‖² + λ‖w‖²
     *
     * Augments X with a bias column of ones, forms Z^T Z + Λ, then solves the
     * linear system via Gaussian elimination with partial pivoting.
     *
     * @param X   Feature matrix (n_samples × n_features).
     * @param y   Target vector  (n_samples).
     * @param l2  Ridge coefficient (0 = ordinary least squares).
     */
    void fit_closed_form(const Mat &X, const Vec &y, double l2 = 0.0)
    {
      validate_supervised(X, y, "LinearRegression::fit_closed_form");

      const std::size_t m = nrows(X);
      const std::size_t d = ncols(X);

      // Z = [X | 1]  (augment with bias column)
      Mat Z(m, Vec(d + 1, 0.0));
      for (std::size_t i = 0; i < m; ++i)
      {
        for (std::size_t j = 0; j < d; ++j)
          Z[i][j] = X[i][j];
        Z[i][d] = 1.0;
      }

      // A = Z^T Z,  bvec = Z^T y
      Mat A(d + 1, Vec(d + 1, 0.0));
      Vec bvec(d + 1, 0.0);

      for (std::size_t i = 0; i < d + 1; ++i)
      {
        for (std::size_t j = i; j < d + 1; ++j)
        {
          double s = 0.0;
          for (std::size_t r = 0; r < m; ++r)
            s += Z[r][i] * Z[r][j];
          A[i][j] = A[j][i] = s;
        }
        double t = 0.0;
        for (std::size_t r = 0; r < m; ++r)
          t += Z[r][i] * y[r];
        bvec[i] = t;
      }

      // Ridge: add λ to diagonal of weight block (not bias)
      if (l2 > 0.0)
      {
        for (std::size_t j = 0; j < d; ++j)
          A[j][j] += l2;
      }

      // Solve (A * θ = bvec) for θ = [w; b]
      Vec theta = gaussian_solve(A, bvec);

      w_.assign(d, 0.0);
      for (std::size_t j = 0; j < d; ++j)
        w_[j] = theta[j];
      b_ = theta[d];
    }

    /// Returns ŷ = w · x + b.
    double predict_one(const Vec &x) const override
    {
      return dot(x, w_) + b_;
    }

    /// Convenience wrapper for scalar (1-D) inputs.
    double predict_scalar(double x) const
    {
      return predict_one(Vec{x});
    }

    void save(std::ostream &os) const override
    {
      os << std::setprecision(17)
         << w_.size() << " " << b_ << "\n";
      for (double wi : w_)
        os << wi << " ";
      os << "\n";
    }

    void load(std::istream &is) override
    {
      std::size_t d = 0;
      is >> d >> b_;
      w_.assign(d, 0.0);
      for (std::size_t j = 0; j < d; ++j)
        is >> w_[j];
    }

    const Vec &weights() const noexcept { return w_; }
    double bias() const noexcept { return b_; }
    double l2() const noexcept { return l2_; }

  private:
    static double dot(const Vec &a, const Vec &b)
    {
      double s = 0.0;
      for (std::size_t j = 0; j < a.size(); ++j)
        s += a[j] * b[j];
      return s;
    }

    double mse_only(const Mat &X, const Vec &y) const
    {
      const std::size_t m = nrows(X);
      if (m == 0)
        return 0.0;
      double s = 0.0;
      for (std::size_t i = 0; i < m; ++i)
      {
        const double e = (dot(X[i], w_) + b_) - y[i];
        s += e * e;
      }
      return s / static_cast<double>(m);
    }

    double loss_with_l2(const Mat &X, const Vec &y) const
    {
      double loss = mse_only(X, y);
      if (l2_ > 0.0)
      {
        double norm2 = 0.0;
        for (double wi : w_)
          norm2 += wi * wi;
        loss += l2_ * norm2;
      }
      return loss;
    }

    /// Gauss-Jordan elimination with partial pivoting.  Solves A·x = b in-place.
    static Vec gaussian_solve(Mat A, Vec b)
    {
      const std::size_t n = A.size();

      // Augment: [A | b]
      for (std::size_t i = 0; i < n; ++i)
        A[i].push_back(b[i]);

      for (std::size_t col = 0; col < n; ++col)
      {
        // Partial pivoting
        std::size_t piv = col;
        double best = std::abs(A[col][col]);
        for (std::size_t r = col + 1; r < n; ++r)
        {
          const double v = std::abs(A[r][col]);
          if (v > best)
          {
            best = v;
            piv = r;
          }
        }
        if (best == 0.0)
          continue; // singular column — skip

        if (piv != col)
          std::swap(A[piv], A[col]);

        // Normalise pivot row
        const double diag = A[col][col];
        for (std::size_t c = col; c <= n; ++c)
          A[col][c] /= diag;

        // Eliminate column from all other rows
        for (std::size_t r = 0; r < n; ++r)
        {
          if (r == col)
            continue;
          const double f = A[r][col];
          if (f == 0.0)
            continue;
          for (std::size_t c = col; c <= n; ++c)
            A[r][c] -= f * A[col][c];
        }
      }

      Vec x(n, 0.0);
      for (std::size_t i = 0; i < n; ++i)
        x[i] = A[i][n];
      return x;
    }

    Vec w_;
    double b_{0.0};

    double learning_rate_{0.05};
    std::size_t max_iters_{2000};
    std::size_t batch_size_{0};
    double l2_{0.0};
    bool shuffle_{true};
    double tol_{1e-8};
    std::size_t patience_{20};
    std::size_t verbose_every_{0};
  };

  /**
   * @brief Binary logistic regression with mini-batch gradient descent and
   *        optional L2 regularisation.
   *
   * Models the conditional probability of the positive class:
   *   p(y=1 | x) = σ(w · x + b),    σ(z) = 1 / (1 + exp(−z))
   *
   * Training minimises binary cross-entropy with optional Ridge penalty:
   *   L = −(1/m) Σ [ yᵢ log(pᵢ) + (1−yᵢ) log(1−pᵢ) ] + λ‖w‖²
   *
   * Supports the same hyperparameters and early-stopping strategy as
   * LinearRegression for consistency across the module.
   *
   * ### Prediction
   * - `predict_one(x)` returns the **probability** p ∈ (0, 1).
   * - `predict_class(x)` returns the **hard label** (0 or 1, threshold = 0.5).
   * - `predict(X)` returns a vector of **probabilities**.
   *
   * ### Example
   * @code
   * LogisticRegression clf;
   * clf.set_hyperparams(0.1, 500, 64, 1e-3);
   * clf.fit(X_train, y_train);
   * Vec probs  = clf.predict(X_test);
   * int label  = clf.predict_class(x_new);
   * @endcode
   */
  class LogisticRegression final : public Model
  {
  public:
    LogisticRegression() = default;

    /**
     * @brief Set all gradient-descent hyperparameters in one call.
     *
     * @param lr            Learning rate (step size).
     * @param iters         Maximum number of full passes over the dataset.
     * @param batch_size    Mini-batch size (0 = full-batch gradient descent).
     * @param l2            L2 (Ridge) regularisation coefficient.
     * @param shuffle       Shuffle sample order each epoch when true.
     * @param tol           Minimum loss improvement to reset the patience counter.
     * @param patience      Stop training after this many non-improving epochs.
     * @param verbose_every Print loss every N iterations (0 = silent).
     */
    void set_hyperparams(
        double lr,
        std::size_t iters,
        std::size_t batch_size = 0,
        double l2 = 0.0,
        bool shuffle = true,
        double tol = 1e-8,
        std::size_t patience = 20,
        std::size_t verbose_every = 0)
    {
      learning_rate_ = lr;
      max_iters_ = iters;
      batch_size_ = batch_size;
      l2_ = l2;
      shuffle_ = shuffle;
      tol_ = tol;
      patience_ = patience;
      verbose_every_ = verbose_every;
    }

    /**
     * @brief Train via mini-batch gradient descent on binary cross-entropy.
     *
     * Gradient derivation per sample:
     *   z   = w · x + b
     *   p   = σ(z)
     *   err = p − y         (error signal)
     *   ∂L/∂wⱼ = err · xⱼ  (+λwⱼ for L2)
     *   ∂L/∂b  = err
     *
     * @param X Feature matrix (n_samples × n_features).
     * @param y Binary label vector with values in {0, 1} (n_samples).
     */
    void fit(const Mat &X, const Vec &y) override
    {
      validate_supervised(X, y, "LogisticRegression::fit");

      const std::size_t m = nrows(X);
      const std::size_t d = ncols(X);

      // Initialise parameters to zero
      w_.assign(d, 0.0);
      b_ = 0.0;

      std::vector<std::size_t> idx(m);
      std::iota(idx.begin(), idx.end(), std::size_t{0});

      std::mt19937 rng(42);

      const std::size_t B =
          (batch_size_ == 0 || batch_size_ > m) ? m : batch_size_;

      double best_loss = std::numeric_limits<double>::infinity();
      std::size_t bad_rounds = 0;

      Mat batchX;
      Vec batchY;

      for (std::size_t it = 0; it < max_iters_; ++it)
      {
        if (shuffle_)
          std::shuffle(idx.begin(), idx.end(), rng);

        // ---- mini-batch loop ----
        for (std::size_t start = 0; start < m; start += B)
        {
          const std::size_t end = std::min(start + B, m);
          const std::size_t curB = end - start;

          // Gather batch
          batchX.assign(curB, Vec(d, 0.0));
          batchY.assign(curB, 0.0);
          for (std::size_t r = 0; r < curB; ++r)
          {
            const auto i = idx[start + r];
            batchX[r] = X[i];
            batchY[r] = y[i];
          }

          // Accumulate gradients: ∂L/∂w, ∂L/∂b
          Vec gw(d, 0.0);
          double gb = 0.0;

          for (std::size_t r = 0; r < curB; ++r)
          {
            const double z = dot(batchX[r], w_) + b_;
            const double p = sigmoid(z);
            const double err = p - batchY[r]; // ∂L/∂z = p − y

            for (std::size_t j = 0; j < d; ++j)
              gw[j] += err * batchX[r][j];
            gb += err;
          }

          const double invB = 1.0 / static_cast<double>(curB);

          // L2 gradient contribution (ridge, not applied to bias)
          if (l2_ > 0.0)
          {
            for (std::size_t j = 0; j < d; ++j)
              gw[j] += l2_ * w_[j] * static_cast<double>(curB);
          }

          // Parameter update
          for (std::size_t j = 0; j < d; ++j)
            w_[j] -= learning_rate_ * (gw[j] * invB);
          b_ -= learning_rate_ * (gb * invB);
        }

        // ---- early stopping ----
        const double cur_loss = loss_with_l2(X, y);

        if (verbose_every_ && (it % verbose_every_ == 0))
          std::cout << "[LogisticRegression it=" << it
                    << "] loss=" << cur_loss << "\n";

        if (best_loss - cur_loss > tol_)
        {
          best_loss = cur_loss;
          bad_rounds = 0;
        }
        else
        {
          if (++bad_rounds >= patience_)
            break;
        }
      }
    }

    /**
     * @brief Return the predicted **probability** p(y=1 | x) ∈ (0, 1).
     * @param x Feature vector (n_features).
     */
    double predict_one(const Vec &x) const override
    {
      return sigmoid(dot(x, w_) + b_);
    }

    /**
     * @brief Return the predicted **hard label** (0 or 1) using threshold 0.5.
     * @param x Feature vector (n_features).
     */
    int predict_class(const Vec &x) const
    {
      return predict_one(x) >= 0.5 ? 1 : 0;
    }

    /**
     * @brief Batch-predict probabilities for every row in @p X.
     * @param X Feature matrix (n_samples × n_features).
     * @return Vector of probabilities (n_samples), each in (0, 1).
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
     * @brief Serialise parameters to a text stream.
     *
     * Format (two lines):
     * ```
     * <d> <b>
     * <w[0]> <w[1]> ... <w[d-1]>
     * ```
     */
    void save(std::ostream &os) const override
    {
      os << std::setprecision(17)
         << w_.size() << " " << b_ << "\n";
      for (double wi : w_)
        os << wi << " ";
      os << "\n";
    }

    /**
     * @brief Deserialise parameters from a text stream written by `save()`.
     */
    void load(std::istream &is) override
    {
      std::size_t d = 0;
      is >> d >> b_;
      w_.assign(d, 0.0);
      for (std::size_t j = 0; j < d; ++j)
        is >> w_[j];
    }

    const Vec &weights() const noexcept { return w_; }
    double bias() const noexcept { return b_; }
    double l2() const noexcept { return l2_; }

  private:
    static double dot(const Vec &a, const Vec &b)
    {
      double s = 0.0;
      for (std::size_t j = 0; j < a.size(); ++j)
        s += a[j] * b[j];
      return s;
    }

    /// Numerically stable sigmoid: avoids overflow for large |z|.
    static double sigmoid(double z)
    {
      // For z >= 0: σ(z) = 1 / (1 + exp(−z))
      // For z <  0: σ(z) = exp(z) / (1 + exp(z))  — avoids exp(+large)
      if (z >= 0.0)
        return 1.0 / (1.0 + std::exp(-z));
      const double ez = std::exp(z);
      return ez / (1.0 + ez);
    }

    /// Clamp probability to (ε, 1−ε) to prevent log(0) in cross-entropy.
    static double clamp_prob(double p)
    {
      constexpr double EPS = 1e-12;
      return std::max(EPS, std::min(1.0 - EPS, p));
    }

    /// Binary cross-entropy for a single predicted probability and true label.
    static double bce(double p, double y)
    {
      const double pc = clamp_prob(p);
      return -(y * std::log(pc) + (1.0 - y) * std::log(1.0 - pc));
    }

    /// Mean binary cross-entropy over the full dataset (no regularisation).
    double logloss(const Mat &X, const Vec &y) const
    {
      const std::size_t m = nrows(X);
      if (m == 0)
        return 0.0;
      double total = 0.0;
      for (std::size_t i = 0; i < m; ++i)
        total += bce(sigmoid(dot(X[i], w_) + b_), y[i]);
      return total / static_cast<double>(m);
    }

    /// Cross-entropy plus L2 penalty term: used for early-stopping comparisons.
    double loss_with_l2(const Mat &X, const Vec &y) const
    {
      double loss = logloss(X, y);
      if (l2_ > 0.0)
      {
        double norm2 = 0.0;
        for (double wi : w_)
          norm2 += wi * wi;
        loss += l2_ * norm2;
      }
      return loss;
    }

    Vec w_;
    double b_{0.0};

    double learning_rate_{0.1};
    std::size_t max_iters_{1000};
    std::size_t batch_size_{0};
    double l2_{0.0};
    bool shuffle_{true};
    double tol_{1e-8};
    std::size_t patience_{20};
    std::size_t verbose_every_{0};
  };

} // namespace vix::ai::ml

#endif // VIX_AI_ML_REGRESSION_HPP
