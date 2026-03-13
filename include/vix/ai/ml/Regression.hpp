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
#include <vector>
#include <random>
#include <limits>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace vix::ai::ml
{

  class LinearRegression final : public Model
  {
  public:
    LinearRegression() = default;

    // Convenience 1D : y ~= a*x + b
    LinearRegression(double a, double b)
    {
      w_.assign(1, a);
      b_ = b;
    }

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

    void fit(const Mat &X, const Vec &y) override
    {
      const std::size_t m = nrows(X);
      const std::size_t d = ncols(X);
      if (m == 0 || d == 0 || y.size() != m)
        return;

      w_.assign(d, 0.0);
      b_ = 0.0;

      std::vector<std::size_t> idx(m);
      for (std::size_t i = 0; i < m; ++i)
        idx[i] = i;

      std::mt19937 rng(42);

      const double lr = learning_rate_;
      const std::size_t iters = max_iters_;
      const std::size_t B = (batch_size_ == 0 || batch_size_ > m) ? m : batch_size_;

      // Early stopping
      double best_loss = std::numeric_limits<double>::infinity();
      std::size_t bad_rounds = 0;
      Mat batchX;
      Vec batchY;

      for (std::size_t it = 0; it < iters; ++it)
      {
        if (shuffle_)
          std::shuffle(idx.begin(), idx.end(), rng);

        for (std::size_t start = 0; start < m; start += B)
        {
          const std::size_t end = std::min(start + B, m);
          const std::size_t curB = end - start;

          batchX.assign(curB, Vec(d, 0.0));
          batchY.assign(curB, 0.0);
          for (std::size_t r = 0; r < curB; ++r)
          {
            const auto i = idx[start + r];
            batchX[r] = X[i];
            batchY[r] = y[i];
          }

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

          if (l2_ > 0.0)
          {
            for (std::size_t j = 0; j < d; ++j)
              gw[j] += l2_ * w_[j] * static_cast<double>(curB);
          }

          for (std::size_t j = 0; j < d; ++j)
            w_[j] -= lr * (gw[j] * invB);
          b_ -= lr * (gb * invB);
        }

        // Early stopping : (MSE + L2)
        const double cur_loss = loss_with_l2(X, y);
        if (verbose_every_ && (it % verbose_every_ == 0))
        {
          std::cout << "[it=" << it << "] loss=" << cur_loss << "\n";
        }

        const double gain = best_loss - cur_loss;
        if (gain > tol_)
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

    // -----------------------------
    // Closed-form solution (Normal Equation) with Ridge
    // -----------------------------
    // Solves argmin ||Xw + b - y||^2 + l2*||w||^2
    // Implementation: append a column of 1s to X (for the bias),
    // then solve (Z^T Z + Lambda) theta = Z^T y,
    // where Lambda = diag(l2, ..., l2, 0) — no L2 regularization on the bias.
    void fit_closed_form(const Mat &X, const Vec &y, double l2 = 0.0)
    {
      const std::size_t m = nrows(X);
      const std::size_t d = ncols(X);
      if (m == 0 || d == 0 || y.size() != m)
        return;

      // Z = [X | 1]
      Mat Z(m, Vec(d + 1, 0.0));
      for (std::size_t i = 0; i < m; ++i)
      {
        for (std::size_t j = 0; j < d; ++j)
          Z[i][j] = X[i][j];
        Z[i][d] = 1.0;
      }

      // A = Z^T Z, b = Z^T y
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

      if (l2 > 0.0)
      {
        for (std::size_t j = 0; j < d; ++j)
          A[j][j] += l2;
      }

      // A * theta = bvec
      Vec theta = gaussian_solve(A, bvec); // d+1

      w_.assign(d, 0.0);
      for (std::size_t j = 0; j < d; ++j)
        w_[j] = theta[j];
      b_ = theta[d];
    }

    double predict_one(const Vec &x) const override { return dot(x, w_) + b_; }
    double predict_scalar(double x) const { return predict_one(Vec{x}); }

    void save(std::ostream &os) const override
    {
      os << std::setprecision(17) << w_.size() << " " << b_ << "\n";
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
      const std::size_t d = a.size();
      for (std::size_t j = 0; j < d; ++j)
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

    static Vec gaussian_solve(Mat A, Vec b)
    {
      const std::size_t n = A.size();
      // [A|b]
      for (std::size_t i = 0; i < n; ++i)
        A[i].push_back(b[i]);

      for (std::size_t col = 0; col < n; ++col)
      {
        std::size_t piv = col;
        double best = std::abs(A[col][col]);
        for (std::size_t r = col + 1; r < n; ++r)
        {
          double v = std::abs(A[r][col]);
          if (v > best)
          {
            best = v;
            piv = r;
          }
        }
        if (best == 0.0)
          continue;

        if (piv != col)
          std::swap(A[piv], A[col]);

        const double diag = A[col][col];
        for (std::size_t c = col; c <= n; ++c)
          A[col][c] /= diag;

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

  class LogisticRegression final : public Model
  {
  public:
    void fit(const Mat &X, const Vec &y) override
    {
      (void)X;
      (void)y;
    }
    double predict_one(const Vec &x) const override
    {
      (void)x;
      return 0.5;
    }
  };

} // namespace vix::ai::ml

#endif
