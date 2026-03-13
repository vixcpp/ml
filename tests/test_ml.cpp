/**
 * @file test_ml.cpp
 * @brief End-to-end smoke test for Vix.AI.ML:
 *        - LinearRegression on a noisy 1D dataset (y = 2x + 1 + noise)
 *        - KMeans on two 2D Gaussian blobs
 *
 * The test prints metrics and also performs simple assertions with tolerances.
 */

#include <vix/ai/ml/Regression.hpp>
#include <vix/ai/ml/Clustering.hpp>
#include <vix/ai/ml/Preprocessing.hpp>
#include <vix/ai/ml/Metrics.hpp>

#include <iostream>
#include <random>
#include <cmath>
#include <cstdlib>
#include <algorithm>

using namespace vix::ai::ml;

int main()
{
  // 1) Linear Regression: y = 2x + 1 + noise
  std::mt19937 rng(7);
  std::normal_distribution<double> noise(0.0, 0.2);

  Mat X;
  Vec y;
  X.reserve(200);
  y.reserve(200);
  for (int i = 0; i < 200; ++i)
  {
    const double x = i * 0.05;
    X.push_back({x});
    y.push_back(2.0 * x + 1.0 + noise(rng));
  }

  // Optional scaling (usually helps convergence for GD-based methods)
  StandardScaler scaler;
  Mat Xs = scaler.fit_transform(X);

  // GD / Mini-batch + L2 + Early-stopping
  LinearRegression lr;
  lr.set_hyperparams(
      /*lr*/ 0.05,
      /*iters*/ 3000,
      /*batch_size*/ 32, // mini-batch
      /*l2*/ 1e-4,       // ridge
      /*shuffle*/ true,
      /*tol*/ 1e-9, // early stopping tolerance
      /*patience*/ 50,
      /*verbose_every*/ 0);
  lr.fit(Xs, y);

  const Vec yhat = lr.predict(Xs);
  const double loss = mse(y, yhat);
  std::cout << "[LR:GD] MSE: " << loss << "\n";

  // Basic sanity check
  if (!(loss < 0.2))
  {
    std::cerr << "[LR:GD] MSE too high: " << loss << " (expected < 0.2)\n";
    return EXIT_FAILURE;
  }

  // Closed-form (Normal Equation) with Ridge (b not regularized)
  LinearRegression lr_cf;
  lr_cf.fit_closed_form(Xs, y, /*l2=*/1e-4);
  const Vec yhat_cf = lr_cf.predict(Xs);
  const double loss_cf = mse(y, yhat_cf);
  std::cout << "[LR:ClosedForm] MSE: " << loss_cf << "\n";

  // Closed-form should be competitive with GD
  if (!(loss_cf <= loss * 1.5))
  {
    std::cerr << "[LR:ClosedForm] Unexpectedly worse MSE: " << loss_cf
              << " vs GD " << loss << "\n";
    return EXIT_FAILURE;
  }

  // Also check the 1D convenience path
  {
    LinearRegression lr2{2.0, 1.0};           // y ≈ 2x + 1
    const double p = lr2.predict_scalar(4.0); // expect ~9
    if (std::fabs(p - 9.0) > 1e-9)
    {
      std::cerr << "[LR] predict_scalar mismatch: got " << p << " expected 9.0\n";
      return EXIT_FAILURE;
    }
  }

  // 2) KMeans: two 2D blobs
  Mat U;
  U.reserve(200);
  std::normal_distribution<double> g1x(0.0, 0.5), g1y(0.0, 0.5);
  std::normal_distribution<double> g2x(3.0, 0.5), g2y(3.0, 0.5);

  for (int i = 0; i < 100; ++i)
    U.push_back({g1x(rng), g1y(rng)});
  for (int i = 0; i < 100; ++i)
    U.push_back({g2x(rng), g2y(rng)});

  KMeans km(2, 50);
  km.fit(U);

  const auto &C = km.centers();
  std::cout << "[KMeans] centers:\n";
  for (const auto &c : C)
    std::cout << "  (" << c[0] << ", " << c[1] << ")\n";

  if (C.size() != 2 || C[0].size() != 2 || C[1].size() != 2)
  {
    std::cerr << "[KMeans] invalid centers shape\n";
    return EXIT_FAILURE;
  }

  // A point near (0,0) should be labeled 0 or 1 deterministically (depending on init)
  const double lbl = km.predict_one({0.0, 0.0});
  if (!(lbl == 0.0 || lbl == 1.0))
  {
    std::cerr << "[KMeans] label not in {0,1}: " << lbl << "\n";
    return EXIT_FAILURE;
  }
  std::cout << "[KMeans] example label @ (0,0): " << lbl << "\n";

  std::cout << "[OK] test_ml passed.\n";
  return EXIT_SUCCESS;
}
