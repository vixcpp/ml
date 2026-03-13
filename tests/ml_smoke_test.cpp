/**
 * @file ml_smoke_test.cpp
 * @brief Minimal smoke test for Vix.AI.ML:
 *        - LinearRegression (1D convenience ctor + predict_scalar)
 *        - KMeans (fit on a tiny 2D dataset)
 *        - Dataset shape helpers (X,y,U)
 */

#include <cstdlib>
#include <iostream>
#include <cmath>
#include <vix/ai/ml/Regression.hpp>
#include <vix/ai/ml/Clustering.hpp>
#include <vix/ai/ml/Dataset.hpp>

int main()
{
  using namespace vix::ai::ml;

  // LinearRegression (1D convenience ctor)
  {
    LinearRegression lr{2.0, 1.0};              // y ≈ 2x + 1
    const double pred = lr.predict_scalar(2.0); // expect ~5
    const double eps = 1e-9;
    if (std::fabs(pred - 5.0) > eps)
    {
      std::cerr << "LinearRegression predict_scalar failed: got "
                << pred << ", expected 5.0\n";
      return EXIT_FAILURE;
    }
  }

  // KMeans (need data + fit(U))
  {
    Mat U = {
        {0.0, 0.0}, {0.9, 1.1}, // cluster A
        {3.2, 3.0},
        {3.1, 2.9} // cluster B
    };
    KMeans km{2};
    km.fit(U);
    if (km.k() != 2)
    {
      std::cerr << "KMeans k() failed\n";
      return EXIT_FAILURE;
    }
  }

  // Dataset (X,y,U) + size helpers
  {
    Dataset ds;
    // supervised features (2 samples, 2 features)
    ds.X = {{0.0, 0.0}, {1.0, 1.0}};

    // targets (2)
    ds.y = {1.0, 3.0};

    // unsupervised features (2 samples)
    ds.U = {{2.0, 2.0}, {3.0, 3.0}};

    if (ds.size_supervised() != 2 || ds.size_unsupervised() != 2)
    {
      std::cerr << "Dataset size check failed (X or U)\n";
      return EXIT_FAILURE;
    }
    if (ds.n_features() != 2)
    {
      std::cerr << "Dataset n_features() failed: got "
                << ds.n_features() << ", expected 2\n";
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
