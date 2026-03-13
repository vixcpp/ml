/**
 * @file Model.hpp
 * @author Gaspard Kirira
 *
 * Copyright 2025, Gaspard Kirira. All rights reserved.
 * https://github.com/vixcpp/vix
 * Use of this source code is governed by a MIT license that can be found in the License file.
 *
 * Vix.cpp
 */
#ifndef VIX_AI_ML_MODEL_HPP
#define VIX_AI_ML_MODEL_HPP

#include <vix/ai/ml/Types.hpp>
#include <istream>
#include <ostream>

namespace vix::ai::ml
{
  /**
   * @file Model.hpp
   * @brief Base abstract class for all machine learning models in the Vix.AI.ML module.
   *
   * This class defines a unified interface for supervised and unsupervised learning algorithms.
   * It provides virtual methods for training (`fit`), inference (`predict`, `predict_one`),
   * and simple serialization (`save`/`load`).
   *
   * All derived models (e.g. LinearRegression, LogisticRegression, KMeans, PCA)
   * should inherit from this class and implement at least `fit()` and `predict_one()`.
   *
   * ### Design goals:
   * - Minimal and extensible API
   * - Works uniformly for both supervised and unsupervised models
   * - Allows generic evaluation and composition (e.g., pipelines)
   *
   * ### Example:
   * @code
   * std::unique_ptr<Model> model = std::make_unique<LinearRegression>();
   * model->fit(X_train, y_train);
   * auto y_pred = model->predict(X_test);
   * @endcode
   */
  class Model
  {
  public:
    virtual ~Model() = default;

    /**
     * @brief Train the model on supervised data (X, y).
     * @param X Feature matrix (n_samples × n_features).
     * @param y Target vector (n_samples).
     *
     * Default implementation is a no-op.
     * Derived classes (e.g. regression, classification) should override it.
     */
    virtual void fit(const Mat &X, const Vec &y)
    {
      (void)X;
      (void)y;
    }

    /**
     * @brief Train the model on unsupervised data (X).
     * @param X Feature matrix (n_samples × n_features).
     *
     * Default implementation is a no-op.
     * Used by algorithms like KMeans or PCA.
     */
    virtual void fit(const Mat &X)
    {
      (void)X;
    }

    /**
     * @brief Predict the output for a single input vector.
     * @param x Input feature vector.
     * @return Predicted scalar value (e.g., regression output or cluster ID).
     *
     * Default implementation returns 0.0.
     * Derived models should override this with the actual prediction logic.
     */
    virtual double predict_one(const Vec &x) const
    {
      (void)x;
      return 0.0;
    }

    /**
     * @brief Predict outputs for a batch of input samples.
     * @param X Input feature matrix (n_samples × n_features).
     * @return Vector of predictions (n_samples).
     *
     * This method calls `predict_one()` for each row of X.
     * Derived classes can override for vectorized implementations.
     */
    virtual Vec predict(const Mat &X) const
    {
      Vec out;
      out.reserve(nrows(X));
      for (const auto &row : X)
        out.push_back(predict_one(row));
      return out;
    }

    /**
     * @brief Save model parameters to a text stream.
     * @param os Output stream.
     *
     * Default implementation does nothing.
     * Derived models (e.g. LinearRegression) should override this.
     */
    virtual void save(std::ostream &os) const
    {
      (void)os;
    }

    /**
     * @brief Load model parameters from a text stream.
     * @param is Input stream.
     *
     * Default implementation does nothing.
     * Derived models (e.g. LinearRegression) should override this.
     */
    virtual void load(std::istream &is)
    {
      (void)is;
    }
  };

} // namespace vix::ai::ml

#endif
