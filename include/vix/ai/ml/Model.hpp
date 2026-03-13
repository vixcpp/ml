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
#include <stdexcept>
#include <string>

namespace vix::ai::ml
{

  /**
   * @brief Abstract base class for all machine learning models in Vix.AI.ML.
   *
   * Defines a unified, scikit-learn-style interface for both supervised and
   * unsupervised learning algorithms. Concrete models (e.g. LinearRegression,
   * KMeans) inherit from this class and override the relevant virtual methods.
   *
   * ### Lifecycle
   * 1. Construct the derived model.
   * 2. Call `fit(X, y)` (supervised) or `fit(X)` (unsupervised).
   * 3. Call `predict(X)` or `predict_one(x)` for inference.
   * 4. Optionally call `save(os)` / `load(is)` for persistence.
   *
   * ### Example
   * @code
   * std::unique_ptr<Model> m = std::make_unique<LinearRegression>();
   * m->fit(X_train, y_train);
   * Vec preds = m->predict(X_test);
   * @endcode
   *
   * ### Thread safety
   * - `predict` / `predict_one` are `const` and safe to call from multiple
   *   threads simultaneously, provided no concurrent `fit` / `load` is active.
   * - `fit` and `load` mutate model state and must not overlap with other calls.
   */
  class Model
  {
  public:
    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    virtual ~Model() = default;

    // Prevent accidental slicing while keeping value semantics optional for
    // derived types that manage no resources.
    Model() = default;
    Model(const Model &) = default;
    Model(Model &&) = default;
    Model &operator=(const Model &) = default;
    Model &operator=(Model &&) = default;

    // -------------------------------------------------------------------------
    // Training interface
    // -------------------------------------------------------------------------

    /**
     * @brief Supervised training on feature matrix @p X and target vector @p y.
     *
     * The default implementation is a validated no-op: it checks that @p X is
     * non-empty, that all rows are the same width, and that
     * `y.size() == nrows(X)`, then returns without modifying any state.
     *
     * Derived supervised models (regression, classification, …) must override
     * this method to store learned parameters.
     *
     * @param X Feature matrix  (n_samples × n_features), row-major.
     * @param y Target vector   (n_samples).
     *
     * @throws std::invalid_argument if @p X is empty, rows have inconsistent
     *         widths, or `y.size() != nrows(X)`.
     */
    virtual void fit(const Mat &X, const Vec &y);

    /**
     * @brief Unsupervised training on feature matrix @p X.
     *
     * The default implementation validates @p X (non-empty, uniform row width)
     * and returns without modifying any state.
     *
     * Unsupervised models (KMeans, PCA, …) must override this method.
     *
     * @param X Feature matrix (n_samples × n_features), row-major.
     *
     * @throws std::invalid_argument if @p X is empty or rows have inconsistent
     *         widths.
     */
    virtual void fit(const Mat &X);

    // -------------------------------------------------------------------------
    // Inference interface
    // -------------------------------------------------------------------------

    /**
     * @brief Predict the output for a single feature vector @p x.
     *
     * The default implementation returns `0.0`.  Derived models must override
     * this with their actual prediction logic.
     *
     * @param x Feature vector (n_features).
     * @return Predicted scalar (e.g. regression value, class label, cluster id).
     */
    virtual double predict_one(const Vec &x) const;

    /**
     * @brief Batch-predict outputs for every row in @p X.
     *
     * The default implementation calls `predict_one()` for each row.  Derived
     * classes may override for a vectorised implementation.
     *
     * @param X Feature matrix (n_samples × n_features), row-major.
     * @return Prediction vector (n_samples).
     */
    virtual Vec predict(const Mat &X) const;

    // -------------------------------------------------------------------------
    // Serialisation interface
    // -------------------------------------------------------------------------

    /**
     * @brief Serialise learned parameters to a text stream.
     *
     * The default implementation writes nothing.  Derived models should write
     * their parameters in a format that `load()` can round-trip.
     *
     * @param os Writable output stream (file, stringstream, …).
     */
    virtual void save(std::ostream &os) const;

    /**
     * @brief Deserialise learned parameters from a text stream.
     *
     * The default implementation reads nothing.  Derived models should restore
     * exactly the state written by `save()`.
     *
     * @param is Readable input stream.
     */
    virtual void load(std::istream &is);

  protected:
    // -------------------------------------------------------------------------
    // Validation helpers (available to all derived classes)
    // -------------------------------------------------------------------------

    /**
     * @brief Assert that @p X is a valid, non-empty matrix with uniform row
     *        widths.
     *
     * @param X Matrix to validate.
     * @param context Caller description embedded in the exception message.
     *
     * @throws std::invalid_argument on any violation.
     */
    static void validate_matrix(const Mat &X,
                                const std::string &context = "Model");

    /**
     * @brief Assert that @p X is a valid matrix **and** that @p y aligns with
     *        it row-wise (`y.size() == nrows(X)`).
     *
     * Internally calls `validate_matrix(X, context)` first.
     *
     * @param X Feature matrix.
     * @param y Target vector.
     * @param context Caller description embedded in the exception message.
     *
     * @throws std::invalid_argument on any violation.
     */
    static void validate_supervised(const Mat &X,
                                    const Vec &y,
                                    const std::string &context = "Model");
  };

} // namespace vix::ai::ml

#endif // VIX_AI_ML_MODEL_HPP
