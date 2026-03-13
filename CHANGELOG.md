# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Initial examples for KMeans, Linear Regression, Logistic Regression, Preprocessing.
- Support for saving and loading ML models.
- Elbow curve method for determining optimal number of clusters.
- MiniBatchKMeans implementation.
- Dataset loader from CSV files.

### Changed
- N/A

### Fixed
- N/A

---

## [0.1.0] - 2026-03-13

### Added
- Core ML library with:
  - `Dataset` handling
  - `Model` base class
  - `Regression` (linear and logistic)
  - `Clustering` (KMeans)
- CMake build system with options for tests, warnings, and installation.
- Example programs in `examples/` directory.
- Integration with `vix::ai::tensor` library (optional for future tensor-based operations).

### Changed
- N/A

### Fixed
- N/A
