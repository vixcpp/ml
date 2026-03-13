# Dataset example
add_executable(example_dataset_csv dataset_csv.cpp)
target_link_libraries(example_dataset_csv PRIVATE vix::ai::ml)

# Preprocessing example
add_executable(example_preprocessing preprocessing.cpp)
target_link_libraries(example_preprocessing PRIVATE vix::ai::ml)

# Linear regression example
add_executable(example_linear_regression linear_regression.cpp)
target_link_libraries(example_linear_regression PRIVATE vix::ai::ml)

# Logistic regression example
add_executable(example_logistic_regression logistique_regression.cpp)
target_link_libraries(example_logistic_regression PRIVATE vix::ai::ml)

# KMeans clustering example
add_executable(example_kmeans kmeans.cpp)
target_link_libraries(example_kmeans PRIVATE vix::ai::ml)

# Full pipeline example
add_executable(example_full_pipeline full_pipeline.cpp)
target_link_libraries(example_full_pipeline PRIVATE vix::ai::ml)
