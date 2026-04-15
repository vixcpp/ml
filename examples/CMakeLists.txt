#  @file examples/CMakeLists.txt
#  @brief Examples for the vix_ai_ml module.

add_executable(example_dataset_csv
  dataset_csv.cpp
)
target_link_libraries(example_dataset_csv
  PRIVATE
    vix::ai::ml
)

add_executable(example_preprocessing
  preprocessing.cpp
)
target_link_libraries(example_preprocessing
  PRIVATE
    vix::ai::ml
)

add_executable(example_linear_regression
  linear_regression.cpp
)
target_link_libraries(example_linear_regression
  PRIVATE
    vix::ai::ml
)

add_executable(example_logistic_regression
  logistique_regression.cpp
)
target_link_libraries(example_logistic_regression
  PRIVATE
    vix::ai::ml
)

add_executable(example_kmeans
  kmeans.cpp
)
target_link_libraries(example_kmeans
  PRIVATE
    vix::ai::ml
)

add_executable(example_full_pipeline
  full_pipeline.cpp
)
target_link_libraries(example_full_pipeline
  PRIVATE
    vix::ai::ml
)
