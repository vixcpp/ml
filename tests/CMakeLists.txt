# --------------------------------------------------------------
# Minimal tests for vix-ai-ml (no external framework)
# --------------------------------------------------------------
if (BUILD_TESTING)
  add_executable(vix_ai_ml_smoke_test ml_smoke_test.cpp)
  target_link_libraries(vix_ai_ml_smoke_test PRIVATE Vix::AI::ml)
  target_compile_features(vix_ai_ml_smoke_test PRIVATE cxx_std_20)
  add_test(NAME vix_ai_ml_smoke_test COMMAND vix_ai_ml_smoke_test)
  set_tests_properties(vix_ai_ml_smoke_test PROPERTIES LABELS "ml;smoke")

  add_executable(vix_ai_ml_test test_ml.cpp)
  target_link_libraries(vix_ai_ml_test PRIVATE Vix::AI::ml)
  target_compile_features(vix_ai_ml_test PRIVATE cxx_std_20)
  add_test(NAME vix_ai_ml_test COMMAND vix_ai_ml_test)
  set_tests_properties(vix_ai_ml_test PROPERTIES LABELS "ml;functional")
endif()
