cmake_minimum_required(VERSION 3.20)

project(ml
  VERSION 0.1.0
  DESCRIPTION "Vix AI ML: datasets, models, regression, clustering"
  LANGUAGES CXX
)

# ---- Options ---------------------------------------------------------------
option(VIX_AI_ML_BUILD_TESTS "Build ML tests" ON)
option(VIX_AI_ML_INSTALL     "Install ML targets" ON)
option(VIX_AI_ML_WARNINGS    "Enable extra warnings" ON)

# ---- Library ---------------------------------------------------------------
add_library(vix_ai_ml
  src/Dataset.cpp
  src/Model.cpp
  src/Regression.cpp
  src/Clustering.cpp
)

# Alias pratique
add_library(vix::ai::ml ALIAS vix_ai_ml)

# Include directories
target_include_directories(vix_ai_ml
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# C++20
target_compile_features(vix_ai_ml PUBLIC cxx_std_20)

# Warnings
if(VIX_AI_ML_WARNINGS)
  if(MSVC)
    target_compile_options(vix_ai_ml PRIVATE /W4 /permissive-)
  else()
    target_compile_options(vix_ai_ml PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  endif()
endif()

# ---- Examples --------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples")
  add_subdirectory(examples)
endif()

# ---- Tests -----------------------------------------------------------------
if(VIX_AI_ML_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
  enable_testing()
  add_subdirectory(tests)
endif()

# ---- Install ---------------------------------------------------------------
if(VIX_AI_ML_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  # Installer la librairie
  install(TARGETS vix_ai_ml
    EXPORT vix_ai_mlTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )

  # Installer les headers
  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  # Export CMake targets
  install(EXPORT vix_ai_mlTargets
    FILE vix_ai_mlTargets.cmake
    NAMESPACE vix::ai::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vix_ai/ml
  )

  # Version + config files
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/vix_ai_mlConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
  )

  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/vix_ai_mlConfig.cmake
"include(\"\${CMAKE_CURRENT_LIST_DIR}/vix_ai_mlTargets.cmake\")\n"
  )

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/vix_ai_mlConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/vix_ai_mlConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vix_ai/ml
  )
endif()
