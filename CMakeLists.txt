cmake_minimum_required(VERSION 3.20)

project(ml
  VERSION 0.1.0
  DESCRIPTION "Vix AI ML: datasets, models, regression, clustering"
  LANGUAGES CXX)

# ---- Options ---------------------------------------------------------------
option(VIX_AI_ML_BUILD_TESTS "Build ML tests" ON)
option(VIX_AI_ML_INSTALL     "Install ML targets" ON)
option(VIX_AI_ML_WARNINGS    "Enable extra warnings" ON)

# ---- Library sources -------------------------------------------------------
add_library(vix_ai_ml
  src/Dataset.cpp
  src/Model.cpp
  src/Regression.cpp
  src/Clustering.cpp
)

add_library(Vix::AI::ml ALIAS vix_ai_ml)

# ---- Include directories & compile features --------------------------------
target_include_directories(vix_ai_ml
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(vix_ai_ml PUBLIC cxx_std_20)

# ---- Warnings --------------------------------------------------------------
if(VIX_AI_ML_WARNINGS)
  if(MSVC)
    target_compile_options(vix_ai_ml PRIVATE /W4 /permissive-)
  else()
    target_compile_options(vix_ai_ml PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  endif()
endif()

# ---- Dependencies ----------------------------------------------------------
target_link_libraries(vix_ai_ml
  PUBLIC Vix::AI::core
)

# ---- Tests -----------------------------------------------------------------
if(VIX_AI_ML_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
  enable_testing()
  add_subdirectory(tests)
endif()

# ---- Install / Package -----------------------------------------------------
if(VIX_AI_ML_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  install(TARGETS vix_ai_ml
    EXPORT vix_ai_mlTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )

  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  install(EXPORT vix_ai_mlTargets
    FILE vix_ai_mlTargets.cmake
    NAMESPACE Vix::AI::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vix_ai/ml
  )

  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/vix_ai_mlConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
  )

  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/vix_ai_mlConfig.cmake
"include(\"\${CMAKE_CURRENT_LIST_DIR}/vix_ai_mlTargets.cmake\")\n")

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/vix_ai_mlConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/vix_ai_mlConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vix_ai/ml
  )
endif()
