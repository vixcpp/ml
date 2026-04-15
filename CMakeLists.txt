#  @file CMakeLists.txt
#  @author Gaspard Kirira
#
#  Copyright 2025, Gaspard Kirira. All rights reserved.
#  https://github.com/vixcpp/vix
#  Use of this source code is governed by a MIT license
#  that can be found in the License file.

cmake_minimum_required(VERSION 3.20)
project(vix_ai_ml VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Global settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(VIX_AI_ML_BUILD_TESTS "Build vix_ai_ml tests" OFF)
option(VIX_AI_ML_INSTALL     "Install vix_ai_ml targets" ON)
option(VIX_AI_ML_WARNINGS    "Enable extra warnings for vix_ai_ml" ON)

# Sources
set(VIX_AI_ML_SOURCES
  src/Dataset.cpp
  src/Model.cpp
  src/Regression.cpp
  src/Clustering.cpp
)

# Library
add_library(vix_ai_ml STATIC ${VIX_AI_ML_SOURCES})
add_library(vix::ai::ml ALIAS vix_ai_ml)

target_compile_features(vix_ai_ml PUBLIC cxx_std_20)

target_include_directories(vix_ai_ml
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(VIX_AI_ML_WARNINGS)
  if(MSVC)
    target_compile_options(vix_ai_ml PRIVATE /W4 /permissive-)
  else()
    target_compile_options(vix_ai_ml PRIVATE
      -Wall
      -Wextra
      -Wpedantic
      -Wconversion
      -Wshadow
    )
  endif()
endif()

if(TARGET vix_warnings)
  target_link_libraries(vix_ai_ml PUBLIC vix_warnings)
endif()

if(VIX_ENABLE_SANITIZERS AND TARGET vix_sanitizers)
  target_link_libraries(vix_ai_ml PUBLIC vix_sanitizers)
endif()

set_target_properties(vix_ai_ml PROPERTIES
  OUTPUT_NAME vix_ai_ml
  VERSION ${PROJECT_VERSION}
  SOVERSION 0
  EXPORT_NAME ai::ml
)

# Examples
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt")
  add_subdirectory(examples)
endif()

# Tests
if(VIX_AI_ML_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt")
  include(CTest)
  enable_testing()
  add_subdirectory(test)
endif()

# Install
if(VIX_AI_ML_INSTALL)
  install(TARGETS vix_ai_ml
    EXPORT vix_ai_mlTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
  )

  install(EXPORT vix_ai_mlTargets
    FILE vix_ai_mlTargets.cmake
    NAMESPACE vix::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vix_ai_ml
  )

  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vix_ai_mlConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/vix_ai_mlConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/vix_ai_ml"
  )

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/vix_ai_mlConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
  )

  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/vix_ai_mlConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/vix_ai_mlConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/vix_ai_ml"
  )
endif()

# Summary
message(STATUS "------------------------------------------------------")
message(STATUS "vix::ai::ml configured (${PROJECT_VERSION})")
message(STATUS "Include dir: ${CMAKE_CURRENT_SOURCE_DIR}/include")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Install: ${VIX_AI_ML_INSTALL}")
message(STATUS "Tests: ${VIX_AI_ML_BUILD_TESTS}")
message(STATUS "------------------------------------------------------")
