project(Optimization_test)
enable_testing()

include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# UNIT TEST
add_executable(test_optimization
    test_optimization.cxx
)

target_link_libraries(test_optimization
    optimization
    gtest
    gtest_main
)

add_test(NAME test_optimization COMMAND run_test_optimization)

#INTEGRATION TEST
add_executable(integration_optimization
integration_optimization.cxx
)

target_link_libraries(integration_optimization
    optimization
    gtest
    gtest_main
)

add_test(NAME integration_optimization COMMAND run_integration_optimization)

# BENCHMARK
add_executable(bench_optimization
    bench_optimization.cxx
)

target_link_libraries(bench_optimization
    optimization
    benchmark::benchmark
)

# EVALUATION
add_executable(evaluate_iris_classifier
    evaluate_iris_classifier.cxx
)

target_link_libraries(evaluate_iris_classifier
    optimization
)

# We want to generate a hex representation of the CSV file that can be loaded
# into our tests

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/data)

string(
  REGEX REPLACE
  ".csv$"
  ".h"
  GENERATED_FILE
  ${CMAKE_CURRENT_BINARY_DIR}/data/iris.csv
)

add_custom_command(
  OUTPUT ${GENERATED_FILE}
  COMMAND xxd -i data/iris.csv > ${GENERATED_FILE}
  MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/data/iris.csv
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating ${GENERATED_FILE}"
  VERBATIM
)

add_custom_target(
  Generate_data
  DEPENDS
  ${GENERATED_FILE}
)

add_dependencies(test_optimization Generate_data)