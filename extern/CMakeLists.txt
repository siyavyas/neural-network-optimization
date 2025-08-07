project(Optimization_extern)

add_compile_options(-Wno-c++17-attribute-extensions)

set(GTEST_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googletest/include PARENT_SCOPE)

add_subdirectory(googletest)

add_subdirectory(benchmark)

add_subdirectory(eigen)
