project(Optimization_src)

add_library(optimization
    optimization.cpp
    nn.cpp 
    activation.cpp 
    kFold.cpp  # add additional source files here
)

target_link_libraries(optimization
    Eigen3::Eigen
)

