cmake_minimum_required(VERSION 3.8)

project(rs VERSION 1.0 LANGUAGES C CXX)

find_package(MPI REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)

add_executable(rs Zero.cpp main.cpp)
target_compile_features(rs PRIVATE cxx_std_17)
target_link_libraries(rs PRIVATE PkgConfig::PETSC MPI::MPI_CXX)

add_executable(rs_im main_im.cpp)
target_compile_features(rs_im PRIVATE cxx_std_17)
target_link_libraries(rs_im PRIVATE PkgConfig::PETSC MPI::MPI_CXX)

add_executable(sample Zero.cpp sample.cpp)
target_compile_features(sample PRIVATE cxx_std_17)
target_link_libraries(rs PRIVATE PkgConfig::PETSC MPI::MPI_CXX)
