cmake_minimum_required(VERSION 3.15)
project(insorderset LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Find Python ──────────────────────────────────────────────────────────────
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

# ── Find nanobind (via its CMake helpers) ────────────────────────────────────
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# ── Build the extension module ────────────────────────────────────────────────
nanobind_add_module(
    _insorderset_core
    NB_STATIC           # statically link nanobind (smaller wheel)
    src/_insorderset_core.cpp
)

install(TARGETS _insorderset_core LIBRARY DESTINATION insorderset)
