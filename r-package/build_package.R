  library(Rcpp)
  library(devtools)
  library(testthat)
  library(roxygen2)
  
  package.name <- "grftsls"
  package.src <- "grftsls/src"
  
  # Copy Rcpp bindings and C++ source into the package src directory. Note that we
  # don't copy in third_party/Eigen, because for the R package build we provide
  # access to the library through RcppEigen.
  unlink(package.src, recursive = TRUE)
  dir.create(package.src)
  
  binding.files <- list.files("grftsls/bindings", full.names = TRUE)
  file.copy(binding.files, package.src, recursive = FALSE)
  file.copy("../core/src", package.src, recursive = TRUE)
  file.copy("../core/third_party/optional", package.src, recursive = TRUE)
  
  # Auto-generate documentation files
  roxygen2::roxygenise(package.name)
  
  # Run Rcpp and build the package.
  compileAttributes(package.name)
  clean_dll(package.name)
  build(package.name)
  
  # Test installation and run some smoke tests.
  install(package.name)
  library(package.name, character.only = TRUE)
  #test_package(package.name)
