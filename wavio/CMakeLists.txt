cmake_minimum_required(VERSION 3.14)

# set the project name and version
project(wavio VERSION 1.0)

add_library(wavio wavwriter.c wavreader.c)

# state that anybody linking to us needs to include the current source dir
# to find MathFunctions.h, while we don't.
target_include_directories(wavio
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )

# install rules
install(TARGETS wavio DESTINATION lib)
install(FILES wavwriter.h DESTINATION include)
install(FILES wavreader.h DESTINATION include)
