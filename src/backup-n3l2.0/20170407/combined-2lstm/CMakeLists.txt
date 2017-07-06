
include_directories(
  model
)

add_executable(CombinedExtractor CombinedExtractor.cpp)
target_link_libraries(CombinedExtractor ${LIBS})

