
include_directories(
  model
)

add_executable(CombinedSYNExtractor CombinedSYNExtractor.cpp)
target_link_libraries(CombinedSYNExtractor ${LIBS})

