
include_directories(
  model
)

add_executable(CombinedEXTSYNExtractor CombinedEXTSYNExtractor.cpp)
target_link_libraries(CombinedEXTSYNExtractor ${LIBS})

