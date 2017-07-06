
include_directories(
  model
)

add_executable(GreedyExtractor GreedyExtractor.cpp)
target_link_libraries(GreedyExtractor ${LIBS})

