
include_directories(
  model
)

add_executable(CombinedDEPExtractor CombinedDEPExtractor.cpp)
target_link_libraries(CombinedDEPExtractor ${LIBS})

