
include_directories(
  model
)

add_executable(BeamExtractor BeamExtractor.cpp)
target_link_libraries(BeamExtractor ${LIBS})

