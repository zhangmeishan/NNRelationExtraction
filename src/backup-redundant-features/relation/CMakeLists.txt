
include_directories(
  model
)

add_executable(RelationClassifier RelationClassifier.cpp)
target_link_libraries(RelationClassifier ${LIBS})

