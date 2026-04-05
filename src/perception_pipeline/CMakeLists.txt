cmake_minimum_required(VERSION 3.22)
project(perception_pipeline)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── ROS 2 / ament ─────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(visualization_msgs REQUIRED)

# ── conda/pixi prefix — helps all find_package calls below locate packages ────
if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
  set(Python3_ROOT_DIR "$ENV{CONDA_PREFIX}")
  # Pin OpenCV to the conda env so we don't accidentally pick up Homebrew's
  set(OpenCV_DIR "$ENV{CONDA_PREFIX}/lib/cmake/opencv4")
endif()

# ── Open3D (conda-forge — found via $CONDA_PREFIX/lib/cmake/Open3D/) ──────────
find_package(Open3D REQUIRED)
# Open3D's conda dylib embeds Python bindings — link Python explicitly so its
# symbols (_PyExc_RuntimeError etc.) resolve at load time on macOS.
find_package(Python3 REQUIRED COMPONENTS Development)

# macOS: Open3D's CMake config references "atomic" in INTERFACE_LINK_LIBRARIES
# across multiple transitive targets. libatomic is a GCC/Linux library — Apple
# clang provides atomic operations natively. By declaring an INTERFACE target
# named "atomic", CMake resolves the name to this no-op target instead of
# passing -latomic to the macOS linker (which has no libatomic).
if(APPLE)
  if(NOT TARGET atomic)
    add_library(atomic INTERFACE)
  endif()
endif()

# ── OpenCV C++ (conda-forge: libopencv) ───────────────────────────────────────
find_package(OpenCV REQUIRED)

# ── ONNX Runtime C++ (conda-forge: libonnxruntime) ───────────────────────────
find_package(onnxruntime REQUIRED)

# ── C++ library: LidarPreprocessor (no ROS dependency, unit-testable) ─────────
add_library(lidar_preprocessor
  src/lidar_preprocessor.cpp
)
target_include_directories(lidar_preprocessor PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(lidar_preprocessor rclcpp sensor_msgs)
target_link_libraries(lidar_preprocessor Open3D::Open3D Python3::Python)

# ── C++ executable: lidar_processor ───────────────────────────────────────────
add_executable(lidar_processor
  src/lidar_processor_node.cpp
)
target_include_directories(lidar_processor PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
ament_target_dependencies(lidar_processor rclcpp sensor_msgs)
target_link_libraries(lidar_processor lidar_preprocessor Open3D::Open3D Python3::Python)
# macOS: rosidl_generator_py dylibs use flat namespace and expect Python symbols
# to already be in the process. Force libpython into the load table so dyld can
# resolve _PyExc_RuntimeError etc. even though lidar_processor has no direct
# Python call sites.
if(APPLE)
  get_target_property(_python_lib Python3::Python IMPORTED_LOCATION)
  target_link_options(lidar_processor PRIVATE "-Wl,-needed_library,${_python_lib}")
endif()

# ── C++ library: CameraDetector (no ROS dependency, unit-testable) ───────────
add_library(camera_detector
  src/camera_detector.cpp
)
target_include_directories(camera_detector PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(camera_detector
  ${OpenCV_LIBS}
  onnxruntime::onnxruntime
)

# ── C++ executable: camera_detector_node ─────────────────────────────────────
add_executable(camera_detector_node
  src/camera_detector_node.cpp
)
target_include_directories(camera_detector_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
ament_target_dependencies(camera_detector_node rclcpp sensor_msgs vision_msgs)
target_link_libraries(camera_detector_node
  camera_detector
  ${OpenCV_LIBS}
  onnxruntime::onnxruntime
)
if(APPLE)
  get_target_property(_python_lib Python3::Python IMPORTED_LOCATION)
  target_link_options(camera_detector_node PRIVATE
    "-Wl,-needed_library,${_python_lib}")
endif()

# ── Python nodes via ament_cmake_python ───────────────────────────────────────
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS perception_pipeline/kitti_publisher_node.py
  DESTINATION lib/${PROJECT_NAME} RENAME kitti_publisher)

# ── C++ library: LidarCameraProjector (no ROS / no Open3D dependency) ─────────
add_library(lidar_camera_projector
  src/lidar_camera_projector.cpp
)
target_include_directories(lidar_camera_projector PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(lidar_camera_projector Eigen3::Eigen)
target_compile_features(lidar_camera_projector PUBLIC cxx_std_17)

# ── C++ executable: fusion_node ────────────────────────────────────────────────
add_executable(fusion_node
  src/fusion_node.cpp
)
target_include_directories(fusion_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
ament_target_dependencies(fusion_node
  rclcpp sensor_msgs vision_msgs message_filters ament_index_cpp visualization_msgs)
target_link_libraries(fusion_node lidar_camera_projector yaml-cpp::yaml-cpp ${OpenCV_LIBS})
if(APPLE)
  get_target_property(_python_lib Python3::Python IMPORTED_LOCATION)
  target_link_options(fusion_node PRIVATE "-Wl,-needed_library,${_python_lib}")
endif()

# ── Install C++ targets ────────────────────────────────────────────────────────
install(TARGETS
  lidar_preprocessor
  lidar_processor
  camera_detector
  lidar_camera_projector
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
# Install camera_detector_node binary renamed to "camera_detector" so the
# launch file and ros2 run both find it under the expected name.
install(TARGETS camera_detector_node
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(CODE
  "file(RENAME
    \"\${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}/camera_detector_node\"
    \"\${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}/camera_detector\")"
)
install(TARGETS fusion_node
  RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
  DESTINATION include
)

# ── Install shared resources ──────────────────────────────────────────────────
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
# OPTIONAL: models/ dir is generated by `pixi run export-yolo` and not tracked
# in git. The install is a no-op until yolo11s.onnx is present.
install(DIRECTORY models/ DESTINATION share/${PROJECT_NAME}/models OPTIONAL)

# ── Testing ───────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
