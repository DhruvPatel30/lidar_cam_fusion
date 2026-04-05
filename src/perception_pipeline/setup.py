from setuptools import find_packages, setup

package_name = "perception_pipeline"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="LiDAR-Camera fusion pipeline using YOLOv8 and Open3D",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # lidar_processor is a C++ binary installed by CMakeLists.txt
            "fusion_node = perception_pipeline.fusion_node:main",
            "kitti_publisher = perception_pipeline.kitti_publisher_node:main",
        ],
    },
)
