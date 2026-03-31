from setuptools import find_packages, setup
import os
from glob import glob

package_name = "perception_pipeline"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="LiDAR-Camera fusion pipeline using YOLOv8 and Open3D",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "camera_detector = perception_pipeline.camera_detector_node:main",
            "lidar_processor = perception_pipeline.lidar_processor_node:main",
            "fusion_node = perception_pipeline.fusion_node:main",
            "kitti_publisher = perception_pipeline.kitti_publisher_node:main",
        ],
    },
)
