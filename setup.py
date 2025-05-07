from setuptools import find_packages, setup

package_name = 'yolo12_bbox_coordinate'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jwg',
    maintainer_email='wjddnrud4487@naver.com',
    description='YOLOV12 Boundting Box Center Coordinate',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_bbox_center_node = yolo12_bbox_coordinate.yolo_bbox_center_node:main',
            'yolo_XYZ_pub = yolo12_bbox_coordinate.yolo_XYZ_pub:main',
            'yolo_XYZ_sub = yolo12_bbox_coordinate.yolo_XYZ_sub:main'
        ],
    },
)
