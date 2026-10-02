from setuptools import find_packages, setup

package_name = 'daaam_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='ngorlo@mit.edu',
    description='ROS2 interface for a package for creating large-scale spatio-temporal memory with detailed annotations in real-time',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'daaam_node = daaam_ros.nodes.daaam_node:main',
            'rerun_visualizer_node = daaam_ros.nodes.rerun_visualizer_node:main',
            'dataloader_node = daaam_ros.nodes.dataloader.dataloader_node:main',
        ],
    },
)