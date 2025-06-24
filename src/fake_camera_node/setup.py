from setuptools import setup

package_name = 'fake_camera_node'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yahuan',
    maintainer_email='info.yahuan@gmail.com',
    description='Fake camera node for ROS2 image testing',
    license='MIT',
    entry_points={
        'console_scripts': [
            'fake_camera_node = fake_camera_node.fake_camera_node:main',
        ],
    },
)
