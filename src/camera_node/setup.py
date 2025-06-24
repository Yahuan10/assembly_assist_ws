from setuptools import setup

package_name = 'camera_node'

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
    description='Basler camera node with image publishing',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = camera_node.camera_node:main',
        ],
    },
)
