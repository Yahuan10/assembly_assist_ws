from setuptools import setup

package_name = 'state_manager_node'

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
    description='Assembly process state management based on gestures',
    license='MIT',
    entry_points={
        'console_scripts': [
            'state_manager_node = state_manager_node.state_manager_node:main',
        ],
    },
)
