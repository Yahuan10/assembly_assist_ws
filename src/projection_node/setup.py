from setuptools import setup

package_name = 'projection_node'

setup(
    name=package_name,
    version='0.0.2',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'pillow'],
    zip_safe=True,
    maintainer='Yahuan',
    maintainer_email='info.yahuan@gmail.com',
    description='Full-screen projection node using Tkinter',
    license='MIT',
    entry_points={
        'console_scripts': [
            'projection_node = projection_node.projection_node:main',
        ],
    },
)
