import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'moveit_api'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hans',
    maintainer_email='hans@todo.todo',
    description='REST API and MoveIt2 end-effector controller for OpenArm bimanual robot',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'robot_api_server = moveit_api.robot_api_server:main',
        ],
    },
)
