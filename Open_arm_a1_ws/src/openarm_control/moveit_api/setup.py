import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'moveit_api'


def web_visualizer_data_files():
    """Recursively install web_visualizer/ (dashboard HTML/JS + vendored
    three.js/urdf-loader/socket.io), preserving its directory structure
    under share/moveit_api/web_visualizer/ - ament data_files has no
    built-in recursive-copy, so walk it manually."""
    entries = []
    src_root = 'web_visualizer'
    for dirpath, _dirnames, filenames in os.walk(src_root):
        if not filenames:
            continue
        dest = os.path.join('share', package_name, dirpath)
        files = [os.path.join(dirpath, f) for f in filenames]
        entries.append((dest, files))
    return entries


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
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
        *web_visualizer_data_files(),
    ],
    install_requires=[
        'setuptools',
        'Flask',
        'Flask-SocketIO',
        'Flask-Compress',
    ],
    zip_safe=True,
    maintainer='hans',
    maintainer_email='hans@todo.todo',
    description='REST API and MoveIt2 end-effector controller for OpenArm bimanual robot',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'robot_api_server = moveit_api.robot_api_server:main',
            'get_robot_state = moveit_api.get_robot_state:main',
            'launch_manager_server = moveit_api.launch_manager_server:main',
        ],
    },
)
