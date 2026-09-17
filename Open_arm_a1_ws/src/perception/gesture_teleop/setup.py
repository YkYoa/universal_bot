import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'gesture_teleop'


def web_data_files():
    """Recursively install web/ (gesture mockup HTML/JS/CSS + vendored
    socket.io), preserving directory structure under share/gesture_teleop/web/
    - ament data_files has no built-in recursive copy, so walk it manually
    (same pattern as moveit_api/setup.py's web_visualizer_data_files())."""
    entries = []
    src_root = 'web'
    for dirpath, _dirnames, filenames in os.walk(src_root):
        if not filenames:
            continue
        dest = os.path.join('share', package_name, dirpath)
        files = [os.path.join(dirpath, f) for f in filenames]
        entries.append((dest, files))
    return entries


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test', 'test.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
        *web_data_files(),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='robot-healthmate',
    maintainer_email='tech@infall.io',
    description='Human-arm gesture teleoperation for the OpenArm bimanual robot',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'teleop_node = gesture_teleop.teleop_node:main',
            'mock_server = gesture_teleop.mock_server:main',
        ],
    },
)
