from setuptools import setup, find_packages

package_name = 'camera_bridge'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'Flask', 'opencv-python'],
    zip_safe=True,
    maintainer='hans',
    maintainer_email='hans@todo.todo',
    description='HTTP camera bridge tu Jetson NX sang sensor_msgs/Image (IQ-9075 Type-C camera port bi chay)',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'camera_bridge_node = camera_bridge.camera_bridge_node:main',
            'pi05_camera_test_node = camera_bridge.pi05_camera_test_node:main',
        ],
    },
)
