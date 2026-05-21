from setuptools import find_packages, setup

package_name = 'bt_viewer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={
        package_name: ['templates/*.html', 'static/*/*'],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hans',
    maintainer_email='hoanghuypham6801@gmail.com',
    description='A basic web UI for behavior tree execution monitoring and blackboard telemetry',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'bt_viewer_node = bt_viewer.bt_viewer_node:main',
        ],
    },
)
