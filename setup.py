from distutils.core import setup
import os

version_file = os.path.join(os.path.dirname(__file__), "telepath_worker", "version.txt")
with open(version_file, "r") as file_handler:
    telepath_version = file_handler.read().strip()

setup(
    name='telepath_worker',
    version=telepath_version,
    packages=['telepath_worker','telepath_gym','deepgym'],
    license='...',
    #long_description=open('README.txt').read(),
)
