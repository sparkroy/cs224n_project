from setuptools import setup, find_packages


setup(
    name="bert",
    packages=[
        package for package in find_packages() if package.startswith("bert")
    ],
    install_requires=[
    ],
    eager_resources=['*'],
    include_package_data=True,
    python_requires='>=3',
    description="bert",
    author="",
    url="",
    author_email="",
    version="0.1.0",
)