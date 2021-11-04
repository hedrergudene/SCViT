import setuptools

packages = setuptools.find_packages(where=".")

setuptools.setup(
    name="hvit",  # Replace with your own username
    version="0.0.1",
    author="Antonio Zarauz",
    description="HVIT",
    long_description="HVIT",
    #url=gitlab project,
    classifiers=[
                "Programming Language :: Python :: 3",
                # "License :: OSI Approved :: MIT License",
                "Operating System :: OS Independent",
                ],
    python_requires='>=3.6',
    package_dir={"": "."},
    packages=packages
)
