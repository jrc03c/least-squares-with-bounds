from setuptools import setup

setup(
    name="least_squares_with_bounds",
    version="0.0.4",
    description="just a little wrapper around scipy.optimize.minimize",
    url="https://github.com/jrc03c/least-squares-with-bounds",
    author="jrc03c",
    author_email="jrc03c@pm.me",
    license="none",
    packages=["least_squares_with_bounds"],
    install_requires=["numpy", "scipy", "pyds @ git+https://github.com/jrc03c/pyds"],
    classifiers=[
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
