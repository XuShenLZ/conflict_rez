import setuptools

setuptools.setup(
    name="confrez",
    version="0.1.0",
    description="Conflict resolution for multiple vehicles in tight spaces",
    author="Xu Shen",
    author_email="xu_shen@berkeley.edu",
    packages=["confrez"],
    install_requires=[
        "black",
        "numpy",
        "scipy",
        "matplotlib",
        "casadi",
        "pytope",
        "pettingzoo",
        "stable-baselines3[extra]",
        "supersuit",
        "jupyterlab",
        "pygame",
        "pymunk",
        "triangle",
    ],
    # install pytorch according to CUDA version / CPU
)
