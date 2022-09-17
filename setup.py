from setuptools import find_packages, setup

VERSION = {"VERSION": "2.0"}  # type: ignore


setup(
    name="tmynnlp",
    version=VERSION["VERSION"],
    description="An open-source NLP research library, built on PyTorch.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="tmynnlp NLP deep learning machine learning",
    url="...",
    author="Hovhannes Tamoyan",
    author_email="...",
    license="Apache",
    packages=find_packages(
        include=[
            "tmynnlp",
            "tmynnlp.*"
        ],
        exclude=[
            "*.tests",
            "*.tests.*",
            "tests.*",
            "tests",
            "test_fixtures",
            "test_fixtures.*",
            "benchmarks",
            "benchmarks.*",
        ]
    ),
    install_requires=[
        "torch>=1.6.0",
        "overrides>=3.1.0",
        "datasets",
        "sklearn",
        "numpy",
        "tqdm>=4.19",
        "transformers>=4.0",
    ],
    entry_points={"console_scripts": ["tmynnlp=tmynnlp.__main__:run"]},
    include_package_data=True,
    python_requires=">=3.8.9",
    zip_safe=False,
)
