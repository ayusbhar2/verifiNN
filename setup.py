""" source:
https://github.com/pypa/sampleproject/blob/db5806e0a3204034c51b1c00dde7d5eb3fa2532e/setup.py
"""

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="verifiNN",  # Required
    version="0.0.0.dev5",  # Required
    description="A package for optimization based neural network verification.",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",  # Optional
    url="https://github.com/harshgit/verifiNN/tree/ayush_packaging",  # Optional
    author="Ayush Bharadwaj",  # Optional
    author_email="ayush.bharadwaj@gmail.com",  # Optional
    classifiers=[  # Optional
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7"
    ],
    keywords="neural networks, verification, convex optimization, semidefinit programming",  # Optional
    package_dir={"": "verifiNN"},  # Optional
    packages=find_packages(include="verifiNN", exclude="tests"),  # Required
    python_requires=">=3.7, <4",
    install_requires=[
        "numpy == 1.21.5",
        "scipy == 1.7.3"
    ],  # Optional
    # extras_require={  # Optional
    #     "dev": ["check-manifest"],
    #     "test": ["coverage"],
    # },
    # package_data={  # Optional
    #     "sample": ["package_data.dat"],
    # },
    # entry_points={  # Optional
    #     "console_scripts": [
    #         "sample=sample:main",
    #     ],
    # },
    # # List additional URLs that are relevant to your project as a dict.
    # #
    # # This field corresponds to the "Project-URL" metadata fields:
    # # https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
    # #
    # # Examples listed include a pattern for specifying where the package tracks
    # # issues, where the source is hosted, where to say thanks to the package
    # # maintainers, and where to support the project financially. The key is
    # # what's used to render the link text on PyPI.
    # project_urls={  # Optional
    #     "Bug Reports": "https://github.com/pypa/sampleproject/issues",
    #     "Funding": "https://donate.pypi.org",
    #     "Say Thanks!": "http://saythanks.io/to/example",
    #     "Source": "https://github.com/pypa/sampleproject/",
    # },
)