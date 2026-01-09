"""Setup script that exposes the `agents` package for easy importing."""

from setuptools import find_packages, setup


SRC_RELATIVE = ".."


setup(
    name="research-agents",
    version="0.1.0",
    description="Agent modules for the ResearchAgent project.",
    packages=find_packages(where=SRC_RELATIVE, include=["agents", "agents.*"]),
    package_dir={"": SRC_RELATIVE},
    include_package_data=True,
)
