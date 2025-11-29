from pathlib import Path

from setuptools import setup


def read_requirements() -> list[str]:
    """Return dependencies listed in requirements.txt if the file exists."""
    req_file = Path(__file__).with_name("requirements.txt")
    if not req_file.exists():
        return []

    requirements: list[str] = []
    for line in req_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            requirements.append(stripped)
    return requirements


ROOT_DIR = Path(__file__).parent
PACKAGES = [
    "memory",
    "memory.api",
    "memory.memory_system",
]
PACKAGE_DIR = {
    "memory": ".",
    "memory.api": "api",
    "memory.memory_system": "memory_system",
}

setup(
    name="memory",
    version="0.1.0",
    description="ResearchAgent memory subsystem for semantic, episodic, procedural, and working memory.",
    long_description=ROOT_DIR.joinpath("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="ResearchAgent",
    url="https://github.com/your-org/researchagent",
    packages=PACKAGES,
    package_dir=PACKAGE_DIR,
    python_requires=">=3.10",
    include_package_data=True,
)
