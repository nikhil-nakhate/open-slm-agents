from setuptools import setup, find_packages


setup(
    name="open-slm-agents",
    version="0.1.0",
    description="Modular GPT-style training/eval framework with configs and registry",
    packages=find_packages(exclude=("outputs", "configs", "scripts", "data")),
    py_modules=["train", "eval"],
    python_requires=">=3.8",
    install_requires=[
        "pyyaml>=6.0",
        "torch>=2.0",
        "numpy>=1.23",
        "transformers>=4.40",
        "tensorflow>=2.11",
        "tiktoken>=0.5.0",
    ],
    extras_require={
        # Hugging Face tokenizer/model support
        "hf": ["transformers>=4.40"],
        # Logging backends
        "wandb": ["wandb>=0.15"],
        "tb": ["tensorboard>=2.11"],
        # Test dependencies
        "test": ["pytest>=7.0.0", "pytest-cov>=4.0.0"],
        # Developer tools
        "dev": ["black>=23.0", "flake8>=6.0"],
        # Everything optional
        "all": ["transformers>=4.40", "wandb>=0.15", "tensorboard>=2.11", "pytest>=7.0.0", "pytest-cov>=4.0.0"],
    },
    entry_points={
        "console_scripts": [
            "oslma-train=train:main",
            "oslma-eval=eval:main",
        ]
    },
)
