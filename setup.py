from setuptools import setup

setup(
    name='src',
    package_dir = {"": "src"},
    description='Code for the paper "An Investigation into the Performance of Non-Contrastive Self-Supervised Learning Methods for Network Intrusion Detection"',
    install_requires=[
        'bayesian-optimization',
        'matplotlib',
        'numpy',
        'pandas',
        'pyyaml',
        'pyarrow',
        'ray',
        'scikit-learn',
        'tqdm',
        'torch',
    ]
)
