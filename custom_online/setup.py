from setuptools import setup

setup(
    name="transfer_em",
    version='0.1',
    scripts=['predictor.py'],
    install_requires=["requests", "google-cloud_storage"]
)
