from setuptools import setup, find_packages

setup(name='interactions_package',
      packages=find_packages(),
      install_required = ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'shap', 'PyALE'],
      version ='1.0')