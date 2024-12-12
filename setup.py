from setuptools import setup, find_packages

setup(
    name = "psps",
    version = "0.0.3", 
    #packages = find_packages(),
    packages = ['psps'],
    package_dir={"":"src"},
    author = 'Chris Lam',
    author_email = 'c.lam@ufl.edu',
    description = 'planetary system population synthesis',
    license = 'MIT License'
    #install_requires = [
    #    'numpy'
    #]
)
