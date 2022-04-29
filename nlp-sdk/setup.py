from setuptools import setup, find_packages

setup(
    name='huaweicloud-python-sdk-nlp',
    version='1.0.0',
	author='Huaweicloud NLP',
    packages=find_packages(),
    zip_safe=False,
    description='nlp python sdk',
    long_description='nlp python sdk',
	install_requires=['requests'],
    license='Apache-2.0',
    keywords=('nlp', 'sdk', 'python'),
    platforms='Independant'
)