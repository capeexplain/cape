try:
    from setuptools.core import setup
except ImportError:
    from distutils.core import setup

from setuptools import find_packages
    
PACKAGE = 'capexplain'
VERSION = '0.1'

setup(
        name=PACKAGE,
        version=VERSION,
        author='US <e@g.com>',
        author_email='who@gmail.com',
        url='https://github.com/iitdbgroup/cape',
        install_requires=[],

  #install_requires=[
  #      'Django>=1.6.0',
    #],
    
        entry_points={
            'console_scripts': [
                'cape-mine=capexplain.cape_miner:main',
                'cape-xplain=capexplain.cape_xplain:main',
            ]
        },

        description='Cape is a system for explaining outliers in aggregation results.',
        long_description='Cape ... \n\n',

        packages=find_packages(exclude=['capexplain.dev'])
        # packages=[
        #     'capexplain',
        #     'capexplain.pattern_miner',
        #     'capexplain.fd',
        # ],
        keywords='db',
        platforms='any',
        license='Apache2',
    
        classifiers=[
            'Programming Language :: Python',
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',

            'Intended Audience :: End Users/Desktop',
            'Intended Audience :: Developers',

            'Topic :: Data Analysis :: Explanations',
        ],
    )
