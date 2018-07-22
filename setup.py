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
    install_requires=[
        'certifi>=2018.4.16',
        'chardet>=3.0.4',
        'colorful>=0.4.1',
        'idna>=2.7',
        'numpy>=1.14.5',
        'pandas>=0.23.3',
        'patsy>=0.5.0',
        'pkginfo>=1.4.2',
        'psycopg2>=2.7.5',
        'python-dateutil>=2.7.3',
        'pytz>=2018.5',
        'requests>=2.19.1',
        'requests-toolbelt>=0.8.0',
        'scikit-learn>=0.19.2',
        'scipy>=1.1.0',
        'six>=1.11.0',
        'sklearn>=0.0',
        'SQLAlchemy==1.2.10',
        'statsmodels>=0.9.0',
        'tqdm>=4.23.4',
        'urllib3>=1.23'
    ],
    
    entry_points={
        'console_scripts': [
            'cape-mine=capexplain.cape_miner:main',
            'cape-xplain=capexplain.cape_xplain:main',
        ]
    },

    description='Cape is a system for explaining outliers in aggregation results.',
    long_description='Cape ... \n\n',

    packages=find_packages(exclude=['capexplain.dev']),

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
