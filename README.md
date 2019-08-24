# Cape
As an initial contribution we have developed a novel system called Cape (Counterbalancing with Aggregation Patterns for Explanations) for explaining outliers in aggregation queries through counterbalancing. That is, explanations are outliers in the opposite direction of the outlier of interest.
**Cape** (**C**ounterbalancing with **A**ggregation **P**atterns for **E**xplanations) is a system that explains outliers (surprisingly low or high) aggregation function results for group-by queries in SQL. The user provides the system with a query and a surprising outcome for this query. Cape then uses patterns discovered over the input data of the query in an offline pattern mining step to explain the outlier by finding an *related* outlier in the opposite direction that **counterbalances** the outlier of interest.

# Installation

## Prerequisites

Cape requires python 3 and uses python's `tkinter` for its graphical UI. For example, on ubuntu you can install the prerequisites with:

~~~shell
sudo apt-get install python3 python3-pip python3-tk
~~~

## Install with pip

Make sure you have pip installed (see previous step).

~~~shell
pip3 install capexplain
~~~

## Install from github

Alternatively, you can directly install from source. For that you need python3 and the setuptools package. You probably would want to use a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) for that. If you have `python3` and `pip` installed you can install/update the `setuptools` package by running:

~~~shell
~~~

To install Cape, clone the github repository and use `setup.py` to install.

~~~shell
git clone git@github.com:IITDBGroup/cape.git capexplain
cd capexplain
python3 setup.py install
~~~

# Usage

Cape provides a single binary `capexplain` that support multiple subcommands. The general form is:

~~~shell
capexplain COMMAND [OPTIONS]
~~~

Options are specific to each subcommand. Use `capexplain help` to see a list of supported commands and `capexplain help COMMAND` get more detailed help for a subcommand.

## Overview

Cape currently only supports PostgreSQL as a backend database (version 9 or higher). To use Cape to explain an aggregation outlier, you first have to let cape find patterns for the table over which you are aggregating. This an offline step that only has to be executed only once for each table (unless you want to re-run pattern mining with different parameter settings). Afterwards, you can either use the commandline or Cape's UI to request explanations for an outlier in an aggregation query result.

## Mining Patterns

Use `capexplain mine [OPTIONS]` to mine patterns. Cape will store the discovered patterns in the database. The "mined" patterns will be stored in a created schema called `pattern`, and the pattern tables generated after running `mine` command are `pattern.{target_table}_global` and `pattern.{target_table}_local`. At the minimum you have to tell Cape how to connect to the database you want to use and which table it should generate patterns for. Run `capexplain help mine` to get a list of all supported options for the mine command. The options needed to specify the target table and database connection are:

~~~shell
-h ,--host <arg>               - database connection host IP address (DEFAULT: 127.0.0.1)
-u ,--user <arg>               - database connection user (DEFAULT: postgres)
-p ,--password <arg>           - database connection password
-d ,--db <arg>                 - database name (DEFAULT: postgres)
-P ,--port <arg>               - database connection port (DEFAULT: 5432)
-t ,--target-table <arg>       - mine patterns for this table
~~~

For instance, if you run a postgres server locally (default) with user `postgres` (default), password `test`, and want to mine patterns for a table `employees` in database `mydb`, then run:

~~~shell
capexplain mine -p test -d mydb -t employees
~~~

### Mining algorithm parameters

Cape's mining algorithm takes the following arguments:

~~~shell
--gof-const <arg>              - goodness-of-fit threshold for constant regression (DEFAULT: 0.1)
--gof-linear <arg>             - goodness-of-fit threshold for linear regression (DEFAULT: 0.1)
--confidence <arg>             - global confidence threshold
-r ,--regpackage <arg>         - regression analysis package to use {'statsmodels', 'sklearn'} (DEFAULT: statsmodels)
--local-support <arg>          - local support threshold (DEFAULT: 10)
--global-support <arg>         - global support thresh (DEFAULT: 100)
-f ,--fd-optimizations <arg>   - activate functional dependency detection and optimizations (DEFAULT: False)
-a ,--algorithm <arg>          - algorithm to use for pattern mining {'naive', 'optimized', 'naive_alternative'} (DEFAULT: optimized)
--show-progress <arg>          - show progress meters (DEFAULT: True)
--manual-config                - manually configure numeric-like string fields (treat fields as string or numeric?) (DEFAULT: False)

~~~

### Running our "crime" data example

We included a subset of the "Chicago Crime" dataset (https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/)
in our repository for user to play with. To import this dataset in your postgres databse, under `/testdb` directory, run the following command template:

~~~shell
psql -h <host> -U <user name> -d <local database name where you want to store our example table> < ~/cape/testdb/crime_demonstration.sql
~~~
then run the `capexplain` commands accordingly to explore this example.

## Explaining Outliers

To explain an aggregation outlier use `capexplain explain [OPTIONS]`.

~~~shell
-l ,--log <arg>                - select log level {DEBUG,INFO,WARNING,ERROR} (DEFAULT: ERROR)
--help                         - show this help message
-h ,--host <arg>               - database connection host IP address (DEFAULT: 127.0.0.1)
-u ,--user <arg>               - database connection user (DEFAULT: postgres)
-p ,--password <arg>           - database connection password
-d ,--db <arg>                 - database name (DEFAULT: postgres)
-P ,--port <arg>               - database connection port (DEFAULT: 5432)
--ptable <arg>                 - table storing aggregate regression patterns
--qtable <arg>                 - table storing aggregation query result
--ufile <arg>                  - file storing user question
-o ,--ofile <arg>              - file to write output to
-a ,--aggcolumn <arg>          - column that was input to the aggregation function
~~~
for `explain` option, besides the common options, user should give `--ptable`,the `pattern.{target_table}` and `--qtable`, the `target_table`. Also, we currently only allow user pass question through a `.txt` file, user need to put the question in the following format:

~~~shell
attribute1, attribute 2, attribute3...., direction
value1,value2,value3...., high/low
~~~
please refer to `input.txt` to look at an example.


## Starting the Explanation Explorer GUI

Cape comes with a graphical UI for running queries, selecting outliers of interest, and exploring patterns that are relevant for an outlier and browsing explanations generated by the system. You need to specify the Postgres server to connect to. The explorer can only generate explanations for queries over tables for which patterns have mined beforehand using `capexplain mine`.
Here is our demo video : (https://www.youtube.com/watch?v=gWqhIUrcwz8)

~~~shell
$ capexplain help gui
capexplain gui [OPTIONS]:
	Open the Cape graphical explanation explorer.

SUPPORTED OPTIONS:
-l ,--log <arg>                - select log level {DEBUG,INFO,WARNING,ERROR} (DEFAULT: ERROR)
--help                         - show this help message
-h ,--host <arg>               - database connection host IP address (DEFAULT: 127.0.0.1)
-u ,--user <arg>               - database connection user (DEFAULT: postgres)
-p ,--password <arg>           - database connection password
-d ,--db <arg>                 - database name (DEFAULT: postgres)
-P ,--port <arg>               - database connection port (DEFAULT: 5432)
~~~

For instance, if you run a postgres server locally (default) with user `postgres` (default), password `test`, and database `mydb`, then run:

~~~shell
capexplain gui -p test -d mydb
~~~

# Links

Cape is developed by researchers at Illinois Institute of Technology and Duke University. For more information and publications see the Cape project page [http://www.cs.iit.edu/~dbgroup/projects/cape.html](http://www.cs.iit.edu/~dbgroup/projects/cape.html).
