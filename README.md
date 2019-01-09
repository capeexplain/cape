# Cape

**Cape** () is a system for explaining outliers (surprisingly low or high) aggregation function results for group-by queries in SQL. The user provides the system with a query and a surprising outcome for this query. Cape uses patterns discovered the describe trends that hold in the data in aggregation to explain such outliers.

# Installation

Cape requires python 3. Install with pip:

~~~shell
pip install capexplain
~~~

Install from github

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

Cape currently only supports PostgreSQL as a backend database. To use Cape to explain an aggregation outlier, you first have to let cape find patterns for the table over which are you are aggregating. This an offline step that only has to be executed once for each table (unless you want to rerun pattern mining with different parameter settings).

## Mining Patterns

Use `capexplain mine [OPTIONS]` to mine patterns. Cape will store the discovered patterns in the database.

## Explaining Outliers

To explain an aggregation outlier use `capexplain explain [OPTIONS]`.


# Links

Cape is developed by researchers at Illinois Institute of Technology and Duke University. For more information and publications see the Cape project page [http://www.cs.iit.edu/~dbgroup/projects/cape.html](http://www.cs.iit.edu/~dbgroup/projects/cape.html).
