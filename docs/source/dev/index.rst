Contributing to Puan
====================

All contributions and ideas to further develop Puan are very much appreciated. This document describes the workflow when providing contributions.

.. _issue:

Create an issue
----------------
The first step for any contribution is to create an issue at `Github <https://github.com/ourstudio-se/puan-python/issues>`__. An issue could describe a bug to fix, an enhancement or a general idea for the future of Puan.
Make sure that there isn't an open issue on the same topic, if so, please use the comments field to add additional information.

Basic workflow for code contribution
------------------------------------
:ref:`Create an issue<issue>` for your contribution if you haven't done so already.

Clone the git repository

.. code:: shell

    git clone https://github.com/ourstudio-se/puan-python.git

Change directories

.. code:: shell

    cd puan-python

Checkout a feature branch

.. code:: shell

    git fetch origin                                         
    git checkout <issue>

Make sure your developed code is documented and tested with tests included in the code base. Also make sure that the entire test-suite passes.

Commit and push your changes to the feature branch. In case of conflicts with main, please rebase your feature branch. 

.. code:: shell

    git fetch origin
    git rebase main

Create a pull request (PR) in Github.

Your code will be reviewed before merged into to main branch. The purpose of the review is to increase quality of the code and spread knowledge about the code base. Review comments should always be friendly and without critizism,
by helping eachother we will get the best results and that is what we aim for. When the code is reviewed and updated accordingly, it must be approved by atleast one other developer before it can be merged to main. 
