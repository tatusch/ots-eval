#!/bin/bash
rm -r venv
virtualenv venv
source venv/bin/activate
pip install .
pip install plotly