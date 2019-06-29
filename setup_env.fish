#! /usr/local/bin/fish
virtualenv -p python3 env
source env/bin/activate.fish
pip install -r requirements.txt