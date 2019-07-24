# -*- coding: utf-8 -*-
"""
Script for installation in developer mode
"""
import sys
import argparse
import subprocess
import configparser
import os

config = configparser.ConfigParser()
config.read('./meta.cfg')

list_of_requirements = config['condadata']['dependencies'].replace(' ', '').splitlines()[1:]
version = config['metadata']['version']

# Install dependencies
install_command = ['conda', 'install', '-c', 'conda-forge'] + list_of_requirements
subprocess.check_call(install_command)
