#!/usr/bin/env python3
"""Wrapper: sets API key then runs experiment."""
import os, sys, subprocess

key = 'sk-or-...d093'
env = os.environ.copy()
env['OPENROUTER_API_KEY'] = key

result = subprocess.run(
    [sys.executable, 'run_b4_experiment.py'],
    cwd='.',
    env=env,
    capture_output=False,
    text=True
)
