"""
Entry point for running survey_bot as a module.

Usage:
    $ python -m survey_bot run --image receipt.jpg --mood happy --email user@email.com
    $ python -m survey_bot direct --url https://survey.example.com --mood happy --email user@email.com
    $ python -m survey_bot version
"""
from .main import run_cli

if __name__ == "__main__":
    run_cli()
