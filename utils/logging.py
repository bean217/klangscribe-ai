#
# file: utils/logging.py
# desc: Provides utility functions for logging and error handling in the KlangScribe AI project.
# auth: Benjamin Piro (brp8396@rit.edu)
# date: 2026-02-28
#


import traceback


def format_exception(e: Exception) -> str:
    """
    Formats an exception into a readable string, including the stack trace.
    
    Args:
        - e (Exception): The exception to be formatted.

    Returns:
        - str: A formatted string representation of the exception, including the stack trace.
    """
    return f"{str(e)}\n{traceback.format_exc()}"