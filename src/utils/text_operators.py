class bcolors:
    OKGREEN = '\033[92m'
    INFO = '\033[38;5;195m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    WARN = '\033[38;2;255;165;0m'

def format_error_text(text: str) -> str:
    """
    Function: Produces a red bold text with the inpu text
    Args:
        text (str): The text to color as a ERROR
    """

    return f"""{bcolors.FAIL}{bcolors.BOLD}ERROR: {str(text)}{bcolors.ENDC}"""

def format_success_text(text: str) -> str:
    """
    Function: Produces a green bold text with the input text
    Args:
        text (str): The text to color as a SUCCESS
    """

    return f"""{bcolors.BOLD}{bcolors.OKGREEN}SUCCESS: {str(text)}{bcolors.ENDC}"""

def format_info_text(text: str) -> str:
    """
    Function: Produces a yellow text with the input text
    Args:
        text (str): The text to color as a INFO
    """

    return f"""{bcolors.INFO}INFO: {str(text)}{bcolors.ENDC}"""

def format_warn_text(text: str) -> str:
    """
    Function: Produces a orange text with the input text
    Args:
        text (str): The text to color as a WARN
    """

    return f"""{bcolors.BOLD}{bcolors.WARN}WARN: {str(text)}{bcolors.ENDC}"""
