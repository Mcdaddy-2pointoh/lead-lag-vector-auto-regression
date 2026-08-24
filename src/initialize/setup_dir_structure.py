from pathlib import Path

# function to initialise the directory structure
def create_directory_structure(path: str):
    """
    Function: Accepts a path to a base directory and creates all intermediate folders
    Args:
        path (str | Path): The path to the base directory
    """

    # Convert the string to path
    path = Path(path)
    print(f"INFO: Creating Base Directory")

    # Check if directory exists
    if path.exists():
        print(f"Base Directory path exists: {str(path)}")

    # Create directory + all intermediate parents (safe)
    else:
        path.mkdir(parents=True, exist_ok=True)
        print(f"Base Directory path created at: {str(path)}")

    print(f"Creating Sub Directories")

    # Create sub directories
    sub_directories = [

        # Raw Directories
        "collector_yfin/99_logs",
        "collector_yfin/01_cleaned",
        "collector_yfin/02_processed"
        "collector_yfin/03_summary"
    ]

    # for sub directories 
    # Create all child directories
    for sub_directory in sub_directories:

        # Create a temp path to create sub directories
        temp_path = path / sub_directory

        # Create the sub directory
        if temp_path.exists():
            print(f"Sub Directory path exists: {str(temp_path)}")

        # Create directory + all intermediate parents (safe)
        else:
            temp_path.mkdir(parents=True, exist_ok=True)
            print(f"Sub Directory path created at: {str(temp_path)}")

    # Logger for successful directory
    print(f"Created all directories successfully")
