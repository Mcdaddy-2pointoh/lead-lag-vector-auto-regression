# External Imports
from pathlib import Path
import yaml
from datetime import datetime
import pandas as pd
import re
import os
import json
import joblib

# Local Imports
from src.utils.text_operators import format_error_text, format_info_text, format_success_text


# Config Loading Segment
class ConfigLoadError(Exception):
    """Raised when a YAML configuration file cannot be loaded."""

class DictionaryLoadError(Exception):
    """Raised when loading a dictionary from JSON fails."""

class DictionarySaveError(Exception):
    """Raised when saving a dictionary as JSON fails."""

class TextSaveError(Exception):
    """Raised when saving text fails."""

class TextLoadError(Exception):
    """Raised when loading text fails."""

# Model Saving and Loading Segment
class ModelSaveError(Exception):
    """Raised when saving a machine learning model fails."""

class ModelLoadError(Exception):
    """Raised when loading a machine learning model fails."""

def load_yaml(path: str | Path) -> dict:
    """
    Function: Load and return the contents of a YAML file.
    Args:
        path (str | Path): The file path poining to the YAML file
    Returns:
        Dictionary containing the contents of the YAML file
    """
    path = Path(path)

    # Try to load the yaml file using safe load
    try:
        with path.open("r") as f:
            return yaml.safe_load(f)

    # Raise exceptions
    except FileNotFoundError as e:
        raise ConfigLoadError(f"Config file not found: {path}") from e

    except PermissionError as e:
        raise ConfigLoadError(f"Permission denied reading config file: {path}") from e

    except yaml.YAMLError as e:
        raise ConfigLoadError(f"Invalid YAML syntax in config file: {path}") from e

    except Exception as e:
        raise ConfigLoadError(
            f"Unexpected error while loading config file: {path}"
        ) from e


# Data frame saving segment
class DataFrameSaveError(Exception):
    """Raised when saving a DataFrame fails."""

def save_dataframe(
    df: pd.DataFrame,
    directory: str | Path,
    file_name: str,
    file_format: str = "csv",
    versioned: bool = False,
    timestamp_format: str = "%Y%m%d_%H%M%S",
    suffix: str | None = None,
    save_index: bool = False,
    **kwargs,
) -> Path:
    """
    Save a pandas DataFrame to disk with optional timestamp versioning.
    Automatically creates the directory path if it does not exist.

    Args:
        df (pd.DataFrame): pandas DataFrame to save
        directory (str | Path): directory path to save the file
        file_name (str): base file name without extension
        file_format (str): one of {'csv', 'excel', 'parquet'}
        versioned (bool): if True, append timestamp to filename
        timestamp_format (str): datetime format for versioning
        save_index (bool): Save data with index
        **kwargs: passed to pandas writer (e.g. index=False)

    Returns:
        Path to the saved file
    """

    try:
        # Validate the dataframe and the directory
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        directory = Path(directory)

        if directory.exists() and not directory.is_dir():
            raise NotADirectoryError(
                f"Path exists but is not a directory: {directory}"
            )

        # Create directory tree if missing
        directory.mkdir(parents=True, exist_ok=True)

        file_format = file_format.lower()
        allowed_formats = {"csv", "excel", "parquet"}

        if file_format not in allowed_formats:
            raise ValueError(
                f"file_format must be one of {allowed_formats}, got '{file_format}'"
            )

        # Construct suitable file name
        timestamp = (
            datetime.now().strftime(timestamp_format) if versioned else None
        )

        extension_map = {
            "csv": ".csv",
            "excel": ".xlsx",
            "parquet": ".parquet",
        }

        name_parts = [file_name]

        if suffix:
            name_parts.append(suffix)

        if timestamp:
            name_parts.append(timestamp)

        file_path = (
            directory
            / f"{'_'.join(name_parts)}{extension_map[file_format]}"
        )

        # Save in specified format
        if file_format == "csv":
            df.to_csv(file_path, index=save_index, **kwargs)

        elif file_format == "excel":
            df.to_excel(file_path, index=save_index, **kwargs)

        elif file_format == "parquet":
            df.to_parquet(file_path, index=save_index, **kwargs)

        return file_path

    except Exception as e:
        raise DataFrameSaveError(
            f"Failed to save DataFrame '{file_name}' "
            f"as {file_format} in directory {directory}"
        ) from e


# DataFrame Loader Segment
class DataFrameLoadError(Exception):
    """Raised when loading a DataFrame fails."""

def load_latest_dataframe(
    directory: str | Path,
    file_name: str,
    file_format: str = "csv",
    timestamp_format: str = "%Y%m%d_%H%M%S",
    suffix: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Load the latest timestamp-versioned DataFrame from disk.

    Expected filename format:
        file_name_<timestamp>.<extension>

    Args:
        directory: directory where files are stored
        file_name: base file name (without timestamp or extension)
        file_format: one of {'csv', 'excel', 'parquet'}
        timestamp_format: datetime format used in filenames
        **kwargs: passed to pandas read_* functions

    Returns:
        pandas DataFrame
    """

    try:
        directory = Path(directory)

        # Validate extension, filepath
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {directory}")

        extension_map = {
            "csv": ".csv",
            "excel": ".xlsx",
            "parquet": ".parquet",
        }

        file_format = file_format.lower()
        if file_format not in extension_map:
            raise ValueError(f"Unsupported file format: {file_format}")

        extension = extension_map[file_format]

        # Match to find files with correct time stamp
        if suffix is None:
            pattern = re.compile(
                rf"^{re.escape(file_name)}_(?P<ts>.+){re.escape(extension)}$"
            )
        else:
            pattern = re.compile(
                rf"^{re.escape(file_name)}_{re.escape(suffix)}_(?P<ts>.+){re.escape(extension)}$"
            )

        candidates: list[tuple[datetime, Path]] = []

        for file in directory.iterdir():
            match = pattern.match(file.name)
            if not match:
                continue

            timestamp_str = match.group("ts")

            try:
                timestamp = datetime.strptime(timestamp_str, timestamp_format)
            except ValueError:
                # Skip files with malformed timestamps
                continue

            candidates.append((timestamp, file))

        # Set the latest file variable outside the if scope
        latest_file = None

        # If no candidates check if the file name exists
        if not candidates:
            files_in_dir = os.listdir(directory)

            # Select the file with plain name
            plain_file = f"{file_name}{extension}"
            if plain_file in files_in_dir:
                latest_file = directory / plain_file

            # Else raise error
            else:
                raise FileNotFoundError(
                    f"No versioned files found for '{file_name}' in {directory}"
                )

        # Pick latest candidate
        else:
            # Select the latest file
            latest_file = max(candidates, key=lambda x: x[0])[1]

        # Load the latest file
        if file_format == "csv":
            return pd.read_csv(latest_file, **kwargs)

        elif file_format == "excel":
            return pd.read_excel(latest_file, **kwargs)

        elif file_format == "parquet":
            return pd.read_parquet(latest_file, **kwargs)

    except Exception as e:
        raise DataFrameLoadError(
            f"Failed to load latest DataFrame '{file_name}' from {directory}"
        ) from e
    
def create_directory_structure(path: str, verbose: bool = False):
    """
    Function: Accepts a path to a base directory and creates all intermediate folders
    Args:
        path (str | Path): The path to the base directory
    """

    # Convert the string to path
    path = Path(path)
    if verbose:
        print(format_info_text(f"Creating Base Directory"))

    # Check if directory exists
    if path.exists():
        if verbose:
            print(format_info_text(f"Base Directory path exists: {str(path)}"))

    # Create directory + all intermediate parents (safe)
    else:
        path.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(format_success_text(f"Base Directory path created at: {str(path)}"))

    if verbose:
        print(format_info_text(f"Creating Sub Directories"))

    # Create sub directories
    sub_directories = [

        # Raw Directories
        "01_raw",
        "02_processed",
        "03_analysis"
    ]

    # for sub directories 
    # Create all child directories
    for sub_directory in sub_directories:

        # Create a temp path to create sub directories
        temp_path = path / sub_directory

        # Create the sub directory
        if temp_path.exists():
            if verbose:
                print(format_info_text(f"Sub Directory path exists: {str(temp_path)}"))

        # Create directory + all intermediate parents (safe)
        else:
            temp_path.mkdir(parents=True, exist_ok=True)

            if verbose:
                print(format_success_text(f"Sub Directory path created at: {str(temp_path)}"))

        if verbose:
            print()

    # Logger for successful directory
    print(format_success_text(f"Created all directories successfully!"))
    print()

def load_latest_dictionary(
    directory: str | Path,
    file_name: str,
    timestamp_format: str = "%Y%m%d_%H%M%S",
    suffix: str | None = None,
) -> dict:
    """
    Load the latest timestamp-versioned dictionary from disk.
    """

    try:

        directory = Path(directory)

        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {directory}")

        extension = ".json"

        if suffix is None:
            pattern = re.compile(
                rf"^{re.escape(file_name)}_(?P<ts>.+){re.escape(extension)}$"
            )
        else:
            pattern = re.compile(
                rf"^{re.escape(file_name)}_{re.escape(suffix)}_(?P<ts>.+){re.escape(extension)}$"
            )

        candidates: list[tuple[datetime, Path]] = []

        for file in directory.iterdir():

            match = pattern.match(file.name)

            if not match:
                continue

            timestamp_str = match.group("ts")

            try:
                timestamp = datetime.strptime(timestamp_str, timestamp_format)
            except ValueError:
                continue

            candidates.append((timestamp, file))

        latest_file = None

        if not candidates:

            files_in_dir = os.listdir(directory)
            plain_file = f"{file_name}{extension}"

            if plain_file in files_in_dir:
                latest_file = directory / plain_file
            else:
                raise FileNotFoundError(
                    f"No versioned JSON files found for '{file_name}' in {directory}"
                )

        else:
            latest_file = max(candidates, key=lambda x: x[0])[1]

        with latest_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as e:
        print(format_error_text(
            f"Failed to load latest dictionary '{file_name}' from {directory}"
        ))
        raise DictionaryLoadError(
            f"Failed to load latest dictionary '{file_name}' from {directory}"
        ) from e
    
def save_dictionary(
    data: dict,
    directory: str | Path,
    file_name: str,
    versioned: bool = False,
    timestamp_format: str = "%Y%m%d_%H%M%S",
    suffix: str | None = None,
    indent: int | None = 4,
    ensure_ascii: bool = False,
) -> Path:
    """
    Save a dictionary to disk as a JSON file with optional timestamp versioning.
    Automatically creates the directory path if it does not exist.
    """

    try:

        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary")

        directory = Path(directory)

        if directory.exists() and not directory.is_dir():
            raise NotADirectoryError(
                f"Path exists but is not a directory: {directory}"
            )

        directory.mkdir(parents=True, exist_ok=True)

        timestamp = (
            datetime.now().strftime(timestamp_format) if versioned else None
        )

        name_parts = [file_name]

        if suffix:
            name_parts.append(suffix)

        if timestamp:
            name_parts.append(timestamp)

        file_path = directory / f"{'_'.join(name_parts)}.json"

        with file_path.open("w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                indent=indent,
                ensure_ascii=ensure_ascii,
                default=str,
            )

        return file_path

    except Exception as e:
        print(format_error_text(
            f"Failed to save dictionary '{file_name}' as JSON in directory {directory}"
        ))
        raise DictionarySaveError(
            f"Failed to save dictionary '{file_name}' as JSON in directory {directory}"
        ) from e

def save_text(
    text: str,
    directory: str | Path,
    file_name: str,
    versioned: bool = False,
    timestamp_format: str = "%Y%m%d_%H%M%S",
    suffix: str | None = None,
    encoding: str = "utf-8",
) -> Path:
    """
    Save text to a .txt file with optional timestamp versioning.

    Args:
        text (str): Text content to save.
        directory (str | Path): Directory where the file will be saved.
        file_name (str): Base file name without extension.
        versioned (bool): If True, append timestamp to filename.
        timestamp_format (str): Datetime format used for versioning.
        suffix (str | None): Optional suffix appended before timestamp.
        encoding (str): Text encoding.

    Returns:
        Path: Path to the saved file.
    """

    try:

        # Validate text
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        # Convert directory to Path
        directory = Path(directory)

        # Validate directory
        if directory.exists() and not directory.is_dir():
            raise NotADirectoryError(
                f"Path exists but is not a directory: {directory}"
            )

        # Create directory tree if missing
        directory.mkdir(parents=True, exist_ok=True)

        # Generate timestamp
        timestamp = (
            datetime.now().strftime(timestamp_format)
            if versioned
            else None
        )

        # Construct filename
        name_parts = [file_name]

        if suffix:
            name_parts.append(suffix)

        if timestamp:
            name_parts.append(timestamp)

        file_path = directory / f"{'_'.join(name_parts)}.txt"

        # Save text
        with file_path.open("w", encoding=encoding) as f:
            f.write(text)

        return file_path

    except Exception as e:
        print(
            format_error_text(
                f"Failed to save text '{file_name}' in directory {directory}"
            )
        )

        raise TextSaveError(
            f"Failed to save text '{file_name}' in directory {directory}"
        ) from e

def load_latest_text(
    directory: str | Path,
    file_name: str,
    timestamp_format: str = "%Y%m%d_%H%M%S",
    suffix: str | None = None,
    encoding: str = "utf-8",
) -> str:
    """
    Load the latest timestamp-versioned text file from disk.

    Expected filename format:
        file_name_<timestamp>.txt

    Args:
        directory (str | Path): Directory containing the files.
        file_name (str): Base file name without timestamp or extension.
        timestamp_format (str): Datetime format used in filenames.
        suffix (str | None): Optional suffix used in filename matching.
        encoding (str): Text encoding.

    Returns:
        str: Contents of the latest text file.
    """

    try:

        directory = Path(directory)

        # Validate directory
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(
                f"Directory does not exist: {directory}"
            )

        extension = ".txt"

        # Build filename pattern
        if suffix is None:
            pattern = re.compile(
                rf"^{re.escape(file_name)}_(?P<ts>.+){re.escape(extension)}$"
            )
        else:
            pattern = re.compile(
                rf"^{re.escape(file_name)}_{re.escape(suffix)}_(?P<ts>.+){re.escape(extension)}$"
            )

        candidates: list[tuple[datetime, Path]] = []

        # Find matching files
        for file in directory.iterdir():

            match = pattern.match(file.name)

            if not match:
                continue

            timestamp_str = match.group("ts")

            try:
                timestamp = datetime.strptime(
                    timestamp_str,
                    timestamp_format
                )
            except ValueError:
                continue

            candidates.append((timestamp, file))

        # No versioned files found
        if not candidates:

            plain_file = f"{file_name}{extension}"

            if plain_file in os.listdir(directory):
                latest_file = directory / plain_file

            else:
                raise FileNotFoundError(
                    f"No versioned text files found for "
                    f"'{file_name}' in {directory}"
                )

        # Select latest version
        else:
            latest_file = max(
                candidates,
                key=lambda x: x[0]
            )[1]

        # Load text
        with latest_file.open("r", encoding=encoding) as f:
            return f.read()

    except Exception as e:

        print(
            format_error_text(
                f"Failed to load latest text "
                f"'{file_name}' from {directory}"
            )
        )

        raise TextLoadError(
            f"Failed to load latest text "
            f"'{file_name}' from {directory}"
        ) from e

def save_model(
    model_object: object,
    directory: str | Path,
    file_name: str,
    versioned: bool = False,
    timestamp_format: str = "%Y%m%d_%H%M%S",
    suffix: str | None = None,
) -> Path:
    """
    Save a trained model object (e.g., statsmodels VAR, scikit-learn estimator) to disk using joblib.
    Automatically creates the directory path if it does not exist.

    Args:
        model_object (object): Model object to serialize.
        directory (str | Path): Directory path to save the file.
        file_name (str): Base file name without extension.
        versioned (bool): If True, append timestamp to filename.
        timestamp_format (str): Datetime format for versioning.
        suffix (str | None): Optional suffix appended before timestamp.

    Returns:
        Path: Path to the saved model file.
    """
    import joblib

    try:
        if model_object is None:
            raise ValueError("model_object cannot be None")

        directory = Path(directory)

        if directory.exists() and not directory.is_dir():
            raise NotADirectoryError(
                f"Path exists but is not a directory: {directory}"
            )

        directory.mkdir(parents=True, exist_ok=True)

        timestamp = (
            datetime.now().strftime(timestamp_format) if versioned else None
        )

        name_parts = [file_name]

        if suffix:
            name_parts.append(suffix)

        if timestamp:
            name_parts.append(timestamp)

        file_path = directory / f"{'_'.join(name_parts)}.joblib"

        # Serialize model
        joblib.dump(model_object, file_path)

        return file_path

    except Exception as e:
        print(
            format_error_text(
                f"Failed to save model '{file_name}' in directory {directory}"
            )
        )
        raise ModelSaveError(
            f"Failed to save model '{file_name}' in directory {directory}"
        ) from e


def load_latest_model(
    directory: str | Path,
    file_name: str,
    timestamp_format: str = "%Y%m%d_%H%M%S",
    suffix: str | None = None,
) -> object:
    """
    Load the latest timestamp-versioned serialized model (.joblib) from disk.

    Args:
        directory (str | Path): Directory containing the model files.
        file_name (str): Base file name without timestamp or extension.
        timestamp_format (str): Datetime format used in filenames.
        suffix (str | None): Optional suffix used in filename matching.

    Returns:
        object: Deserialized model object.
    """

    try:
        directory = Path(directory)

        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(
                f"Directory does not exist: {directory}"
            )

        extension = ".joblib"

        # Build filename pattern
        if suffix is None:
            pattern = re.compile(
                rf"^{re.escape(file_name)}_(?P<ts>.+){re.escape(extension)}$"
            )
        else:
            pattern = re.compile(
                rf"^{re.escape(file_name)}_{re.escape(suffix)}_(?P<ts>.+){re.escape(extension)}$"
            )

        candidates: list[tuple[datetime, Path]] = []

        # Find matching files
        for file in directory.iterdir():
            match = pattern.match(file.name)
            if not match:
                continue

            timestamp_str = match.group("ts")

            try:
                timestamp = datetime.strptime(
                    timestamp_str,
                    timestamp_format
                )
            except ValueError:
                continue

            candidates.append((timestamp, file))

        # Handle non-versioned fallback or missing file
        if not candidates:
            plain_file = f"{file_name}{extension}"

            if plain_file in os.listdir(directory):
                latest_file = directory / plain_file
            else:
                raise FileNotFoundError(
                    f"No versioned model files found for '{file_name}' in {directory}"
                )

        # Select latest version
        else:
            latest_file = max(candidates, key=lambda x: x[0])[1]

        # Deserialize and return model
        return joblib.load(latest_file)

    except Exception as e:
        print(
            format_error_text(
                f"Failed to load latest model '{file_name}' from {directory}"
            )
        )
        raise ModelLoadError(
            f"Failed to load latest model '{file_name}' from {directory}"
        ) from e