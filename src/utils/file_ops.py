import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import xml.etree.ElementTree as ET

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FileOperations:

    @staticmethod
    def read_text(filepath: str, encoding: str = "utf-8") -> Optional[str]:
        """
        Read text file safely

        Args:
            filepath: Path to file
            encoding: File encoding

        Returns:
            File content or None if error
        """
        try:
            with open(filepath, "r", encoding=encoding) as f:
                content = f.read()
            logger.debug("Read text file: %s", filepath)
            return content
        except Exception as e:
            logger.error("Error reading file %s: %s", filepath, str(e))
            return None

    @staticmethod
    def write_text(
        filepath: str, content: str, encoding: str = "utf-8", create_dirs: bool = True
    ) -> bool:
        """
        Write text file safely

        Args:
            filepath: Path to file
            content: Content to write
            encoding: File encoding
            create_dirs: Create parent directories if needed

        Returns:
            True if successful
        """
        try:
            path = Path(filepath)
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, "w", encoding=encoding) as f:
                f.write(content)

            logger.debug("Wrote text file: %s", filepath)
            return True
        except Exception as e:
            logger.error("Error writing file %s: %s", filepath, str(e))
            return False

    @staticmethod
    def read_json(filepath: str) -> Optional[Dict[str, Any]]:
        """
        Read JSON file safely

        Args:
            filepath: Path to JSON file

        Returns:
            Parsed JSON or None if error
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.debug("Read JSON file: %s", filepath)
            return data
        except json.JSONDecodeError as e:
            logger.error("JSON decode error in %s: %s", filepath, str(e))
            return None
        except Exception as e:
            logger.error("Error reading JSON file %s: %s", filepath, str(e))
            return None

    @staticmethod
    def write_json(
        filepath: str, data: Dict[str, Any], indent: int = 2, create_dirs: bool = True
    ) -> bool:
        """
        Write JSON file safely

        Args:
            filepath: Path to JSON file
            data: Data to write
            indent: JSON indentation
            create_dirs: Create parent directories if needed

        Returns:
            True if successful
        """
        try:
            path = Path(filepath)
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)

            logger.debug("Wrote JSON file: %s", filepath)
            return True
        except Exception as e:
            logger.error("Error writing JSON file %s: %s", filepath, str(e))
            return False

    @staticmethod
    def read_pickle(filepath: str) -> Optional[Any]:
        """
        Read pickle file safely

        Args:
            filepath: Path to pickle file

        Returns:
            Unpickled object or None if error
        """
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            logger.debug("Read pickle file: %s", filepath)
            return data
        except Exception as e:
            logger.error("Error reading pickle file %s: %s", filepath, str(e))
            return None

    @staticmethod
    def write_pickle(filepath: str, data: Any, create_dirs: bool = True) -> bool:
        """
        Write pickle file safely

        Args:
            filepath: Path to pickle file
            data: Data to pickle
            create_dirs: Create parent directories if needed

        Returns:
            True if successful
        """
        try:
            path = Path(filepath)
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, "wb") as f:
                pickle.dump(data, f)

            logger.debug("Wrote pickle file: %s", filepath)
            return True
        except Exception as e:
            logger.error("Error writing pickle file %s: %s", filepath, str(e))
            return False

    @staticmethod
    def read_xml(filepath: str) -> Optional[ET.Element]:
        """
        Read XML file safely

        Args:
            filepath: Path to XML file

        Returns:
            XML root element or None if error
        """
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            logger.debug("Read XML file: %s", filepath)
            return root
        except Exception as e:
            logger.error("Error reading XML file %s: %s", filepath, str(e))
            return None

    @staticmethod
    def write_xml(filepath: str, root: ET.Element, create_dirs: bool = True) -> bool:
        """
        Write XML file safely

        Args:
            filepath: Path to XML file
            root: XML root element
            create_dirs: Create parent directories if needed

        Returns:
            True if successful
        """
        try:
            path = Path(filepath)
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)

            tree = ET.ElementTree(root)
            tree.write(filepath, encoding="utf-8", xml_declaration=True)

            logger.debug("Wrote XML file: %s", filepath)
            return True
        except Exception as e:
            logger.error("Error writing XML file %s: %s", filepath, str(e))
            return False

    @staticmethod
    def ensure_dir(dirpath: str) -> bool:
        """
        Ensure directory exists

        Args:
            dirpath: Directory path

        Returns:
            True if successful
        """
        try:
            Path(dirpath).mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured directory exists: %s", dirpath)
            return True
        except Exception as e:
            logger.error("Error creating directory %s: %s", dirpath, str(e))
            return False

    @staticmethod
    def list_files(
        dirpath: str, pattern: str = "*", recursive: bool = False
    ) -> list[Path]:
        """
        List files in directory

        Args:
            dirpath: Directory path
            pattern: File pattern (e.g., '*.py')
            recursive: Search recursively

        Returns:
            List of file paths
        """
        try:
            path = Path(dirpath)
            if recursive:
                files = list(path.rglob(pattern))
            else:
                files = list(path.glob(pattern))

            logger.debug("Listed %d files in %s", len(files), dirpath)
            return files
        except Exception as e:
            logger.error("Error listing files in %s: %s", dirpath, str(e))
            return []


# Convenience functions for backward compatibility
def read_file(filepath: str) -> str:
    """Read text file (convenience function)"""
    content = FileOperations.read_text(filepath)
    if content is None:
        raise FileNotFoundError(f"Could not read file: {filepath}")
    return content


def write_file(filepath: str, content: str) -> None:
    """Write text file (convenience function)"""
    success = FileOperations.write_text(filepath, content)
    if not success:
        raise IOError(f"Could not write file: {filepath}")
