import os
import glob
from typing import List, Tuple

from ..dependency_manager import S4ToolLogger


class ImageFromFolder:
    """
    Load images from a local folder and output each image with its original dimensions.
    Supports common image formats (PNG, JPG, JPEG, BMP, GIF, TIFF, WEBP).
    Preserves alpha channel and original dimensions.
    Each image is output separately to maintain original size.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Enter folder path containing images"
                }),
                "recursive": ("BOOLEAN", {"default": False}),
                "sort_method": (["filename", "modified", "size"], {"default": "filename"}),
                "reverse_order": ("BOOLEAN", {"default": False}),
                "max_images": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1})
            }
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("count", "filepaths")
    FUNCTION = "load_images_from_folder"
    CATEGORY = "💀S4Tool"
    
    # Supported image extensions
    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif", ".webp"}

    def _get_image_files(self, folder_path: str, recursive: bool = False) -> List[str]:
        """Get all image files from the specified folder."""
        if not os.path.exists(folder_path):
            S4ToolLogger.warning("ImageFromFolder", f"Folder path does not exist: {folder_path}")
            return []
        
        if not os.path.isdir(folder_path):
            S4ToolLogger.warning("ImageFromFolder", f"Path is not a directory: {folder_path}")
            return []

        image_files = []
        
        if recursive:
            # Search recursively using glob pattern
            for ext in self.SUPPORTED_EXTENSIONS:
                pattern = os.path.join(folder_path, "**", f"*{ext}")
                files = glob.glob(pattern, recursive=True)
                image_files.extend(files)
                
                # Also search for uppercase extensions
                pattern_upper = os.path.join(folder_path, "**", f"*{ext.upper()}")
                files_upper = glob.glob(pattern_upper, recursive=True)
                image_files.extend(files_upper)
        else:
            # Search only in the specified folder
            for ext in self.SUPPORTED_EXTENSIONS:
                pattern = os.path.join(folder_path, f"*{ext}")
                files = glob.glob(pattern)
                image_files.extend(files)
                
                # Also search for uppercase extensions
                pattern_upper = os.path.join(folder_path, f"*{ext.upper()}")
                files_upper = glob.glob(pattern_upper)
                image_files.extend(files_upper)

        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for file in image_files:
            if file not in seen:
                seen.add(file)
                unique_files.append(file)

        return unique_files

    def _sort_files(self, files: List[str], sort_method: str, reverse: bool) -> List[str]:
        """Sort files based on the specified method."""
        if not files:
            return files

        try:
            if sort_method == "filename":
                sorted_files = sorted(files, key=lambda x: os.path.basename(x).lower(), reverse=reverse)
            elif sort_method == "modified":
                sorted_files = sorted(files, key=lambda x: os.path.getmtime(x), reverse=reverse)
            elif sort_method == "size":
                sorted_files = sorted(files, key=lambda x: os.path.getsize(x), reverse=reverse)
            else:
                sorted_files = files

            return sorted_files
        except Exception as e:
            S4ToolLogger.warning("ImageFromFolder", f"Failed to sort files: {str(e)}")
            return files


    def _is_valid_image_file(self, file_path: str) -> bool:
        """Quick validation of image file without loading it."""
        try:
            # Check if file exists and is readable
            if not os.path.isfile(file_path) or not os.access(file_path, os.R_OK):
                return False
            
            # Check file extension
            _, ext = os.path.splitext(file_path.lower())
            if ext not in self.SUPPORTED_EXTENSIONS:
                return False
                
            # Quick file size check (avoid 0-byte files)
            if os.path.getsize(file_path) == 0:
                return False
                
            return True
            
        except Exception as e:
            S4ToolLogger.warning("ImageFromFolder", f"File validation failed for {file_path}: {str(e)}")
            return False
    

    def load_images_from_folder(self, folder_path: str, recursive: bool = False, 
                               sort_method: str = "filename", reverse_order: bool = False, 
                               max_images: int = 0) -> Tuple[int, List[str]]:
        """Load images from folder and return as batch."""
        try:
            folder_path = folder_path.strip()
            if not folder_path:
                # Return empty results when folder path is empty
                return (0, [])

            # Expand environment variables and user home
            expanded_path = os.path.expandvars(os.path.expanduser(folder_path))
            
            S4ToolLogger.info("ImageFromFolder", f"Loading images from: {expanded_path}")
            S4ToolLogger.info("ImageFromFolder", f"Recursive: {recursive}, Sort: {sort_method}, Reverse: {reverse_order}, Max: {max_images}")

            # Get image files
            image_files = self._get_image_files(expanded_path, recursive)
            if not image_files:
                S4ToolLogger.warning("ImageFromFolder", f"No image files found in: {expanded_path}")
                return (0, [])

            # Sort files
            sorted_files = self._sort_files(image_files, sort_method, reverse_order)
            
            # Limit number of images if specified
            if max_images > 0 and len(sorted_files) > max_images:
                sorted_files = sorted_files[:max_images]

            S4ToolLogger.info("ImageFromFolder", f"Processing {len(sorted_files)} image files")

            # Validate image files and collect valid paths
            valid_filepaths = []
            
            for file_path in sorted_files:
                if self._is_valid_image_file(file_path):
                    valid_filepaths.append(file_path)
                    filename = os.path.basename(file_path)
                    S4ToolLogger.info("ImageFromFolder", f"Found valid image: {filename}")

            count = len(valid_filepaths)
            
            if count == 0:
                S4ToolLogger.warning("ImageFromFolder", "No valid images found")
                return (0, [])
                
            S4ToolLogger.success("ImageFromFolder", f"Found {count} valid image files")
            return (count, valid_filepaths)

        except Exception as e:
            S4ToolLogger.error("ImageFromFolder", f"Error loading images from folder: {str(e)}")
            return (0, [])
