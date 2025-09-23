"""
S4Tool Image Dependency Manager and Unified Logging System
Production Quality Above All - No Feature Fallbacks
"""
import sys
import importlib
from typing import Dict, List, Tuple, Optional
import warnings

# ANSI color codes for terminal output
class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

class S4ToolLogger:
    """Unified logging system for S4Tool nodes"""
    
    PREFIX = "S4Tool-Image"
    
    @staticmethod
    def _format_message(level: str, node: str, message: str) -> str:
        """Format log message with consistent style"""
        if level == "ERROR":
            icon = "❌"
            color = Colors.RED
        elif level == "WARNING": 
            icon = "⚠️"
            color = Colors.YELLOW
        elif level == "SUCCESS":
            icon = "✅"
            color = Colors.GREEN
        elif level == "INFO":
            icon = "ℹ️"
            color = Colors.CYAN
        elif level == "DEBUG":
            icon = "🔧"
            color = Colors.DIM
        else:
            icon = "📝"
            color = Colors.WHITE
        
        return f"{color}{Colors.BOLD}[{S4ToolLogger.PREFIX}]{Colors.RESET} {icon} {color}{node}:{Colors.RESET} {message}"
    
    @staticmethod
    def error(node: str, message: str):
        """Log error message"""
        print(S4ToolLogger._format_message("ERROR", node, message))
    
    @staticmethod
    def warning(node: str, message: str):
        """Log warning message"""  
        print(S4ToolLogger._format_message("WARNING", node, message))
    
    @staticmethod
    def success(node: str, message: str):
        """Log success message"""
        print(S4ToolLogger._format_message("SUCCESS", node, message))
    
    @staticmethod
    def info(node: str, message: str):
        """Log info message"""
        print(S4ToolLogger._format_message("INFO", node, message))
    
    @staticmethod
    def debug(node: str, message: str):
        """Log debug message"""
        print(S4ToolLogger._format_message("DEBUG", node, message))

class DependencyManager:
    """Manages dependency checking for S4Tool nodes - Production Quality"""
    
    # All dependencies are REQUIRED for production quality
    REQUIRED_DEPENDENCIES = {
        'torch': {
            'package': 'torch',
            'description': 'PyTorch - Core tensor operations and ComfyUI compatibility',
            'install_cmd': 'pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu',
            'features': ['ComfyUI Integration', 'Tensor Operations', 'GPU Acceleration']
        },
        'PIL': {
            'package': 'Pillow',
            'import_name': 'PIL',
            'description': 'Pillow - Professional image processing',
            'install_cmd': 'pip install Pillow>=8.0.0',
            'features': ['Image Processing', 'Format Support', 'Alpha Compositing']
        },
        'numpy': {
            'package': 'numpy',
            'description': 'NumPy - High-performance numerical computing',
            'install_cmd': 'pip install numpy>=1.21.0',
            'features': ['Mathematical Operations', 'Array Processing', 'Performance']
        },
        'sklearn': {
            'package': 'scikit-learn',
            'import_name': 'sklearn',
            'description': 'Scikit-learn - Professional K-Means color clustering',
            'install_cmd': 'pip install scikit-learn>=1.0.0',
            'features': ['K-Means Color Extraction', 'Advanced Color Analysis', 'Professional Clustering']
        },
        'skimage': {
            'package': 'scikit-image',
            'import_name': 'skimage',
            'description': 'Scikit-image - Production-grade image transformations',
            'install_cmd': 'pip install scikit-image>=0.19.0',
            'features': ['High-Quality Image Transforms', 'Professional Overlays', 'Advanced Warping']
        },
        'scipy': {
            'package': 'scipy',
            'description': 'SciPy - Advanced image filtering and scientific computing',
            'install_cmd': 'pip install scipy>=1.8.0',
            'features': ['Advanced Edge Feathering', 'Alpha Matting', 'Scientific Processing']
        },
        'cv2': {
            'package': 'opencv-contrib-python',
            'import_name': 'cv2',
            'description': 'OpenCV - Professional image and video processing',
            'install_cmd': 'pip install opencv-contrib-python>=4.5.0',
            'features': ['Guided Filter', 'Advanced Edge Processing', 'Video Processing']
        },
        'transformers': {
            'package': 'transformers',
            'description': 'HuggingFace Transformers - AI model loading and inference',
            'install_cmd': 'pip install transformers>=4.20.0',
            'features': ['RMBG-2.0 Background Removal', 'AI Model Integration', 'Professional AI Processing']
        },
        'huggingface_hub': {
            'package': 'huggingface_hub',
            'description': 'HuggingFace Hub - Automatic model downloading and management',
            'install_cmd': 'pip install huggingface_hub>=0.15.0',
            'features': ['Automatic Model Downloads', 'Model Management', 'Version Control']
        },
        'safetensors': {
            'package': 'safetensors',
            'description': 'SafeTensors - Secure and fast model weight loading',
            'install_cmd': 'pip install safetensors>=0.3.0',
            'features': ['RMBG Model Weights', 'Secure Loading', 'Performance Optimization']
        },
        'segment_anything': {
            'package': 'segment_anything',
            'description': 'Segment Anything Model - Core SAM functionality',
            'install_cmd': 'pip install segment_anything',
            'features': ['SAM Segmentation', 'Model Loading', 'Image Processing']
        },
        'timm': {
            'package': 'timm',
            'description': 'PyTorch Image Models - Vision transformer support',
            'install_cmd': 'pip install timm',
            'features': ['Vision Transformers', 'Model Registry', 'Pretrained Models']
        },
        'addict': {
            'package': 'addict',
            'description': 'Dictionary subclass for nested attributes',
            'install_cmd': 'pip install addict',
            'features': ['Configuration Management', 'Nested Attributes', 'Dictionary Extensions']
        }
    }
    
    def __init__(self):
        self.dependency_status: Dict[str, bool] = {}
        self.missing_dependencies: List[str] = []
        
    def check_import(self, import_name: str) -> Tuple[bool, Optional[str]]:
        """Check if a module can be imported and get version"""
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'unknown')
            return True, version
        except ImportError:
            return False, None
    
    def check_all_dependencies(self) -> Tuple[bool, Dict[str, bool]]:
        """Check all dependencies - ALL are required for production quality"""
        
        # Check each dependency
        for dep_key, dep_info in self.REQUIRED_DEPENDENCIES.items():
            import_name = dep_info.get('import_name', dep_key)
            is_available, version = self.check_import(import_name)
            self.dependency_status[dep_key] = is_available
            
            if not is_available:
                self.missing_dependencies.append(dep_key)
        
        # ALL dependencies must be available for production quality
        all_deps_ok = len(self.missing_dependencies) == 0
        
        return all_deps_ok, self.dependency_status
    
    def print_startup_report(self):
        """Print production-quality dependency status report at startup"""
        all_deps_ok, status = self.check_all_dependencies()
        
        # Header
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*80}{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}💀 S4Tool-Image Production Dependency Status Report{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}{'='*80}{Colors.RESET}")
        
        # Production quality statement
        print(f"\n{Colors.BOLD}🎯 Production Quality Statement:{Colors.RESET}")
        print(f"  {Colors.CYAN}• All dependencies are REQUIRED - no fallbacks or degraded modes{Colors.RESET}")
        print(f"  {Colors.CYAN}• Production quality above all - complete functionality guaranteed{Colors.RESET}")
        
        # Dependencies section
        print(f"\n{Colors.BOLD}🔧 Required Dependencies (ALL MANDATORY):{Colors.RESET}")
        for dep_key, dep_info in self.REQUIRED_DEPENDENCIES.items():
            import_name = dep_info.get('import_name', dep_key)
            is_available, version = self.check_import(import_name)
            
            if is_available:
                status_icon = "✅"
                status_color = Colors.GREEN
                version_info = f" v{version}" if version != 'unknown' else ""
                print(f"  {status_icon} {status_color}{dep_info['package']:<25}{Colors.RESET}{version_info}")
                features = ", ".join(dep_info.get('features', []))
                print(f"    {Colors.DIM}└─ {features}{Colors.RESET}")
            else:
                status_icon = "❌"
                status_color = Colors.RED  
                print(f"  {status_icon} {status_color}{dep_info['package']:<25}{Colors.RESET} - MISSING")
                print(f"    {Colors.RED}📦 Install: {dep_info['install_cmd']}{Colors.RESET}")
                features = ", ".join(dep_info.get('features', []))
                print(f"    {Colors.RED}⚠️  Missing Features: {features}{Colors.RESET}")
        
        # Summary
        print(f"\n{Colors.BOLD}📊 Production Quality Summary:{Colors.RESET}")
        total_deps = len(self.REQUIRED_DEPENDENCIES)
        available_deps = sum(1 for available in self.dependency_status.values() if available)
        
        if all_deps_ok:
            print(f"  ✅ {Colors.GREEN}{Colors.BOLD}PRODUCTION READY{Colors.RESET} - All dependencies satisfied")
            print(f"  🎨 {Colors.GREEN}Full S4Tool-Image functionality available{Colors.RESET}")
        else:
            print(f"  ❌ {Colors.RED}{Colors.BOLD}NOT PRODUCTION READY{Colors.RESET} - Missing critical dependencies")
            print(f"  ⚠️  {Colors.RED}Plugin functionality will be INCOMPLETE{Colors.RESET}")
        
        print(f"  📈 {Colors.CYAN}Dependencies status: {available_deps}/{total_deps} ({available_deps/total_deps*100:.1f}%){Colors.RESET}")
        
        # Installation commands
        if self.missing_dependencies:
            print(f"\n{Colors.BOLD}🔨 Required Installation Commands:{Colors.RESET}")
            
            missing_packages = [self.REQUIRED_DEPENDENCIES[k]['package'] for k in self.missing_dependencies]
            print(f"  {Colors.RED}Missing packages: {' '.join(missing_packages)}{Colors.RESET}")
            
            # Individual install commands
            print(f"\n  {Colors.YELLOW}Individual install commands:{Colors.RESET}")
            for dep_key in self.missing_dependencies:
                dep_info = self.REQUIRED_DEPENDENCIES[dep_key]
                print(f"    {Colors.YELLOW}{dep_info['install_cmd']}{Colors.RESET}")
                
            # All-in-one command
            print(f"\n  {Colors.GREEN}All-in-one install command:{Colors.RESET}")
            print(f"    {Colors.GREEN}pip install -r requirements.txt{Colors.RESET}")
        
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*80}{Colors.RESET}\n")
        
        # Return status for __init__.py
        return all_deps_ok, self.dependency_status
    
    def require_dependency(self, dep_key: str, feature_name: str = ""):
        """Require a dependency or raise informative error - NO FALLBACKS"""
        if dep_key not in self.dependency_status:
            # Check it now if not already checked
            dep_info = self.REQUIRED_DEPENDENCIES.get(dep_key)
            if dep_info:
                import_name = dep_info.get('import_name', dep_key)
                is_available, _ = self.check_import(import_name)
                self.dependency_status[dep_key] = is_available
            else:
                self.dependency_status[dep_key] = False
        
        if not self.dependency_status[dep_key]:
            dep_info = self.REQUIRED_DEPENDENCIES.get(dep_key)
            if dep_info:
                feature_desc = f" for {feature_name}" if feature_name else ""
                error_msg = f"""
❌ PRODUCTION QUALITY VIOLATION{feature_desc}

📦 Required Package: {dep_info['package']}
📄 Description: {dep_info['description']}
🔧 Install Command: {dep_info['install_cmd']}

🚨 S4Tool-Image operates at PRODUCTION QUALITY ONLY
   No fallbacks or degraded modes are available.
   All dependencies are mandatory for complete functionality.
   
💡 Please install the required package and restart ComfyUI.
"""
                S4ToolLogger.error("DependencyManager", error_msg)
                raise ImportError(f"Missing required dependency: {dep_info['package']}")
            else:
                raise ImportError(f"Unknown dependency: {dep_key}")

# Global dependency manager instance
_dependency_manager = None

def get_dependency_manager() -> DependencyManager:
    """Get global dependency manager instance"""
    global _dependency_manager
    if _dependency_manager is None:
        _dependency_manager = DependencyManager()
    return _dependency_manager

def require_dependency(dep_key: str, feature_name: str = ""):
    """Convenient function to require a dependency"""
    return get_dependency_manager().require_dependency(dep_key, feature_name)

def check_startup_dependencies():
    """Check and print dependency status at startup"""
    return get_dependency_manager().print_startup_report()
