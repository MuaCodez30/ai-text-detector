"""Install required packages"""
import subprocess
import sys

packages = ['scipy', 'scikit-learn', 'pandas', 'matplotlib', 'seaborn']

print("Installing required packages...")
for package in packages:
    print(f"\nInstalling {package}...")
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                          capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode == 0:
        print(f"✓ {package} installed successfully")
    else:
        print(f"✗ Failed to install {package}")

print("\nVerifying installations...")
try:
    import numpy
    print(f"✓ numpy {numpy.__version__}")
except ImportError:
    print("✗ numpy not found")

try:
    import scipy
    print(f"✓ scipy {scipy.__version__}")
except ImportError:
    print("✗ scipy not found")

try:
    import sklearn
    print(f"✓ scikit-learn {sklearn.__version__}")
except ImportError:
    print("✗ scikit-learn not found")

try:
    import pandas
    print(f"✓ pandas {pandas.__version__}")
except ImportError:
    print("✗ pandas not found")

try:
    import matplotlib
    print(f"✓ matplotlib {matplotlib.__version__}")
except ImportError:
    print("✗ matplotlib not found")

try:
    import seaborn
    print(f"✓ seaborn {seaborn.__version__}")
except ImportError:
    print("✗ seaborn not found")

