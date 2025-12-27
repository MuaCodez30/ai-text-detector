"""
Execute the notebook by running all cells sequentially
This script extracts code from the notebook and runs it
"""
import json
import sys
from pathlib import Path

# Read the notebook
notebook_path = Path(__file__).parent / "advanced_training.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Extract code cells
code_cells = []
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if source.strip() and not source.strip().startswith('%'):
            code_cells.append(source)

print(f"Found {len(code_cells)} code cells to execute")
print("=" * 70)

# Create namespace for shared variables
namespace = {
    '__name__': '__main__',
    '__file__': str(notebook_path)
}

# Execute each cell
for i, code in enumerate(code_cells, 1):
    print(f"\n{'='*70}")
    print(f"Executing Cell {i}/{len(code_cells)}")
    print(f"{'='*70}")
    
    try:
        exec(code, namespace)
        print(f"✓ Cell {i} completed")
    except Exception as e:
        print(f"✗ Error in cell {i}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print(f"\n{'='*70}")
print("✓ All notebook cells executed successfully!")
print(f"{'='*70}")

