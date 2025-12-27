"""
Execute notebook cells sequentially
"""
import json
import sys
from pathlib import Path
from IPython import get_ipython
import subprocess

def execute_notebook_cells(notebook_path):
    """Execute all code cells from a notebook"""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Extract and execute code cells
    code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    
    print(f"Found {len(code_cells)} code cells to execute")
    print("=" * 70)
    
    # Create a namespace to share variables between cells
    namespace = {}
    
    for i, cell in enumerate(code_cells, 1):
        source = ''.join(cell['source'])
        
        if not source.strip():
            continue
            
        print(f"\n{'='*70}")
        print(f"Executing Cell {i}/{len(code_cells)}")
        print(f"{'='*70}")
        
        try:
            # Execute the cell code
            exec(compile(source, f'<cell {i}>', 'exec'), namespace)
            print(f"✓ Cell {i} executed successfully")
        except Exception as e:
            print(f"✗ Error in cell {i}: {e}")
            print(f"Cell content:\n{source[:200]}...")
            raise
    
    print(f"\n{'='*70}")
    print("All cells executed successfully!")
    print(f"{'='*70}")

if __name__ == "__main__":
    notebook_path = Path(__file__).parent / "advanced_training.ipynb"
    execute_notebook_cells(notebook_path)

