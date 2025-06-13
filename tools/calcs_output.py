import os
from tabulate import tabulate
import inspect

def save_markdown_table(filename: str, table: any):
    data = tabulate(table, headers='firstrow', tablefmt='pipe')

    # Get the path of the script that called this function
    caller_frame = inspect.stack()[1]
    caller_path = os.path.abspath(caller_frame.filename)
    caller_dir = os.path.dirname(caller_path)

    # Create 'calcs' directory in the caller's directory
    calcs_dir = os.path.join(caller_dir, 'calcs')
    os.makedirs(calcs_dir, exist_ok=True)

    # Normalize filename
    if not filename.endswith('.md'):
        filename += '.md'

    # Full file path
    file_path = os.path.join(calcs_dir, filename)

    # Write the table
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(data)

    print(f"✅ Markdown table saved to: {file_path}")