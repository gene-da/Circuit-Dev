import os

def print_file_tree(start_path='.', prefix=''):
    items = sorted(
        [item for item in os.listdir(start_path)
         if not item.startswith('.') and not item.startswith('_') and not item.endswith('.pyc')]
    )

    for index, item in enumerate(items):
        item_path = os.path.join(start_path, item)
        connector = '└── ' if index == len(items) - 1 else '├── '
        print(f"{prefix}{connector}{item}")
        if os.path.isdir(item_path):
            extension = '    ' if index == len(items) - 1 else '│   '
            print_file_tree(item_path, prefix + extension)

if __name__ == "__main__":
    print("File Tree:")
    print_file_tree(os.getcwd())
