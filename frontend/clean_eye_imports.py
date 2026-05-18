import os
import re

base_dir = r"c:\Users\mahad\.gemini\antigravity\scratch\frontend\src\pages"

def clean_eye_import(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if 'Eye' is imported from 'lucide-react'
    if 'lucide-react' in content and 'Eye' in content:
        # Check if Eye is actually used in the file outside of the import
        # Remove imports first to check if used
        content_without_imports = re.sub(r'import\s+{[^}]+}\s+from\s+[\'"]lucide-react[\'"];?', '', content)
        if not re.search(r'\bEye\b', content_without_imports):
            # Eye is not used, so remove it from the import
            
            def repl(m):
                imports = m.group(1)
                # Remove Eye
                imports = re.sub(r'\bEye\b\s*,?', '', imports)
                # Clean up multiple commas or trailing commas
                imports = re.sub(r',\s*,', ',', imports)
                imports = re.sub(r',\s*$', '', imports).strip()
                if not imports:
                    return '' # If empty, we can just remove the whole import line, but it might be tricky
                return f'import {{ {imports} }} from {m.group(2)}'
            
            new_content = re.sub(r'import\s+{([^}]+)}\s+from\s+([\'"]lucide-react[\'"];?)', repl, content)
            
            # Clean up empty imports like `import {  } from 'lucide-react';`
            new_content = re.sub(r'import\s+{\s*}\s+from\s+[\'"]lucide-react[\'"];?\n?', '', new_content)
            
            if new_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"Cleaned {file_path}")

for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.tsx'):
            clean_eye_import(os.path.join(root, file))
