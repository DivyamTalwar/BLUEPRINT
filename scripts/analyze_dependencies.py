import os
import ast
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict


class DependencyAnalyzer:
    """Analyze all Python file dependencies"""

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.file_imports = {}  # file -> list of imports
        self.import_sources = defaultdict(list)  # import -> list of files that import it
        self.internal_modules = set()  # our own modules
        self.external_modules = set()  # third-party modules

    def find_all_python_files(self) -> List[Path]:
        """Find all Python files in project"""
        python_files = []

        for root, dirs, files in os.walk(self.root_dir):
            # Skip certain directories
            skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'venv', 'env',
                        '.venv', 'node_modules', 'archive', 'output', 'logs', 'data'}
            dirs[:] = [d for d in dirs if d not in skip_dirs]

            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)

        return python_files

    def analyze_file(self, filepath: Path) -> Dict:
        """Analyze imports in a single file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append({
                            'type': 'import',
                            'module': alias.name,
                            'alias': alias.asname,
                            'line': node.lineno
                        })

                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append({
                            'type': 'from_import',
                            'module': module,
                            'name': alias.name,
                            'alias': alias.asname,
                            'line': node.lineno,
                            'level': node.level  # relative import level
                        })

            return {
                'file': str(filepath.relative_to(self.root_dir)),
                'imports': imports,
                'lines': len(content.split('\n'))
            }

        except Exception as e:
            return {
                'file': str(filepath.relative_to(self.root_dir)),
                'error': str(e),
                'imports': []
            }

    def classify_import(self, import_info: Dict) -> str:
        """Classify if import is internal or external"""
        module = import_info.get('module', '')

        # Check if it's our internal module
        if module.startswith('src.') or module == 'src':
            return 'internal'

        # Check if it's a relative import
        if import_info.get('level', 0) > 0:
            return 'internal'

        # Standard library or third-party
        return 'external'

    def analyze_all(self) -> Dict:
        """Analyze all Python files"""
        python_files = self.find_all_python_files()

        print(f"Found {len(python_files)} Python files to analyze...")

        for filepath in python_files:
            analysis = self.analyze_file(filepath)
            relative_path = str(filepath.relative_to(self.root_dir))

            self.file_imports[relative_path] = analysis

            # Build reverse mapping
            for imp in analysis.get('imports', []):
                module = imp.get('module', '')

                # Track who imports what
                self.import_sources[module].append(relative_path)

                # Classify
                if self.classify_import(imp) == 'internal':
                    self.internal_modules.add(module)
                else:
                    self.external_modules.add(module)

        return self.generate_report()

    def generate_report(self) -> Dict:
        """Generate comprehensive dependency report"""

        # Find files with most dependencies
        dependency_counts = {
            file: len(data.get('imports', []))
            for file, data in self.file_imports.items()
        }

        most_dependent = sorted(
            dependency_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        # Find most imported internal modules
        internal_import_counts = {
            module: len(files)
            for module, files in self.import_sources.items()
            if module in self.internal_modules
        }

        most_imported = sorted(
            internal_import_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        # Detect circular dependencies
        circular_deps = self.detect_circular_dependencies()

        report = {
            'summary': {
                'total_files': len(self.file_imports),
                'total_imports': sum(len(d.get('imports', [])) for d in self.file_imports.values()),
                'internal_modules': len(self.internal_modules),
                'external_modules': len(self.external_modules),
                'circular_dependencies': len(circular_deps)
            },
            'most_dependent_files': [
                {'file': file, 'import_count': count}
                for file, count in most_dependent
            ],
            'most_imported_modules': [
                {'module': module, 'imported_by_count': count}
                for module, count in most_imported
            ],
            'circular_dependencies': circular_deps,
            'file_imports': self.file_imports,
            'import_sources': dict(self.import_sources),
            'internal_modules': sorted(list(self.internal_modules)),
            'external_modules': sorted(list(self.external_modules))
        }

        return report

    def detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular import dependencies"""
        # Build dependency graph
        graph = defaultdict(set)

        for file, data in self.file_imports.items():
            for imp in data.get('imports', []):
                if self.classify_import(imp) == 'internal':
                    module = imp.get('module', '')
                    # Convert module name to file path
                    module_file = self.module_to_file(module)
                    if module_file:
                        graph[file].add(module_file)

        # Find cycles using DFS
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, path[:]):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    if cycle not in cycles:
                        cycles.append(cycle)
                    return True

            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                dfs(node, [])

        return cycles

    def module_to_file(self, module: str) -> str:
        """Convert module name to file path"""
        if not module.startswith('src.'):
            return None

        # src.core.rpg -> src/core/rpg.py
        parts = module.split('.')
        path = '/'.join(parts) + '.py'

        # Check if file exists
        if path in self.file_imports:
            return path

        # Try __init__.py
        path = '/'.join(parts) + '/__init__.py'
        if path in self.file_imports:
            return path

        return None

    def print_summary(self, report: Dict):
        """Print human-readable summary"""
        print("\n" + "=" * 70)
        print("DEPENDENCY ANALYSIS REPORT")
        print("=" * 70)

        summary = report['summary']
        print(f"\nTotal Python Files: {summary['total_files']}")
        print(f"Total Import Statements: {summary['total_imports']}")
        print(f"Internal Modules: {summary['internal_modules']}")
        print(f"External Modules: {summary['external_modules']}")
        print(f"Circular Dependencies: {summary['circular_dependencies']}")

        print("\n" + "-" * 70)
        print("TOP 10 FILES WITH MOST DEPENDENCIES")
        print("-" * 70)
        for item in report['most_dependent_files']:
            print(f"  {item['file']:<50} {item['import_count']} imports")

        print("\n" + "-" * 70)
        print("TOP 10 MOST IMPORTED INTERNAL MODULES")
        print("-" * 70)
        for item in report['most_imported_modules']:
            print(f"  {item['module']:<50} imported by {item['imported_by_count']} files")

        if report['circular_dependencies']:
            print("\n" + "-" * 70)
            print("WARNING: CIRCULAR DEPENDENCIES DETECTED")
            print("-" * 70)
            for i, cycle in enumerate(report['circular_dependencies'], 1):
                print(f"\n  Cycle {i}:")
                for file in cycle:
                    print(f"    -> {file}")

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print("\nDetailed report saved to: dependency_map.json")
        print("\nTo view dependencies for a specific file:")
        print("  python analyze_dependencies.py --file src/core/rpg.py")
        print("\nTo check what would break if renaming a file:")
        print("  python analyze_dependencies.py --check-rename src/old.py src/new.py")
        print()

    def check_rename_impact(self, old_path: str, new_path: str):
        """Check what would break if we rename a file"""
        print("\n" + "=" * 70)
        print(f"RENAME IMPACT ANALYSIS")
        print("=" * 70)
        print(f"\nRenaming: {old_path}")
        print(f"      To: {new_path}")

        # Find all files that import this module
        old_module = self.file_to_module(old_path)

        if not old_module:
            print(f"\n[ERROR] Could not determine module name for {old_path}")
            return

        importers = self.import_sources.get(old_module, [])

        if not importers:
            print(f"\n[OK] No files import this module. Safe to rename.")
            return

        print(f"\n[WARNING] {len(importers)} file(s) would need updates:")
        print("-" * 70)

        for file in importers:
            imports = [
                imp for imp in self.file_imports[file].get('imports', [])
                if imp.get('module') == old_module
            ]

            print(f"\n  File: {file}")
            for imp in imports:
                print(f"    Line {imp['line']}: {self.format_import(imp)}")
                print(f"    Update to: {self.format_import(imp, old_module, self.file_to_module(new_path))}")

        print("\n" + "=" * 70)
        print(f"TOTAL FILES TO UPDATE: {len(importers)}")
        print("=" * 70)

    def file_to_module(self, filepath: str) -> str:
        """Convert file path to module name"""
        # src/core/rpg.py -> src.core.rpg
        path = Path(filepath)

        if path.suffix == '.py':
            parts = path.parts
            if parts[0] == 'src':
                module_parts = list(parts[:-1]) + [path.stem]
                return '.'.join(module_parts)

        return None

    def format_import(self, imp: Dict, old_module: str = None, new_module: str = None) -> str:
        """Format import statement"""
        if imp['type'] == 'import':
            module = new_module if (old_module and imp['module'] == old_module) else imp['module']
            return f"import {module}" + (f" as {imp['alias']}" if imp['alias'] else "")
        else:  # from_import
            module = new_module if (old_module and imp['module'] == old_module) else imp['module']
            name = imp['name']
            alias = f" as {imp['alias']}" if imp['alias'] else ""
            return f"from {module} import {name}{alias}"


def main():
    import sys

    analyzer = DependencyAnalyzer()

    if len(sys.argv) > 1:
        if sys.argv[1] == '--file':
            # Show dependencies for specific file
            filepath = sys.argv[2]
            report = analyzer.analyze_all()

            if filepath in report['file_imports']:
                file_data = report['file_imports'][filepath]
                print(f"\nDependencies for: {filepath}")
                print("-" * 70)
                for imp in file_data.get('imports', []):
                    print(f"  {analyzer.format_import(imp)}")
            else:
                print(f"File not found: {filepath}")

        elif sys.argv[1] == '--check-rename':
            # Check rename impact
            old_path = sys.argv[2]
            new_path = sys.argv[3]
            report = analyzer.analyze_all()
            analyzer.check_rename_impact(old_path, new_path)

    else:
        # Full analysis
        report = analyzer.analyze_all()
        analyzer.print_summary(report)

        # Save to JSON
        with open('dependency_map.json', 'w') as f:
            json.dump(report, f, indent=2)

        print("\n[SUCCESS] Dependency map saved to dependency_map.json\n")


if __name__ == "__main__":
    main()
