"""
Simple static UML generator and "important section" identifier for Python repositories.

Features (best-effort static analysis):
- Parse all .py files under a repo path using ast
- Extract top-level classes, their methods, and top-level functions
- Collect call edges (function -> function, method -> method) by name matching
- Build a directed graph (networkx) and compute strongly connected components (SCCs)
- Compute simple importance score per SCC using size, degree-centrality and optional git activity
- Emit a Graphviz DOT (and optionally PNG/SVG) UML-like class/function diagram with SCCs highlighted

Limitations:
- It's a static, name-based, best-effort approach (no type inference). Works best on smaller, well-structured Python codebases.
- Method call detection is conservative: it matches attribute calls by name to defined methods.

Dependencies:
- Python 3.8+
- networkx
- graphviz Python package (or pydot) and Graphviz (system) to render images
- (optional) git present on PATH to compute commit counts

Usage:
    python uml_generator.py /path/to/repo --out uml.dot --render png --top 5

"""
import os
import sys
import ast
import argparse
import subprocess
import math
from collections import defaultdict

try:
    import networkx as nx
except ImportError:
    print("Please install networkx: pip install networkx")
    raise

try:
    from graphviz import Digraph
except ImportError:
    print("Please install graphviz python bindings: pip install graphviz and install graphviz system package")
    raise


class CodeVisitor(ast.NodeVisitor):
    """Collect classes, functions and call sites from an AST tree."""

    def __init__(self, filename, module_prefix=""):
        self.filename = filename
        self.module_prefix = module_prefix
        self.classes = {}  # fullname -> {'methods': set(), 'attrs': set(), 'lineno': int}
        self.functions = {}  # fullname -> {'lineno': int}
        self.calls = []  # tuples (caller_fullname, callee_name, lineno)
        self._current = []  # stack of current container names

    def _fullname(self, name):
        if self._current:
            return ".".join(self._current + [name])
        return name

    def visit_ClassDef(self, node: ast.ClassDef):
        cname = node.name
        full = self._fullname(cname)
        self.classes[full] = {"methods": set(), "lineno": node.lineno, "file": self.filename}
        self._current.append(cname)
        self.generic_visit(node)
        self._current.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        fname = node.name
        full = self._fullname(fname)
        if self._current:
            # method
            cls = ".".join(self._current)
            method_full = cls + "." + fname
            self.classes.setdefault(cls, {"methods": set(), "lineno": None, "file": self.filename})["methods"].add(fname)
            # record as function-like entity as well
            self.functions[method_full] = {"lineno": node.lineno, "file": self.filename}
            caller = method_full
        else:
            self.functions[full] = {"lineno": node.lineno, "file": self.filename}
            caller = full

        # visit the body and collect calls while knowing the caller
        self._current.append(fname)
        CallCollector(self, caller).visit(node)
        self.generic_visit(node)
        self._current.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.visit_FunctionDef(node)


class CallCollector(ast.NodeVisitor):
    def __init__(self, root_visitor: CodeVisitor, caller_fullname: str):
        self.root = root_visitor
        self.caller = caller_fullname

    def visit_Call(self, node: ast.Call):
        # try to extract the name being called
        callee = self._get_name(node.func)
        if callee:
            self.root.calls.append((self.caller, callee, getattr(node, 'lineno', None)))
        self.generic_visit(node)

    def _get_name(self, node):
        # returns 'a.b.c' or simple name 'foo' or None
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts = []
            cur = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
                return ".".join(reversed(parts))
        return None


def parse_python_repo(repo_path):
    defs = {
        'classes': {},  # fullname -> info
        'functions': {},
        'calls': []
    }

    for root, dirs, files in os.walk(repo_path):
        # skip virtualenvs and .git
        if any(p in root for p in (".git", "venv", "__pycache__", "env", ".venv")):
            continue
        for f in files:
            if not f.endswith('.py'):
                continue
            fullpath = os.path.join(root, f)
            rel_module = os.path.relpath(fullpath, repo_path).replace(os.path.sep, '.')
            rel_module = rel_module[:-3] if rel_module.endswith('.py') else rel_module
            try:
                with open(fullpath, 'r', encoding='utf-8') as fh:
                    src = fh.read()
                tree = ast.parse(src)
            except Exception as e:
                print(f"Warning: could not parse {fullpath}: {e}")
                continue

            visitor = CodeVisitor(fullpath, module_prefix=rel_module)
            visitor.visit(tree)

            # merge
            for k, v in visitor.classes.items():
                fullname = f"{rel_module}.{k}" if rel_module and not k.startswith(rel_module) else k
                defs['classes'][fullname] = v
            for k, v in visitor.functions.items():
                fullname = f"{rel_module}.{k}" if rel_module and not k.startswith(rel_module) else k
                defs['functions'][fullname] = v
            # calls: normalize callee by last token (best-effort)
            for caller, callee, lineno in visitor.calls:
                caller_full = f"{rel_module}.{caller}" if rel_module and not caller.startswith(rel_module) else caller
                defs['calls'].append((caller_full, callee, fullpath, lineno))

    return defs


def build_call_graph(defs):
    G = nx.DiGraph()

    # add nodes for each function/method and class
    for f in defs['functions']:
        G.add_node(f, kind='function', file=defs['functions'][f].get('file'))
    for c in defs['classes']:
        G.add_node(c, kind='class', file=defs['classes'][c].get('file'))
        # methods were added as functions earlier with class.method fullnames

    # link calls: attempt to resolve callee by exact match or by suffix match
    func_names = set(G.nodes())

    for caller, callee, file, lineno in defs['calls']:
        # exact match
        if callee in func_names:
            G.add_edge(caller, callee)
            continue
        # try suffix match: match last components
        matches = [n for n in func_names if n.endswith('.' + callee) or n.split('.')[-1] == callee]
        if len(matches) == 1:
            G.add_edge(caller, matches[0])
        elif len(matches) > 1:
            # pick the one in same file if possible
            same_file = [m for m in matches if G.nodes[m].get('file') == file]
            if same_file:
                G.add_edge(caller, same_file[0])
            else:
                # otherwise add edges to all matches (conservative)
                for m in matches:
                    G.add_edge(caller, m)
        else:
            # unresolved - ignore but could record for heuristics
            pass

    return G


def git_commit_count_for_file(repo_root, path):
    # best-effort: count commits touching this file using git
    try:
        rel = os.path.relpath(path, repo_root)
        out = subprocess.check_output(['git', '-C', repo_root, 'log', '--follow', '--pretty=%H', '--', rel], stderr=subprocess.DEVNULL)
        count = len(out.splitlines())
        return count
    except Exception:
        return 0


def score_scc(G, scc_nodes, repo_root=None):
    # simple composite score: size * (1 + avg_degree) * (1 + avg_commit_score)
    sub = G.subgraph(scc_nodes)
    size = sub.number_of_nodes()
    degs = [d for n, d in sub.degree()]
    avg_deg = sum(degs) / max(1, len(degs))

    commit_scores = []
    if repo_root:
        for n in sub.nodes():
            f = G.nodes[n].get('file')
            if f:
                commit_scores.append(git_commit_count_for_file(repo_root, f))
    avg_commits = sum(commit_scores) / max(1, len(commit_scores)) if commit_scores else 0

    # normalise commit with log
    commit_factor = math.log1p(avg_commits)
    centrality = nx.degree_centrality(G)
    central_vals = [centrality.get(n, 0) for n in sub.nodes()]
    avg_central = sum(central_vals) / max(1, len(central_vals))

    score = size * (1 + avg_deg) * (1 + commit_factor) * (1 + avg_central)
    return score


def identify_important_sections(G, repo_root=None, top_k=5):
    sccs = list(nx.strongly_connected_components(G))
    scored = []
    for s in sccs:
        if len(s) == 0:
            continue
        sc = score_scc(G, s, repo_root=repo_root)
        scored.append((sc, s))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:top_k]


def emit_uml_dot(G, important_sccs, out_dot_path, highlight_color='lightgoldenrod'):
    dot = Digraph(comment='UML-ish diagram')
    dot.attr('node', shape='record')

    # build mapping node->cluster index
    cluster_map = {}
    for idx, (_, s) in enumerate(important_sccs):
        for n in s:
            cluster_map[n] = idx

    # add nodes with labels
    for n in G.nodes():
        kind = G.nodes[n].get('kind', 'function')
        label = n
        if kind == 'class':
            # try to extract methods
            methods = []
            # methods may exist as separate nodes with classname.method format
            prefix = n + '.'
            mm = [m for m in G.nodes() if m.startswith(prefix)]
            methods = [m.split('.')[-1] for m in mm]
            label = "{" + n + "|" + "\l".join(sorted(methods)) + "\l}" if methods else n
        dot.node(n, label=label)

    # edges
    for u, v in G.edges():
        dot.edge(u, v)

    # clusters (just draw invisible bounding boxes by re-declaring subgraphs)
    for idx, (score, s) in enumerate(important_sccs):
        with dot.subgraph(name=f'cluster_{idx}') as c:
            c.attr(style='filled', color=highlight_color, fillcolor=highlight_color)
            c.attr(label=f'Important section {idx} (score={score:.2f})')
            for n in s:
                c.node(n)

    dot.render(out_dot_path, format='dot', cleanup=True)
    print(f'Wrote: {out_dot_path}.dot')
    return out_dot_path + '.dot'


def render_dot(dot_path, fmt='png'):
    # use graphviz to render to image
    out = dot_path.replace('.dot', '')
    subprocess.run(['dot', '-T' + fmt, dot_path, '-o', out + '.' + fmt], check=False)
    print(f'Rendered {out}.{fmt}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('repo', help='path to repo root')
    p.add_argument('--out', default='uml', help='output dot basename (no ext)')
    p.add_argument('--top', type=int, default=5, help='top-K important sections to highlight')
    p.add_argument('--render', choices=['png', 'svg', 'pdf', 'none'], default='none')
    args = p.parse_args()

    repo = args.repo
    defs = parse_python_repo(repo)
    G = build_call_graph(defs)
    print(f'Parsed functions/classes: {len([n for n in G if G.nodes[n].get("kind")])} total nodes, {G.number_of_edges()} edges')

    important = identify_important_sections(G, repo_root=repo, top_k=args.top)
    for idx, (score, s) in enumerate(important):
        print(f'Rank {idx}: score={score:.2f}, size={len(s)}')
        for n in list(s)[:10]:
            print('   ', n)

    dot_path = emit_uml_dot(G, important, args.out)
    if args.render != 'none':
        render_dot(dot_path, fmt=args.render)


if __name__ == '__main__':
    main()
