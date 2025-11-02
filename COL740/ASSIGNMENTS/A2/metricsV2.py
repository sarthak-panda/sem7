#!/usr/bin/env python3
"""
cpp_metrics.py

Single-file tool that computes a range of metrics for a C++ repository:
- LOC, SLOC
- Number of functions
- McCabe CCN (max and avg) via lizard
- Halstead Effort (E)
- Class cohesion: LCOM4, TCC, LCC
- Coupling per class (inheritance + usage)
- Fan-in / Fan-out per function
- Knot (approximate via loop counts)
- Writes per-class and per-function CSVs when WRITE_INTERMEDIATE = True

Dependencies:
  pip install lizard
  apt / system: install clang (libclang). Python binding clang.cindex must be importable.
  You may need to set clang.cindex.Config.set_library_file(...) to point to libclang on your system.

Usage:
  python cpp_metrics.py /path/to/cpp/repo

If parsing fails for some headers due to missing include paths, set CLANG_ARGS below or pass compile flags.
"""

import os
import re
import math
import csv
import sys
from collections import defaultdict

# External libs
try:
    import lizard
except Exception as e:
    print("ERROR: lizard is required. Install with `pip install lizard`.")
    raise

# Try to import clang.cindex
try:
    import clang.cindex as clang
except Exception as e:
    clang = None
    print("WARNING: clang.cindex not available. AST-based metrics will be skipped. "
          "Install clang and the Python bindings (libclang).")
# --- Normalize CursorKind names across libclang versions ---
def _ck(name):
    return getattr(clang.CursorKind, name, None) if clang is not None else None

MEMBER_KINDS = tuple(k for k in (_ck("MEMBER_REF_EXPR"), _ck("MEMBER_EXPR")) if k is not None)
CALL_KINDS = tuple(k for k in (_ck("CALL_EXPR"),) if k is not None)
TYPE_REF_KINDS = tuple(k for k in (_ck("TYPE_REF"), _ck("TEMPLATE_REF"), _ck("DECL_REF_EXPR")) if k is not None)
LOOP_KINDS = tuple(k for k in (_ck("FOR_STMT"), _ck("WHILE_STMT"), _ck("DO_STMT")) if k is not None)


# Global toggle to write per-class/function CSVs
WRITE_INTERMEDIATE = True

# Default clang arguments (add include paths or standard if needed)
CLANG_ARGS = ["-std=c++17", "-I/usr/include", "-I/usr/local/include"]

# File extensions to consider
CPP_EXTS = (".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hh", ".hxx")


# ---------- File scanning & LOC / SLOC ----------
def find_cpp_files(root_path):
    files = []
    for dirpath, _, filenames in os.walk(root_path):
        for f in filenames:
            if f.endswith(CPP_EXTS):
                files.append(os.path.join(dirpath, f))
    return files


def count_loc(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(CPP_EXTS):
                try:
                    with open(os.path.join(root, f), "r", errors="ignore") as fp:
                        total += sum(1 for _ in fp)
                except Exception:
                    pass
    return total


def count_sloc(path):
    total = 0
    block_comment_re = re.compile(r'/\*.*?\*/', flags=re.S)
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(CPP_EXTS):
                fp = os.path.join(root, f)
                try:
                    text = open(fp, "r", errors="ignore").read()
                except Exception:
                    continue
                # remove block comments (approx)
                text = block_comment_re.sub("", text)
                for line in text.splitlines():
                    s = line.strip()
                    if not s:
                        continue
                    if s.startswith("//"):
                        continue
                    total += 1
    return total


# ---------- Lizard analysis (functions, cyclomatic) ----------
# def analyze_with_lizard(root_path):
#     # lizard.analyze_path handles recursion
#     res = lizard.analyze_path(root_path)
#     functions = []
#     for file_info in res:
#         for fn in file_info.function_list:
#             # fn: has attributes name, cyclomatic_complexity, nloc, long_name, filename, start_line ...
#             functions.append(fn)
#     return functions
def analyze_with_lizard(root_path):
    functions = []
    for fp in find_cpp_files(root_path):
        try:
            file_info = lizard.analyze_file(fp)
        except Exception as e:
            print(f"lizard failed on {fp}: {e}")
            continue
        for fn in file_info.function_list:
            functions.append(fn)
    return functions


# ---------- Halstead metrics (simple token-based approximation) ----------
# Note: for C++ a precise operator/operand tokenizer is complex; we use regex heuristics.
OPERATOR_RE = re.compile(
    r'(\+\+|--|<<|>>|<=|>=|==|!=|\+=|-=|\*=|/=|%=|&=|\|=|\^=|->|::|&&|\|\||'
    r'[+\-*/%&|\^~!=<>?:;,\[\]\(\)\{\}])'
)
OPERAND_RE = re.compile(r'\b[_A-Za-z]\w*\b|\b\d+\b')


def halstead_metrics(files):
    operators = []
    operands = []
    for fp in files:
        try:
            text = open(fp, "r", errors="ignore").read()
        except Exception:
            continue
        # remove block comments for cleaner tokenization
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
        # remove line comments
        text = re.sub(r'//.*', '', text)
        ops = OPERATOR_RE.findall(text)
        vals = OPERAND_RE.findall(text)
        operators.extend(ops)
        operands.extend(vals)
    N1 = len(operators)
    N2 = len(operands)
    eta1 = len(set(operators))
    eta2 = len(set(operands))
    N = N1 + N2
    eta = eta1 + eta2
    V = N * math.log2(eta) if eta > 0 else 0.0
    D = (eta1 / 2.0) * (N2 / eta2) if eta2 > 0 else 0.0
    E = D * V
    return {"N1": N1, "N2": N2, "eta1": eta1, "eta2": eta2, "V": V, "D": D, "E": E}


# ---------- libclang-based AST analysis for classes, methods, calls, loops ----------
# This part is best-effort. It requires clang.cindex to be importable.
class ClangCppAnalyzer:
    def __init__(self, files, clang_args=None):
        self.files = files
        self.clang_args = clang_args or []
        # Data structures:
        # classes: name -> {methods:set, fields:set, fields_used: {method:set}, calls: {method:set}, bases:set, references:set}
        self.classes = {}
        # functions: global functions outside classes: qualified_name -> {calls:set, loops:int}
        self.functions = {}
        # mapping from qualified function name to its callers (reverse map)
        self.call_graph = defaultdict(set)
        # loops per function
        self.loops = defaultdict(int)

    @staticmethod
    def qual_name(cursor):
        """Build a qualified name (Namespace::Class::method) if possible."""
        parts = []
        cur = cursor
        while cur and cur.kind != clang.CursorKind.TRANSLATION_UNIT:
            if cur.spelling:
                parts.append(cur.spelling)
            cur = cur.semantic_parent
        return "::".join(reversed(parts))

    def handle_translation_unit(self, tu):
        # traverse AST
        for cursor in tu.cursor.get_children():
            self.visit(cursor, parent=None)

    def visit(self, cursor, parent):
        kind = cursor.kind

        # Class / struct declarations (definitions)
        if kind in (clang.CursorKind.CLASS_DECL, clang.CursorKind.STRUCT_DECL):
            if not cursor.is_definition():
                return
            class_name = self.qual_name(cursor)
            if class_name not in self.classes:
                self.classes[class_name] = {
                    "methods": set(),
                    "fields": set(),
                    "fields_used": defaultdict(set),
                    "calls": defaultdict(set),
                    "bases": set(),
                    "references": set(),
                }
            # record base classes (CXX_BASE_SPECIFIER children)
            for c in cursor.get_children():
                if c.kind == clang.CursorKind.CXX_BASE_SPECIFIER:
                    ref = c.get_definition()
                    if ref is not None:
                        self.classes[class_name]["bases"].add(self.qual_name(ref))
                # fields
                if c.kind == clang.CursorKind.FIELD_DECL:
                    name = c.spelling
                    if name:
                        self.classes[class_name]["fields"].add(name)
            # visit children to find methods and nested items
            for c in cursor.get_children():
                self.visit(c, parent=class_name)
            return

        # Methods (CXX_METHOD) and constructors/destructors
        if kind in (clang.CursorKind.CXX_METHOD, clang.CursorKind.CONSTRUCTOR, clang.CursorKind.DESTRUCTOR, clang.CursorKind.FUNCTION_TEMPLATE):
            if cursor.semantic_parent and cursor.semantic_parent.kind in (clang.CursorKind.CLASS_DECL, clang.CursorKind.STRUCT_DECL):
                class_name = self.qual_name(cursor.semantic_parent)
                method_name = cursor.spelling
                qname = f"{class_name}::{method_name}"
                self.classes.setdefault(class_name, {
                    "methods": set(), "fields": set(),
                    "fields_used": defaultdict(set),
                    "calls": defaultdict(set),
                    "bases": set(), "references": set()
                })
                self.classes[class_name]["methods"].add(qname)
                # walk method body
                for c in cursor.get_children():
                    self.visit_in_function_body(c, current_function=qname, current_class=class_name)
                return
            else:
                # free function
                func_q = self.qual_name(cursor)
                self.functions.setdefault(func_q, {"calls": set()})
                for c in cursor.get_children():
                    self.visit_in_function_body(c, current_function=func_q, current_class=None)
                return

        # Free function declarations may appear as FUNCTION_DECL
        if kind == clang.CursorKind.FUNCTION_DECL:
            if cursor.is_definition():
                func_q = self.qual_name(cursor)
                self.functions.setdefault(func_q, {"calls": set()})
                for c in cursor.get_children():
                    self.visit_in_function_body(c, current_function=func_q, current_class=None)
            return

        # For other nodes, recurse
        for c in cursor.get_children():
            self.visit(c, parent=parent)

    # def visit_in_function_body(self, cursor, current_function, current_class):
    #     """
    #     Inspect nodes inside a function/method body to collect:
    #      - member references (self.field) -> fields_used
    #      - calls -> calls list
    #      - loops -> increment loop count
    #      - type/constructor calls -> class references
    #     """
    #     k = cursor.kind
    #
    #     # Member access expressions (this->field or .field)
    #     if k == clang.CursorKind.MEMBER_REF_EXPR or k == clang.CursorKind.MEMBER_EXPR:
    #         # try to get member name
    #         name = cursor.spelling or (cursor.displayname if cursor.displayname else None)
    #         if not name:
    #             # sometimes the child is a declref of field
    #             for ch in cursor.get_children():
    #                 if ch.kind == clang.CursorKind.DECL_REF_EXPR:
    #                     name = ch.spelling
    #         if name and current_class:
    #             # strip potential prefixes
    #             if "::" in current_function:
    #                 self.classes[current_class]["fields_used"][current_function].add(name)
    #     # Call expressions
    #     elif k == clang.CursorKind.CALL_EXPR:
    #         # find referenced function / callee
    #         # sometimes cursor.referenced is the function decl
    #         ref = cursor.referenced
    #         called_name = None
    #         if ref is not None:
    #             called_name = self.qual_name(ref)
    #         else:
    #             # attempt to extract from children (decl ref)
    #             for ch in cursor.get_children():
    #                 if ch.kind == clang.CursorKind.DECL_REF_EXPR:
    #                     called_name = self.qual_name(ch.referenced) if ch.referenced else ch.spelling
    #                 elif ch.kind == clang.CursorKind.MEMBER_REF_EXPR:
    #                     # method call like obj.method()
    #                     # spelled as method name
    #                     called_name = ch.spelling or ch.displayname
    #         if called_name:
    #             # normalize: if it's a method of same class, qualify it
    #             if current_class and "::" not in called_name:
    #                 # called_name may be just method name; attempt to find a method in same class
    #                 for m in self.classes.get(current_class, {}).get("methods", []):
    #                     if m.endswith("::" + called_name):
    #                         called_name = m
    #                         break
    #             # record call
    #             if current_class and "::" in current_function:
    #                 self.classes[current_class]["calls"][current_function].add(called_name)
    #             else:
    #                 self.functions.setdefault(current_function, {"calls": set()})
    #                 self.functions[current_function]["calls"].add(called_name)
    #             # update global call graph
    #             self.call_graph[called_name].add(current_function)
    #         # also visit children
    #         for ch in cursor.get_children():
    #             self.visit_in_function_body(ch, current_function, current_class)
    #         return
    #
    #     # Type references -> potential coupling / references (e.g., constructors)
    #     elif k == clang.CursorKind.TYPE_REF or k == clang.CursorKind.TEMPLATE_REF or k == clang.CursorKind.DECL_REF_EXPR:
    #         # if this references a type definition (class), add reference
    #         ref = cursor.referenced
    #         if ref is not None and ref.kind in (clang.CursorKind.CLASS_DECL, clang.CursorKind.STRUCT_DECL):
    #             typename = self.qual_name(ref)
    #             if current_class:
    #                 self.classes[current_class]["references"].add(typename)
    #     # Loops
    #     elif k in (clang.CursorKind.FOR_STMT, clang.CursorKind.WHILE_STMT, clang.CursorKind.DO_STMT):
    #         self.loops[current_function] += 1
    #
    #     # Recurse for other nodes
    #     for ch in cursor.get_children():
    #         self.visit_in_function_body(ch, current_function, current_class)
    def visit_in_function_body(self, cursor, current_function, current_class):
        """
        Inspect nodes inside a function/method body to collect:
        - member references (self.field) -> fields_used
        - calls -> calls list
        - loops -> increment loop count
        - type/constructor calls -> class references
        """
        k = cursor.kind

            # Member access expressions (this->field or .field)
        if k in MEMBER_KINDS:
            # try to get member name
            name = cursor.spelling or (cursor.displayname if cursor.displayname else None)
            if not name:
                # sometimes the child is a declref of field
                for ch in cursor.get_children():
                    if ch.kind == getattr(clang.CursorKind, "DECL_REF_EXPR", None):
                        name = ch.spelling
            if name and current_class:
                if "::" in current_function:
                    self.classes[current_class]["fields_used"][current_function].add(name)

        # Call expressions
        elif k in CALL_KINDS:
            ref = cursor.referenced
            called_name = None
            if ref is not None:
                called_name = self.qual_name(ref)
            else:
                for ch in cursor.get_children():
                    if ch.kind == getattr(clang.CursorKind, "DECL_REF_EXPR", None):
                        called_name = self.qual_name(ch.referenced) if ch.referenced else ch.spelling
                    elif ch.kind in MEMBER_KINDS:
                        called_name = ch.spelling or ch.displayname
            if called_name:
                if current_class and "::" not in called_name:
                    for m in self.classes.get(current_class, {}).get("methods", []):
                        if m.endswith("::" + called_name):
                            called_name = m
                            break
                if current_class and "::" in current_function:
                    self.classes[current_class]["calls"][current_function].add(called_name)
                else:
                    self.functions.setdefault(current_function, {"calls": set()})
                    self.functions[current_function]["calls"].add(called_name)
                self.call_graph[called_name].add(current_function)
            for ch in cursor.get_children():
                self.visit_in_function_body(ch, current_function, current_class)
            return

        # Type references -> potential coupling / references (e.g., constructors)
        elif k in TYPE_REF_KINDS:
            ref = cursor.referenced
            if ref is not None and ref.kind in (getattr(clang.CursorKind, "CLASS_DECL", None), getattr(clang.CursorKind, "STRUCT_DECL", None)):
                typename = self.qual_name(ref)
                if current_class:
                    self.classes[current_class]["references"].add(typename)

        # Loops
        elif k in LOOP_KINDS:
            self.loops[current_function] += 1

        # Recurse for other nodes
        for ch in cursor.get_children():
            self.visit_in_function_body(ch, current_function, current_class)

    def parse_all(self):
        if clang is None:
            print("clang.cindex not present; skipping AST-based metrics.")
            return
        # Optionally set library file if needed:
        # clang.cindex.Config.set_library_file("/path/to/libclang.so")

        index = clang.Index.create()
        for f in self.files:
            # Skip parsing if file is empty or not existent
            if not os.path.isfile(f):
                continue
            try:
                tu = index.parse(f, args=self.clang_args)
            except Exception as e:
                # Try parsing with fewer args
                try:
                    tu = index.parse(f, args=["-std=c++17"])
                except Exception as e2:
                    print(f"Failed to parse {f}: {e2}")
                    continue
            # process translation unit
            self.handle_translation_unit(tu)


# ---------- Aggregation & CSV output ----------
def compute_class_metrics(classes):
    """
    Given classes dict produced by ClangCppAnalyzer, compute:
    - LCOM4 (number of connected components of method graph)
    - TCC (NDC / NP)
    - LCC ((NDC + NID) / NP)
    - Coupling (len(bases | references))
    """
    out = {}
    for cls, dat in classes.items():
        methods = sorted(dat["methods"])
        if not methods:
            continue
        # Build adjacency graph connecting methods if they share fields or call each other
        graph = {m: set() for m in methods}
        # shared fields
        for i, m1 in enumerate(methods):
            for m2 in methods[i + 1:]:
                if dat["fields_used"].get(m1, set()) & dat["fields_used"].get(m2, set()):
                    graph[m1].add(m2)
                    graph[m2].add(m1)
                if m2 in dat["calls"].get(m1, set()) or m1 in dat["calls"].get(m2, set()):
                    graph[m1].add(m2)
                    graph[m2].add(m1)
        # Count connected components (LCOM4)
        visited = set()
        components = 0
        for m in methods:
            if m not in visited:
                components += 1
                stack = [m]
                while stack:
                    cur = stack.pop()
                    if cur not in visited:
                        visited.add(cur)
                        for nb in graph[cur]:
                            if nb not in visited:
                                stack.append(nb)
        lcom4 = components

        # TCC / LCC among visible methods (non-underscore prefixed)
        visible = [m for m in methods if not os.path.basename(m).startswith("_")]
        NP = len(visible) * (len(visible) - 1) / 2 if len(visible) > 1 else 0
        NDC = 0
        for i, m1 in enumerate(visible):
            for m2 in visible[i + 1:]:
                if m2 in graph[m1]:
                    NDC += 1
        # Indirect connections: for each connected component, pairs minus direct edges
        visited2 = set()
        indirect = 0
        for m in visible:
            if m not in visited2:
                comp = set([m])
                stack = [m]
                while stack:
                    c = stack.pop()
                    if c not in visited2:
                        visited2.add(c)
                        for nb in graph[c]:
                            if nb not in visited2:
                                stack.append(nb)
                                comp.add(nb)
                size = len(comp)
                if size > 1:
                    total_pairs = size * (size - 1) / 2
                    # Count direct edges in this component
                    edges = 0
                    clist = list(comp)
                    for i in range(size):
                        for j in range(i + 1, size):
                            if clist[j] in graph[clist[i]]:
                                edges += 1
                    indirect += (total_pairs - edges)
        TCC = NDC / NP if NP else 0.0
        LCC = (NDC + indirect) / NP if NP else 0.0
        coupling = len(dat["bases"] | dat["references"])
        out[cls] = {"LCOM4": lcom4, "TCC": TCC, "LCC": LCC, "Coupling": coupling, "methods": methods}
    return out


def compute_function_metrics(clang_analyzer):
    # Build fan-in/out across all functions (class methods and free functions)
    calls_map = {}
    # collect calls from classes
    for cls, dat in clang_analyzer.classes.items():
        for m, called in dat["calls"].items():
            calls_map[m] = set(called)
    # collect calls from free functions
    for f, dat in clang_analyzer.functions.items():
        calls_map[f] = set(dat.get("calls", set()))
    # Now fan-in: how many distinct callers for each function
    fan_in = {fn: 0 for fn in calls_map.keys()}
    fan_out = {fn: len(called) for fn, called in calls_map.items()}
    for caller, called_set in calls_map.items():
        for callee in called_set:
            if callee not in fan_in:
                fan_in[callee] = 0
            fan_in[callee] += 1
    # knots approximate from loops recorded
    knots = {}
    for fn in set(list(fan_in.keys()) + list(fan_out.keys())):
        knots[fn] = clang_analyzer.loops.get(fn, 0)
    return {"fan_in": fan_in, "fan_out": fan_out, "knots": knots}


# ---------- Main runner ----------
def main(root_path):
    if not os.path.isdir(root_path):
        print("Provide a valid directory path to a C++ repository.")
        return

    print("Scanning files...")
    cpp_files = find_cpp_files(root_path)
    if not cpp_files:
        print("No C/C++ files found in path.")
        return

    # LOC / SLOC
    print("Counting LOC / SLOC ...")
    loc = count_loc(root_path)
    sloc = count_sloc(root_path)

    # Lizard analysis
    print("Running Lizard for functions & cyclomatic complexity ...")
    functions = analyze_with_lizard(root_path)
    fn_count = len(functions)
    ccn_values = [fn.cyclomatic_complexity for fn in functions]
    max_ccn = max(ccn_values) if ccn_values else 0
    avg_ccn = (sum(ccn_values) / fn_count) if fn_count else 0.0

    # Halstead
    print("Computing Halstead metrics (approx)...")
    hal = halstead_metrics(cpp_files)
    E = hal["E"]

    # AST-based metrics via clang
    clang_analyzer = None
    class_metrics = {}
    function_metrics = {"fan_in": {}, "fan_out": {}, "knots": {}}
    if clang is not None:
        print("Parsing C++ AST with libclang (may be slow)...")
        clang_analyzer = ClangCppAnalyzer(cpp_files, clang_args=CLANG_ARGS)
        clang_analyzer.parse_all()
        class_metrics = compute_class_metrics(clang_analyzer.classes)
        function_metrics = compute_function_metrics(clang_analyzer)
    else:
        print("Skipping AST-based metrics because clang.cindex is not available.")

    # Aggregates
    # Class aggregates
    if class_metrics:
        avg_lcom = sum(v["LCOM4"] for v in class_metrics.values()) / len(class_metrics)
        avg_TCC = sum(v["TCC"] for v in class_metrics.values()) / len(class_metrics)
        avg_LCC = sum(v["LCC"] for v in class_metrics.values()) / len(class_metrics)
        avg_coupling = sum(v["Coupling"] for v in class_metrics.values()) / len(class_metrics)
    else:
        avg_lcom = avg_TCC = avg_LCC = avg_coupling = 0.0

    # Function aggregates
    if function_metrics and function_metrics["fan_in"]:
        fan_in_vals = list(function_metrics["fan_in"].values())
        fan_out_vals = list(function_metrics["fan_out"].values())
        knots_vals = list(function_metrics["knots"].values())
        avg_fin = sum(fan_in_vals) / len(fan_in_vals) if fan_in_vals else 0.0
        avg_fout = sum(fan_out_vals) / len(fan_out_vals) if fan_out_vals else 0.0
        avg_knots = sum(knots_vals) / len(knots_vals) if knots_vals else 0.0
    else:
        avg_fin = avg_fout = avg_knots = 0.0

    # Print summary
    print("\n=== Project-wide Metrics ===")
    print(f"Files scanned: {len(cpp_files)}")
    print(f"LOC (total lines): {loc}")
    print(f"SLOC (non-blank non-comment lines): {sloc}")
    print(f"Number of functions (lizard): {fn_count}")
    print(f"McCabe Cyclomatic Complexity: max={max_ccn}, avg={avg_ccn:.2f}")
    print(f"Halstead Effort (E) (approx): {E:.2f}")
    print(f"Avg LCOM4 (per-class): {avg_lcom:.2f}")
    print(f"Avg TCC (tight cohesion): {avg_TCC:.3f}")
    print(f"Avg LCC (loose cohesion): {avg_LCC:.3f}")
    print(f"Avg coupling per class: {avg_coupling:.2f}")
    print(f"Avg fan-in per function: {avg_fin:.2f}")
    print(f"Avg fan-out per function: {avg_fout:.2f}")
    print(f"Avg knots (loops) per function (approx): {avg_knots:.2f}")

    # Write intermediate CSVs if requested
    if WRITE_INTERMEDIATE:
        print("Writing intermediate CSVs...")
        try:
            if class_metrics:
                with open("class_metrics.csv", "w", newline="", encoding="utf-8") as cf:
                    writer = csv.writer(cf)
                    writer.writerow(["Class", "LCOM4", "TCC", "LCC", "Coupling", "NumMethods"])
                    for cls, m in class_metrics.items():
                        writer.writerow([cls, m["LCOM4"], f"{m['TCC']:.3f}", f"{m['LCC']:.3f}", m["Coupling"], len(m["methods"])])
            if function_metrics and function_metrics["fan_in"]:
                with open("function_metrics.csv", "w", newline="", encoding="utf-8") as ff:
                    writer = csv.writer(ff)
                    writer.writerow(["Function", "FanIn", "FanOut", "Knots"])
                    all_funcs = set(function_metrics["fan_in"].keys()) | set(function_metrics["fan_out"].keys()) | set(function_metrics["knots"].keys())
                    for fn in sorted(all_funcs):
                        fi = function_metrics["fan_in"].get(fn, 0)
                        fo = function_metrics["fan_out"].get(fn, 0)
                        kn = function_metrics["knots"].get(fn, 0)
                        writer.writerow([fn, fi, fo, kn])
            # Also write a lizard-per-function summary if possible
            if functions:
                with open("lizard_functions.csv", "w", newline="", encoding="utf-8") as lf:
                    writer = csv.writer(lf)
                    writer.writerow(["File", "Function", "StartLine", "NLOC", "Cyclomatic"])
                    for fn in functions:
                        writer.writerow([fn.filename, fn.name, fn.start_line, fn.nloc, fn.cyclomatic_complexity])
            print("CSV outputs: class_metrics.csv, function_metrics.csv, lizard_functions.csv (if available)")
        except Exception as e:
            print(f"Failed to write CSVs: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cpp_metrics.py /path/to/repo")
        sys.exit(1)
    root = sys.argv[1]#ok
    main(root)

