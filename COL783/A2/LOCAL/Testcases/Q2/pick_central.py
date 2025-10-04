import networkx as nx
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

gfile = Path(sys.argv[1])  # e.g. docs/diagrams/all_classes.graphml
top_k = int(sys.argv[2]) if len(sys.argv)>2 else 10

# Parse GraphML manually to handle nested graphs and build qualified names
def get_all_nodes_with_qualified_names(file_path):
    """Extract all nodes from GraphML including nested graphs with fully qualified names"""
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Find the namespace
    ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
    
    # Build key mapping
    key_mapping = {}
    for key in root.findall('.//graphml:key', ns):
        key_id = key.get('id')
        attr_name = key.get('attr.name')
        key_mapping[key_id] = attr_name
    
    # Extract all nodes with qualified names
    node_qualified_names = {}
    
    def extract_from_graph(graph_elem, namespace_path=[]):
        for node in graph_elem.findall('graphml:node', ns):
            node_id = node.get('id')
            name = None
            node_type = None
            
            # Get node data
            for data in node.findall('graphml:data', ns):
                key = data.get('key')
                if key in key_mapping:
                    attr_name = key_mapping[key]
                    if attr_name == 'name':
                        name = data.text
                    elif attr_name == 'type':
                        node_type = data.text
            
            if name:
                # Build qualified name based on namespace path and node type
                if node_type == 'namespace':
                    # This is a namespace node - add to path for children
                    qualified_name = '::'.join(namespace_path + [name])
                    node_qualified_names[node_id] = qualified_name
                    new_namespace_path = namespace_path + [name]
                else:
                    # This is a class/enum/etc - build full qualified name
                    if namespace_path:
                        qualified_name = '::'.join(namespace_path) + '::' + name
                    else:
                        qualified_name = name
                    node_qualified_names[node_id] = qualified_name
                    new_namespace_path = namespace_path
            else:
                new_namespace_path = namespace_path
            
            # Check for nested graph
            nested = node.find('graphml:graph', ns)
            if nested is not None:
                extract_from_graph(nested, new_namespace_path)
    
    # Start extraction
    root_graph = root.find('.//graphml:graph', ns)
    if root_graph is not None:
        extract_from_graph(root_graph, [])
    
    return node_qualified_names

# Get node names with full qualification
node_names_dict = get_all_nodes_with_qualified_names(gfile)

# Read graph with NetworkX (this will only get top-level structure)
try:
    G = nx.read_graphml(gfile)
except:
    # If NetworkX fails, create empty graph and add nodes manually
    G = nx.DiGraph()
    for node_id in node_names_dict:
        G.add_node(node_id)

# Enhanced node_name function
def node_name(n):
    # First try the manually extracted qualified names
    if n in node_names_dict:
        return node_names_dict[n]
    # Fall back to NetworkX node attributes
    return G.nodes[n].get('name') or G.nodes[n].get('display_name') or n

# compute measures (only if we have a connected graph)
assert G.nodes is not None 
if len(G.nodes()) > 0:
    deg = dict(G.degree())
    pr = nx.pagerank(G) if len(G) > 0 and G.number_of_edges() > 0 else {n: 1.0/len(G) for n in G}
    bt = nx.betweenness_centrality(G, normalized=True) if G.number_of_edges() > 0 else {n: 0.0 for n in G}
else:
    deg = {}
    pr = {}
    bt = {}

# If NetworkX didn't capture all nodes, add them with default scores
all_node_ids = set(node_names_dict.keys()) | set(G.nodes())
for node_id in all_node_ids:
    if node_id not in G.nodes():
        G.add_node(node_id)
    if node_id not in deg:
        deg[node_id] = 0
    if node_id not in pr:
        pr[node_id] = 1.0 / len(all_node_ids)
    if node_id not in bt:
        bt[node_id] = 0.0

# normalize each score to [0,1]
def normalize(d):
    if not d: return d
    vals = list(d.values())
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return {k: 0.0 for k in d}
    return {k: (v - mn) / (mx - mn) for k, v in d.items()}

nd = normalize(deg)
np_ = normalize(pr)
nb = normalize(bt)

# combined score (simple average)
scores = {n: (nd.get(n, 0) + np_.get(n, 0) + nb.get(n, 0)) / 3.0 for n in all_node_ids}

# Filter out namespace-only entries for final output (optional - remove if you want namespaces too)
def is_namespace_only(qualified_name):
    """Check if this is just a namespace (not a class/template/enum)"""
    # Look up the original node to check its type
    for node_id, name in node_names_dict.items():
        if name == qualified_name:
            # Check if this node was marked as namespace type
            node_attrs = G.nodes.get(node_id, {})
            return node_attrs.get('type') == 'namespace'
    return False

# pick top_k by score, optionally filtering out pure namespaces
filtered_scores = {n: score for n, score in scores.items() 
                  if not is_namespace_only(node_name(n))}

top = sorted(filtered_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

for node, score in top:
    print(node_name(node))
