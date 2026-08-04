#!/usr/bin/env python3
"""
Helper script to export an existing visual Action Graph in Isaac Sim/Omniverse
into programmatic Python syntax (og.Controller.edit) for comparison or script generation.

How to use:
1. Open the Script Editor in Isaac Sim (Window -> Script Editor).
2. Copy and paste this script (or load/import it).
3. Call: export_action_graph_to_python("/World/ActionGraph")
"""

# Safe imports for standalone runs
try:
    import omni.usd
    from pxr import Usd, Sdf, Gf
except ImportError:
    pass

def format_usd_value(val):
    """Recursively formats USD, Sdf, and Gf types into standard Python types."""
    if isinstance(val, (str, bytes)):
        return val
    elif isinstance(val, Sdf.Path):
        return str(val)
    elif isinstance(val, Sdf.AssetPath):
        return val.path
    elif isinstance(val, (list, tuple)):
        return [format_usd_value(x) for x in val]
    elif hasattr(val, "__getitem__") and hasattr(val, "__len__"):
        # Handles Gf vectors (Vec3d, Vec3f, Vec2d, etc.)
        return [format_usd_value(val[i]) for i in range(len(val))]
    return val

def export_action_graph_to_python(graph_path="/World/ActionGraph"):
    """
    Traverses the specified graph prim in USD and prints the og.Controller.edit syntax.
    """
    stage = omni.usd.get_context().get_stage()
    graph_prim = stage.GetPrimAtPath(graph_path)
    
    if not graph_prim.IsValid():
        print(f"❌ Error: Graph prim at '{graph_path}' not found.")
        return None
        
    nodes_info = []
    connections = []
    
    # Iterate through child prims (individual nodes in the graph)
    for child in graph_prim.GetChildren():
        node_name = child.GetName()
        
        # Retrieve the node type (stored as attribute or metadata)
        node_type = ""
        node_type_attr = child.GetAttribute("nodeType")
        if node_type_attr.IsValid():
            node_type = node_type_attr.Get()
        if not node_type:
            node_type = child.GetMetadata("nodeType")
        if not node_type:
            # Fallback to type name
            node_type = child.GetTypeName()
            
        set_values = []
        for prop in child.GetProperties():
            if not isinstance(prop, Usd.Attribute):
                continue
                
            attr_name = prop.GetName()
            # We only look at inputs (since outputs are driven or connected downstream)
            if attr_name.startswith("inputs:"):
                # Check for connections
                targets = prop.GetConnections()
                if targets:
                    for target in targets:
                        # Convert target Sdf.Path representation robustly
                        target_str = str(target)
                        if '.' in target_str:
                            path_part, source_attr = target_str.rsplit('.', 1)
                            source_node_name = path_part.split('/')[-1]
                        else:
                            # Fallback if structure is different
                            try:
                                source_node_name = target.GetParentPath().name
                                source_attr = target.name
                            except AttributeError:
                                source_node_name = str(target.GetParentPath()).split('/')[-1]
                                source_attr = str(target).split('.')[-1]
                                
                        connections.append((
                            f"{source_node_name}.{source_attr}", 
                            f"{node_name}.{attr_name}"
                        ))
                else:
                    # Non-connected authored values (explicitly set values)
                    if prop.IsAuthored():
                        # Skip execution triggers
                        if attr_name == "inputs:execIn":
                            continue
                        val = prop.Get()
                        if val is not None:
                            formatted_val = format_usd_value(val)
                            set_values.append((f"{node_name}.{attr_name}", formatted_val))
                            
        nodes_info.append((node_name, node_type, set_values))

    # Generate the python script
    lines = []
    lines.append("import omni.graph.core as og")
    lines.append("")
    lines.append("keys = og.Controller.Keys")
    lines.append("og.Controller.edit(")
    lines.append("    {")
    lines.append(f'        "graph_path": "{graph_path}",')
    lines.append('        "evaluator_name": "execution",')
    lines.append("    },")
    lines.append("    {")
    
    # ── 1. Create Nodes ───────────────────────────────────────────────────────
    lines.append("        keys.CREATE_NODES: [")
    for name, n_type, _ in sorted(nodes_info):
        lines.append(f'            ("{name}", "{n_type}"),')
    lines.append("        ],")
    lines.append("")
    
    # ── 2. Set Values ─────────────────────────────────────────────────────────
    lines.append("        keys.SET_VALUES: [")
    for name, _, vals in sorted(nodes_info):
        if vals:
            lines.append(f'            # Node: {name}')
            for path, val in sorted(vals):
                lines.append(f'            ("{path}", {repr(val)}),')
    lines.append("        ],")
    lines.append("")
    
    # ── 3. Connections ────────────────────────────────────────────────────────
    lines.append("        keys.CONNECT: [")
    for src, dst in sorted(connections):
        lines.append(f'            ("{src}", "{dst}"),')
    lines.append("        ],")
    
    lines.append("    }")
    lines.append(")")
    
    output_code = "\n".join(lines)
    print("\n" + "="*30 + " GENERATED PYTHON CODE " + "="*30)
    print(output_code)
    print("="*83 + "\n")
    return output_code

if __name__ == "__main__":
    import argparse
    
    # Try importing omni.usd to determine if we are in GUI or standalone terminal
    standalone = False
    try:
        import omni.usd
        from pxr import Usd, Sdf, Gf
    except ImportError:
        standalone = True
        
    if standalone:
        parser = argparse.ArgumentParser(description="Export Action Graph from USD scene to Python script.")
        parser.add_argument("usd_path", help="Path to the USD file containing the Action Graph.")
        parser.add_argument("--graph", default="/World/ActionGraph", help="Path of the Action Graph in the USD stage.")
        args = parser.parse_args()
        
        print(f"Initializing Isaac Sim SimulationApp in headless mode...")
        try:
            from isaacsim import SimulationApp
        except ImportError:
            from omni.isaac.kit import SimulationApp
            
        simulation_app = SimulationApp({"headless": True})
        
        # Now we can import omni and pxr safely
        import omni.usd
        from pxr import Usd, Sdf, Gf
        
        print(f"Opening USD stage: {args.usd_path}")
        omni.usd.get_context().open_stage(args.usd_path)
        
        export_action_graph_to_python(args.graph)
        
        simulation_app.close()
    else:
        # Default behavior when run in Isaac Sim Script Editor
        export_action_graph_to_python("/World/ActionGraph")
