import streamlit as st
import numpy as np
import re
from structure_model import Structure, calculate_rc_properties # <-- Import from the FEA Engine

# --- Utility Functions ---

def parse_grid_input(input_string):
    """Parses grid input like '2x5.0, 6.0' into a list of span lengths."""
    if not input_string: return []
    lengths = []
    for segment in [s.strip() for s in input_string.split(',') if s.strip()]:
        match = re.match(r'^(\d+)x([0-9.]+)$', segment)
        if match:
            count, length = int(match.group(1)), float(match.group(2))
            if count > 0 and length > 0: lengths.extend([length] * count)
        else:
            try:
                if float(segment) > 0: lengths.append(float(segment))
            except ValueError: pass
    return lengths

def parse_and_apply_wall_loads(load_string):
    """Parses wall load input into a dictionary {element_id: load_q}."""
    wall_load_data = {}
    if not load_string: return wall_load_data
    for line in load_string.strip().split('\n'):
        line = line.strip()
        if not line or ':' not in line: continue
        parts = line.split(':')
        elem_ids_str = parts[0].strip()
        load_q_str = parts[1].strip()
        try:
            load_q = float(load_q_str)
            if load_q < 0: continue
        except ValueError: continue
        elem_ids = [int(segment.strip()) for segment in elem_ids_str.split(',') if segment.strip().isdigit()]
        for elem_id in elem_ids: wall_load_data[elem_id] = load_q
    return wall_load_data

# --- The Caching Function (Main Interface) ---

@st.cache_data
def generate_and_analyze_structure(x_dims, y_dims, z_dims, col_props, beam_props, q_total_gravity_psf, density_kn_m3, wall_load_data):
    """Generates the structure, runs FEA, and returns results in a dictionary."""
    s = Structure()
    x_coords, y_coords, z_coords = [0]+list(np.cumsum(x_dims)), [0]+list(np.cumsum(y_dims)), [0]+list(np.cumsum(z_dims))
    node_id, elem_id, node_map = 1, 1, {}

    # 1. Create Nodes and Supports
    for iz, z in enumerate(z_coords):
        for iy, y in enumerate(y_coords):
            for ix, x in enumerate(x_coords):
                s.add_node(node_id, x, y, z)
                node_map[(ix, iy, iz)] = node_id
                if np.isclose(z, 0): s.set_support(node_id, restraints=[True]*6)
                node_id += 1

    # 2. Create Elements (Columns, Beams in X, Beams in Y)
    for iz in range(len(z_coords)-1):
        for iy in range(len(y_coords)):
            for ix in range(len(x_coords)): s.add_element(elem_id, node_map[(ix,iy,iz)], node_map[(ix,iy,iz+1)], col_props); elem_id += 1
    for iz in range(1, len(z_coords)):
        for iy in range(len(y_coords)):
            for ix in range(len(x_coords)-1): s.add_element(elem_id, node_map[(ix,iy,iz)], node_map[(ix+1,iy,iz)], beam_props); elem_id += 1
    for iz in range(1, len(z_coords)):
        for iy in range(len(y_coords)-1):
            for ix in range(len(x_coords)): s.add_element(elem_id, node_map[(ix,iy+1,iz)], node_map[(ix,iy,iz)], beam_props); elem_id += 1
    
    # 3. Assemble and Apply Loads
    s.assemble_matrices()
    s.add_element_loads(density_kn_m3, q_total_gravity_psf, wall_load_data) 
    
    # 4. Solve and Calculate Results
    success, message = s.solve()
    if success: 
        s.calculate_element_results()
        s.calculate_reactions()
    
    if not success: return {'success': False, 'message': message}

    # Prepare results for return (Serializable dictionary)
    elements_data = [{'id':e.id, 'start_node_id':e.start_node.id, 'end_node_id':e.end_node.id, 'start_node_pos':(e.start_node.x,e.start_node.y,e.start_node.z), 'end_node_pos':(e.end_node.x,e.end_node.y,e.end_node.z), 'length':e.length, 'results':e.results} for e in s.elements.values()]
    
    return {
        'success': True, 'message': message,
        'nodes': [{'id':n.id, 'x':n.x, 'y':n.y, 'z':n.z, 'restraints':n.restraints, 'reactions':n.reactions} for n in s.nodes.values()],
        'elements': elements_data,
        'summary': {'num_nodes':len(s.nodes), 'num_elements':len(s.elements), 'k_shape':s.K_global.shape if s.K_global is not None else (0,0)}
    }
