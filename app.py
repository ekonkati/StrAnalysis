import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Import modular functions and utilities ---
# Ensure these files are in the same directory: structure_model.py, fea_utils.py
from fea_utils import generate_and_analyze_structure, parse_grid_input, parse_and_apply_wall_loads
from structure_model import calculate_rc_properties 

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Improved 3D Frame Analyzer")

# --- 1. Utility for Plotting ---

def get_hover_text(elem):
    """Generates detailed hover text for an element."""
    text = f"**Element {elem['id']}** (L={elem.get('length', 0.0):.2f}m)<br>"
    for key, value in elem['results'].items():
        if not key.startswith('Disp') and not key.startswith('Max') and not key.startswith('q_udl'):
            unit = 'kNm' if key.startswith('Moment') or key.startswith('Torsion') else 'kN'
            text += f"{key.replace('_', ' ')}: {value:.2f} {unit}<br>"
    text += f"Total UDL qz: {elem['results'].get('q_udl_z', 0.0):.2f} kN/m"
    return text

# --- 2. 3D Plotting Function (NEW) ---

def plot_3d_frame(nodes, elements, display_mode='Structure'):
    """Plots the 3D frame structure and optionally colors elements by Max Abs Moment."""
    fig = go.Figure()
    
    # 1. Define max values for scaling colors
    max_abs_moment = max(e['results'].get('Max_Abs_Moment', 0) for e in elements) if elements else 1e-6
    
    # 2. Iterate through elements to draw lines
    for elem in elements:
        # Find the node objects based on ID (slower, but necessary for visualization data)
        n1 = next(n for n in nodes if n['id'] == elem['start_node_id'])
        n2 = next(n for n in nodes if n['id'] == elem['end_node_id'])
        
        x_coords = [n1['x'], n2['x']]
        y_coords = [n1['y'], n2['y']]
        z_coords = [n1['z'], n2['z']]
        
        # Determine color based on mode
        line_color = 'darkblue'
        
        if display_mode == 'Max Bending Moment':
            moment = elem['results'].get('Max_Abs_Moment', 0)
            # Use Plotly's color mapping for Viridis scale (0=Blue, Max=Yellow)
            # The Viridis scale is often used for technical data
            normalized_value = moment / max_abs_moment
            # Scale from 0 to 1 for the color gradient
            line_color = f'hsl({255 * (1 - normalized_value)}, 100%, 50%)' 

        fig.add_trace(go.Scatter3d(
            x=x_coords, y=y_coords, z=z_coords,
            mode='lines',
            line=dict(color=line_color, width=5),
            hoverinfo='text',
            hovertext=get_hover_text(elem),
            name=f"Element {elem['id']}",
            showlegend=False
        ))

    # 3. Add node markers
    fig.add_trace(go.Scatter3d(
        x=[n['x'] for n in nodes], y=[n['y'] for n in nodes], z=[n['z'] for n in nodes],
        mode='markers',
        marker=dict(size=5, color='purple', opacity=0.8),
        hoverinfo='text',
        hovertext=[f"Node {n['id']}<br>({n['x']:.2f}, {n['y']:.2f}, {n['z']:.2f})m" for n in nodes],
        name='Nodes',
        showlegend=False
    ))
    
    # 4. Update layout
    fig.update_layout(
        title=f"3D Frame View ({display_mode})",
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
            aspectmode='data', # Ensures true scale
        ),
        height=700
    )
    return fig


# --- 3. 2D Plotting Function (Unchanged, for context) ---

def plot_2d_frame(nodes, elements, plane_axis, coordinate, display_mode, show_values):
    # ... (This function is long and unchanged, omitting here for brevity)
    # ... (It must be included in the final app.py file)
    # Re-insert the full 'plot_2d_frame' implementation from the previous step here.
    # [START OF plot_2d_frame code - not shown here]
    
    fig = go.Figure()
    
    # Filter nodes for the selected plane
    if plane_axis == 'Y': 
        plane_nodes_list, x_key, z_key = [n for n in nodes if np.isclose(n['y'], coordinate)], 'x', 'z'
    else: 
        plane_nodes_list, x_key, z_key = [n for n in nodes if np.isclose(n['x'], coordinate)], 'y', 'z'
        
    plane_node_ids = {n['id'] for n in plane_nodes_list}

    result_map = {
        'Structure': (None, None, None, 1, 'black'),
        'Bending Moment': ('Moment_Z', 'kNm', 'Deflection in X', 0.2, 'red'), 
        'Shear Force': ('Shear_Z', 'kN', 'Deflection in X', 0.2, 'green'),  
        'Axial Force': ('Axial', 'kN', 'Deflection in X', 0.2, 'blue'),
        'Deflection': (None, 'm', 'Deflection in Z', 100, 'orange'), 
    }
    
    result_key_base, unit, defl_key, _, diagram_color = result_map.get(display_mode, (None, None, None, 1, 'black'))

    x_coords = [n[x_key] for n in plane_nodes_list]
    z_coords_all = [n[z_key] for n in plane_nodes_list]
    max_dim = max(max(x_coords, default=1) - min(x_coords, default=0), 
                  max(z_coords_all, default=1) - min(z_coords_all, default=0)) or 1
    
    # --- Scaling Calculation ---
    max_abs_result = 0.0
    relevant_elements = [e for e in elements if e['start_node_id'] in plane_node_ids and e['end_node_id'] in plane_node_ids]
    
    if display_mode != 'Structure':
        for elem in relevant_elements:
            res = elem['results']
            max_abs_result = max(max_abs_result, abs(res.get(f'{result_key_base}_Start', 0.0)))
            max_abs_result = max(max_abs_result, abs(res.get(f'{result_key_base}_End', 0.0)))
            
            # CHECK MID-SPAN MOMENT FOR PARABOLIC DIAGRAM (FIX)
            if display_mode == 'Bending Moment' and elem['length'] > 0:
                w_eff = res.get('q_udl_z', 0.0)
                if w_eff > 1e-3: 
                    L = elem['length']
                    M_start = res.get('Moment_Z_Start', 0.0)
                    V_start = res.get('Shear_Z_Start', 0.0)
                    x_at_V_zero = V_start / w_eff 
                    x_at_V_zero = np.clip(x_at_V_zero, 0, L)
                    M_mid = M_start + V_start * x_at_V_zero - w_eff * x_at_V_zero**2 / 2
                    max_abs_result = max(max_abs_result, abs(M_mid))

        max_abs_result = max(max_abs_result, 1e-3)
        
    global_scale = (max_dim / 8.0) / max_abs_result if display_mode != 'Structure' and display_mode != 'Deflection' else 1.0
    
    if display_mode == 'Deflection':
        max_abs_defl = max(abs(e['results'].get('Disp_Global_Start', np.zeros(6))[2]) for e in relevant_elements)
        max_abs_defl = max(max_abs_defl, max(abs(e['results'].get('Disp_Global_End', np.zeros(6))[2]) for e in relevant_elements)) or 1e-6
        global_scale = max_dim / (max_abs_defl * 10) 

    # --- Plotting Elements and Diagrams ---
    for elem in relevant_elements:
        start_pos, end_pos = elem['start_node_pos'], elem['end_node_pos']
        
        start_x_plot = start_pos[0 if plane_axis=='Y' else 1]
        end_x_plot = end_pos[0 if plane_axis=='Y' else 1]
        start_z, end_z = start_pos[2], end_pos[2]
        
        x_coords_frame = [start_x_plot, end_x_plot]
        z_coords_frame = [start_z, end_z]
        L = elem['length']
        
        # BASE ELEMENT LINE
        fig.add_trace(go.Scatter(x=x_coords_frame, y=z_coords_frame, mode='lines', line=dict(color='darkblue', width=3), 
                                 hoverinfo='text', hovertext=get_hover_text(elem), showlegend=False))
        
        # RESULT DIAGRAMS
        if display_mode != 'Structure':
            num_points = 51 if display_mode in ['Bending Moment', 'Shear Force'] else 21
            local_x = np.linspace(0, L, num_points)
            diagram_coords, value_at_point = [], []

            w_eff = elem['results'].get('q_udl_z', 0.0)
            M_start = elem['results'].get('Moment_Z_Start', 0.0)
            V_start = elem['results'].get('Shear_Z_Start', 0.0)
            Axial_start = elem['results'].get('Axial_Start', 0.0)
            Axial_end = elem['results'].get('Axial_End', 0.0)
                
            for x in local_x:
                value = 0.0
                if L == 0: continue
                
                if result_key_base == 'Moment_Z':
                    value = M_start + V_start * x - w_eff * x**2 / 2
                elif result_key_base == 'Shear_Z':
                    value = V_start - w_eff * x
                elif result_key_base == 'Axial':
                    value = Axial_start + (Axial_end - Axial_start) * (x / L)
                
                value_at_point.append(value)

                # Convert Local (x, value) to Global Plotting Coordinates
                if L > 0: nx, nz = (x_coords_frame[1] - x_coords_frame[0]) / L, (z_coords_frame[1] - z_coords_frame[0]) / L
                else: nx, nz = 1, 0
                nx_perp, nz_perp = -nz, nx 
                
                x_mid = x_coords_frame[0] + nx * x
                z_mid = z_coords_frame[0] + nz * x
                
                plot_value = value * global_scale * (-1 if display_mode in ['Bending Moment', 'Shear Force'] else 1)

                x_plot = x_mid + nx_perp * plot_value
                z_plot = z_mid + nz_perp * plot_value
                
                diagram_coords.append((x_plot, z_plot))

            diag_x = [c[0] for c in diagram_coords]
            diag_z = [c[1] for c in diagram_coords]

            is_horizontal = np.isclose(start_z, end_z)
            
            fill_mode = 'tozeroy' if is_horizontal and display_mode != 'Deflection' else 'tozerox' if display_mode != 'Deflection' else None
            fig.add_trace(go.Scatter(x=diag_x, y=diag_z, mode='lines', line=dict(color=diagram_color, width=2, dash='solid' if display_mode != 'Deflection' else 'dash'), 
                                     fill=fill_mode, 
                                     opacity=0.3 if display_mode != 'Deflection' else 1,
                                     hoverinfo='none', name=f"{display_mode} E{elem['id']}"))
            
            # Plot Values
            if show_values and display_mode != 'Deflection' and abs(max(value_at_point, key=abs)) > 1e-3:
                label_indices = [0, len(local_x) - 1] 
                if len(value_at_point) > 2: label_indices.append(np.argmax(np.abs(value_at_point)))
                
                for idx in sorted(list(set(label_indices))):
                    val = value_at_point[idx]
                    if abs(val) < 1e-1: continue 
                    
                    fig.add_annotation(
                        x=diag_x[idx], y=diag_z[idx],
                        text=f"{val:.1f}", showarrow=False,
                        xshift=nx_perp * (val * global_scale * 10 * (-1 if display_mode in ['Bending Moment', 'Shear Force'] else 1)),
                        yshift=nz_perp * (val * global_scale * 10 * (-1 if display_mode in ['Bending Moment', 'Shear Force'] else 1)),
                        font=dict(color='black', size=10, weight='bold'),
                        bgcolor="rgba(255, 255, 255, 0.7)", bordercolor=diagram_color, borderwidth=1, borderpad=2
                    )
                    

    # Node Markers and Layout
    fig.add_trace(go.Scatter(x=[n[x_key] for n in plane_nodes_list], y=[n[z_key] for n in plane_nodes_list], 
                             mode='markers', marker=dict(size=8, color='purple'), name='Nodes', 
                             text=[f"Node {n['id']}" for n in plane_nodes_list], hoverinfo='text', showlegend=False))

    title_scale = f" (Scale: 1:{int(max_dim/(max_abs_result/8.0)):,})" if display_mode in ['Bending Moment', 'Shear Force', 'Axial Force'] else ""
    title_scale_def = f" (Exaggerated Scale)" if display_mode == 'Deflection' else ""
    
    fig.update_layout(title=f"2D Elevation: **{display_mode}** ({unit}) on {plane_axis.replace('Y', 'X-Z').replace('X', 'Y-Z')} Plane at {plane_axis}={coordinate}m{title_scale}{title_scale_def}", 
                      xaxis_title=f'{x_key.upper()}-axis (m)', yaxis_title='Z-axis (m)', showlegend=False)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

# [END OF plot_2d_frame code]

# --- 4. Main Streamlit App UI ---
st.title("🏗️ 2D/3D Frame Analyzer")
st.write("Define your building grid, sections, and loads to generate and analyze a 3D frame.")

with st.sidebar:
    st.header("1. Frame Geometry")
    x_grid = st.text_input("X-spans (m)", "7.0") 
    y_grid = st.text_input("Y-spans (m)", "0.001") 
    z_grid = st.text_input("Z-heights (m)", "3.5, 3.5")
    
    st.header("2. Section & Material")
    E = st.number_input("E (GPa)", 30.0, help="E=30 GPa for RC")*1e6 
    density_kn_m3 = st.number_input("Material Density (kN/m³)", 25.0, help="25 kN/m³ for Concrete")
    with st.expander("Column & Beam Sizes"):
        col_b, col_h = st.number_input("Col b (mm)", 240)/1000, st.number_input("Col h (mm)", 240)/1000
        beam_b, beam_h = st.number_input("Beam b (mm)", 120)/1000, st.number_input("Beam h (mm)", 240)/1000
        
    st.header("3. Gravity Loads (kN/m²)")
    with st.expander("Slab, Finishes & Live Loads:"):
        slab_d, slab_t = st.number_input("Slab Density (kN/m³)", 25.0), st.number_input("Slab Thickness (m)", 0.150)
        fin_l, live_l = st.number_input("Finishes (kN/m²)", 1.5), st.number_input("Live Load (kN/m²)", 3.0)
        
    st.subheader("4. Specific Brick/Wall Uniform Loads (kN/m)")
    wall_load_input = st.text_area("Element Loads (e.g., 2, 3: 5.0)", 
                                   value="3, 4: 10.0", 
                                   height=100,
                                   help="Enter Element IDs and the load magnitude (kN/m in -Z direction).")
    
    analyze_button = st.button("Generate & Analyze Frame", type="primary")

if analyze_button:
    x_dims, y_dims, z_dims = parse_grid_input(x_grid), parse_grid_input(y_grid), parse_grid_input(z_grid)
    wall_load_data = parse_and_apply_wall_loads(wall_load_input)
    
    if not all([x_dims, y_dims, z_dims]): 
        st.error("Invalid grid input.")
    else:
        # Properties calculation needs to be here as it uses sidebar inputs
        col_p = calculate_rc_properties(col_b, col_h, E)
        beam_p = calculate_rc_properties(beam_b, beam_h, E)
        q_total_gravity_psf = slab_d*slab_t + fin_l + live_l
        
        with st.spinner("Running Finite Element Analysis..."):
            analysis_results = generate_and_analyze_structure(
                x_dims, y_dims, z_dims, col_p, beam_p, 
                q_total_gravity_psf, density_kn_m3, wall_load_data
            )
                                                              
        if not analysis_results['success']: 
            st.error(f"Analysis Failed: {analysis_results['message']}")
        else:
            st.success("Analysis complete!")
            st.session_state['analysis_results'] = analysis_results

# Display Results
if 'analysis_results' in st.session_state and st.session_state['analysis_results']['success']:
    results, nodes, elements = st.session_state['analysis_results'], st.session_state['analysis_results']['nodes'], st.session_state['analysis_results']['elements']
    
    # NEW TAB ORDER
    tab1, tab2, tab3, tab4 = st.tabs(["3D View", "2D Elevation View", "Support Reactions", "Detailed Element Results"])
    
    # 3D View Tab
    with tab1:
        st.subheader("3D Structure Visualization")
        col_3d_1, col_3d_2 = st.columns([0.5, 0.5])
        display_mode_3d = col_3d_1.selectbox("Color Code By:", ('Structure', 'Max Bending Moment'), key='display_mode_3d')
        st.plotly_chart(plot_3d_frame(nodes, elements, display_mode_3d), use_container_width=True)

    # 2D Elevation View Tab
    with tab2:
        st.subheader("2D Elevation View and Result Diagrams")
        
        col_2d_1, col_2d_2, col_2d_3 = st.columns([0.3, 0.4, 0.3])
        
        plane_axis_display = col_2d_1.radio("Grid Plane", ('X-Z (Y-Gridline)', 'Y-Z (X-Gridline)'), key='plane_axis_radio_1')
        diagram_options = ['Bending Moment', 'Shear Force', 'Deflection', 'Axial Force', 'Structure']
        display_mode_2d = col_2d_2.selectbox("Result Diagram", diagram_options, key='display_mode_2d_1')
        show_values = col_2d_3.checkbox("Show Values", key='show_values_1', value=True)
        
        if plane_axis_display == 'X-Z (Y-Gridline)':
            y_coords = sorted(list(set(n['y'] for n in nodes)))
            coordinate = y_coords[0] if y_coords else 0
            plane_key = 'Y'
        else:
            x_coords = sorted(list(set(n['x'] for n in nodes)))
            coordinate = x_coords[0] if x_coords else 0
            plane_key = 'X'
        
        st.info(f"Displaying **{display_mode_2d}** on the **{plane_key}-Gridline** at coordinate **{coordinate:.3f} m**.")
        
        st.plotly_chart(plot_2d_frame(nodes, elements, plane_key, coordinate, display_mode_2d, show_values), use_container_width=True)

    # Support Reactions Tab
    with tab3:
        st.subheader("Support Reactions (Global X, Y, Z)")
        support_nodes = {n['id']: n for n in nodes if any(n['restraints'])}
        if support_nodes:
            node_id = st.selectbox("Select support node", options=sorted(list(support_nodes.keys())), key='support_node_select')
            st.dataframe(pd.DataFrame({
                "DOF": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"], 
                "Value (kN, kNm)": support_nodes[node_id]['reactions']
            }).round(2), use_container_width=True)
        else: st.write("No support nodes found.")
        
    # Detailed Element Results Tab
    with tab4:
        st.subheader("All Element End Forces & Moments (Local Coordinates)")
        data = []
        for e in elements:
            res = e['results']
            data.append({
                'ID': e['id'], 
                'Start Node': e['start_node_id'], 
                'End Node': e['end_node_id'], 
                'Length (m)': e.get('length', 0.0), 
                'UDL (kN/m)': res.get('q_udl_z', 0.0),
                'Max |M| (kNm)': res.get('Max_Abs_Moment', 0),
                'Axial Start (kN)': res.get('Axial_Start', 0), 
                'Moment Z Start (kNm)': res.get('Moment_Z_Start', 0),
                'Moment Z End (kNm)': res.get('Moment_Z_End', 0),
            })
        st.dataframe(pd.DataFrame(data).round(2), use_container_width=True)
