import numpy as np
import re

# --- 1. Object-Oriented Model for the Structure ---
class Node:
    """Represents a single node in the 3D structure."""
    def __init__(self, id, x, y, z):
        self.id = int(id)
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.restraints = [False] * 6
        self.reactions = [0.0] * 6

class Element:
    """Represents a single beam/column element in the 3D structure."""
    def __init__(self, id, start_node, end_node, props):
        self.id = int(id)
        self.start_node = start_node
        self.end_node = end_node
        self.props = props 
        self.length = self.calculate_length()
        self.results = {} 

    def calculate_length(self):
        """Calculates the element's length based on its node coordinates."""
        return np.sqrt(
            (self.end_node.x - self.start_node.x)**2 +
            (self.end_node.y - self.start_node.y)**2 +
            (self.end_node.z - self.start_node.z)**2
        )

    # --- Stiffness and Transformation Matrix methods ---
    def get_local_stiffness_matrix(self):
        E, G = self.props['E'], self.props['G']
        A, Iyy, Izz, J = self.props['A'], self.props['Iyy'], self.props['Izz'], self.props['J']
        L = self.length
        L2, L3 = L**2, L**3
        k = np.zeros((12, 12))
        if L == 0: return k
        
        # Axial (0, 6)
        k[0,0] = k[6,6] = E*A/L; k[0,6] = k[6,0] = -E*A/L
        # Shear Y (1, 7) and Moment Z (5, 11)
        k[1,1] = k[7,7] = 12*E*Izz/L3; k[1,7] = k[7,1] = -12*E*Izz/L3
        k[1,5] = k[5,1] = 6*E*Izz/L2; k[7,5] = k[5,7] = -6*E*Izz/L2
        k[7,11] = k[11,7] = -6*E*Izz/L2; k[1,11] = k[11,1] = 6*E*Izz/L2
        k[5,5] = k[11,11] = 4*E*Izz/L; k[5,11] = k[11,5] = 2*E*Izz/L
        # Shear Z (2, 8) and Moment Y (4, 10)
        k[2,2] = k[8,8] = 12*E*Iyy/L3; k[2,8] = k[8,2] = -12*E*Iyy/L3
        k[2,4] = k[4,2] = 6*E*Iyy/L2; k[8,4] = k[4,8] = -6*E*Iyy/L2
        k[8,10] = k[10,8] = -6*E*Iyy/L2; k[2,10] = k[10,2] = 6*E*Iyy/L2
        k[4,4] = k[10,10] = 4*E*Iyy/L; k[4,10] = k[10,4] = 2*E*Iyy/L
        # Torsion X (3, 9)
        k[3,3] = k[9,9] = G*J/L; k[3,9] = k[9,3] = -G*J/L
        
        # Ensure symmetry
        for i in range(12):
            for j in range(i + 1, 12): k[j, i] = k[i, j]
        return k

    def get_transformation_matrix(self):
        T = np.zeros((12, 12))
        dx, dy, dz = self.end_node.x - self.start_node.x, self.end_node.y - self.start_node.y, self.end_node.z - self.start_node.z
        if self.length == 0: return np.identity(12)
        cx_x, cx_y, cx_z = dx / self.length, dy / self.length, dz / self.length
        is_column = np.isclose(dx, 0) and np.isclose(dy, 0)
        ref_vec = np.array([1, 0, 0]) if is_column else np.array([0, 0, 1])
        local_x_vec = np.array([cx_x, cx_y, cx_z])
        local_z_vec = np.cross(local_x_vec, ref_vec)
        if np.linalg.norm(local_z_vec) < 1e-6:
             ref_vec = np.array([1, 0, 0]) if not is_column and np.isclose(cx_z, 1) else np.array([0, 0, 1])
             local_z_vec = np.cross(local_x_vec, ref_vec)
        local_z_vec /= np.linalg.norm(local_z_vec)
        local_y_vec = np.cross(local_z_vec, local_x_vec)
        R = np.vstack([local_x_vec, local_y_vec, local_z_vec])
        for i in range(4): T[i*3:(i+1)*3, i*3:(i+1)*3] = R
        return T

class Structure:
    """Represents the entire 3D frame structure and handles the FEA."""
    def __init__(self):
        self.nodes, self.elements, self.dof_map = {}, {}, {}
        self.K_global, self.F_global, self.U_global = None, None, None

    def add_node(self, id, x, y, z):
        if id not in self.nodes: self.nodes[id] = Node(id, x, y, z)
        return self.nodes[id]

    def add_element(self, id, start_node_id, end_node_id, props):
        if id not in self.elements and start_node_id in self.nodes and end_node_id in self.nodes:
            self.elements[id] = Element(id, self.nodes[start_node_id], self.nodes[end_node_id], props)
        return self.elements.get(id)

    def set_support(self, node_id, restraints):
        if node_id in self.nodes: self.nodes[node_id].restraints = restraints

    def assemble_matrices(self):
        num_dof = len(self.nodes) * 6
        self.K_global, self.F_global = np.zeros((num_dof, num_dof)), np.zeros(num_dof)
        dof_index = 0
        for node_id in sorted(self.nodes.keys()):
            for i in range(6): self.dof_map[(node_id, i)] = dof_index; dof_index += 1
        for elem in self.elements.values():
            k_local, T = elem.get_local_stiffness_matrix(), elem.get_transformation_matrix()
            k_global_elem = T @ k_local @ T.T
            node_ids = [elem.start_node.id, elem.end_node.id]
            dof_indices = [self.dof_map[(nid, i)] for nid in node_ids for i in range(6)]
            for i, global_i in enumerate(dof_indices):
                for j, global_j in enumerate(dof_indices):
                    self.K_global[global_i, global_j] += k_global_elem[i, j]

    def add_element_loads(self, density_kn_m3, q_total_gravity_psf, wall_load_data):
        """Applies Self-Weight, Slab/Live Loads, and Explicit Wall Loads as UDLs."""
        
        for elem in self.elements.values():
            if elem.length == 0: continue
            
            # 1a. Element Self-Weight (q_self)
            q_self = density_kn_m3 * elem.props['A'] 

            # 1b. Slab/Live Load (q_slab_live) - Only for horizontal beams at level > 0
            q_slab_live = 0.0
            if np.isclose(elem.start_node.z, elem.end_node.z) and elem.start_node.z > 0:
                 # Assumes 1m tributary width for kN/m^2 to UDL (kN/m)
                q_slab_live = q_total_gravity_psf * 1.0 

            q_total_udl = q_self + q_slab_live
            
            # 2. Apply Wall Loads (q_wall): Overrides or supplements gravity load
            if elem.id in wall_load_data:
                # Total UDL for this element is the wall load. 
                # Assuming the explicit wall load INCLUDES the self-weight/slab load for simplicity in plotting, 
                # but adding it to the nodal forces to be conservative.
                q_total_udl += wall_load_data[elem.id]
            
            load_at_node_shear = q_total_udl * elem.length / 2

            # Apply equivalent nodal forces in global -Z direction (index 2 for Fz)
            self.F_global[self.dof_map[(elem.start_node.id, 2)]] -= load_at_node_shear
            self.F_global[self.dof_map[(elem.end_node.id, 2)]] -= load_at_node_shear
            
            # Store FINAL UDL for moment diagram calculation (Parabolic Fix)
            elem.results['q_udl_z'] = q_total_udl

    def solve(self):
        active_dofs = [self.dof_map[(n.id, i)] for n in self.nodes.values() for i in range(6) if not n.restraints[i]]
        active_dofs = np.array(active_dofs)
        K_reduced, F_reduced = self.K_global[active_dofs[:, np.newaxis], active_dofs], self.F_global[active_dofs]
        try:
            U_reduced = np.linalg.solve(K_reduced, F_reduced)
            self.U_global = np.zeros_like(self.F_global)
            self.U_global[active_dofs] = U_reduced
            return True, "Analysis successful."
        except np.linalg.LinAlgError:
            self.U_global = None
            return False, "Analysis failed. The structure may be unstable (singular matrix)."

    def calculate_element_results(self):
        if self.U_global is None: return
        keys = ['Axial_Start', 'Shear_Y_Start', 'Shear_Z_Start', 'Torsion_Start', 'Moment_Y_Start', 'Moment_Z_Start',
                'Axial_End', 'Shear_Y_End', 'Shear_Z_End', 'Torsion_End', 'Moment_Y_End', 'Moment_Z_End']
        for elem in self.elements.values():
            dof_indices = [self.dof_map[(nid, i)] for nid in [elem.start_node.id, elem.end_node.id] for i in range(6)]
            u_global_elem = self.U_global[dof_indices]
            u_local_elem = elem.get_transformation_matrix().T @ u_global_elem
            f_local = elem.get_local_stiffness_matrix() @ u_local_elem
            
            elem.results.update({keys[i]: f_local[i] for i in range(12)})
            elem.results['Max_Abs_Moment'] = max(abs(f_local[4]), abs(f_local[5]), abs(f_local[10]), abs(f_local[11]))

    def calculate_reactions(self):
        if self.U_global is None: return
        R = self.K_global @ self.U_global - self.F_global 
        for node in self.nodes.values():
            if any(node.restraints):
                for i in range(6):
                    if node.restraints[i]: node.reactions[i] = R[self.dof_map[(node.id, i)]]

# --- Utility Function for Properties (used by the main script) ---
def calculate_rc_properties(b, h, E, nu=0.2):
    A, Izz, Iyy, G = b*h, (b*h**3)/12, (h*b**3)/12, E/(2*(1+nu))
    a, c = max(b, h), min(b, h)
    # St. Venant's Torsion Constant (Simplified)
    J = a*(c**3)*(1/3 - 0.21*(c/a)*(1-(c**4)/(12*a**4)))
    return {'E':E, 'G':G, 'A':A, 'Iyy':Iyy, 'Izz':Izz, 'J':J}
