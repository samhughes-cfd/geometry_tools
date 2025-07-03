import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def find_boundary_nodes(tris: np.ndarray) -> np.ndarray:
    """
    Given an (n_elems, 3) array of triangle indices, return the sorted
    array of node indices that lie on exactly one triangle edge.
    """
    # count all edges (undirected, so sort the node pair)
    edge_counts = {}
    for tri in tris:
        for i, j in [(0,1),(1,2),(2,0)]:
            e = tuple(sorted((int(tri[i]), int(tri[j]))))
            edge_counts[e] = edge_counts.get(e, 0) + 1

    # boundary edges occur only once
    boundary_nodes = set()
    for (i,j), count in edge_counts.items():
        if count == 1:
            boundary_nodes.add(i)
            boundary_nodes.add(j)
    return np.array(sorted(boundary_nodes), dtype=int)

def compute_shear_coefficient(geometry, G: float) -> float:
    """
    Compute Timoshenko shear‐correction factor κ for unit shear V=1.
    
    Parameters
    ----------
    geometry : sectionproperties.pre.geometry.Geometry or CompoundGeometry
        Must already have had create_mesh(...) called so that
        geometry.mesh.{points, triangles} (or dict) are available.
    G : float
        Shear modulus of the material (same units as required for U).

    Returns
    -------
    kappa : float
        The shear‐correction factor κ = A / (G·U), for a unit applied shear.
    """
    # 1) extract mesh data
    mesh = geometry.mesh
    if hasattr(mesh, "points"):
        verts = np.asarray(mesh.points)       # (n_pts, 2)
        tris  = np.asarray(mesh.triangles)    # (n_elems, 3)
    elif isinstance(mesh, dict):
        verts = np.asarray(mesh["vertices"])
        tris  = np.asarray(mesh["triangles"])
    else:
        raise ValueError("Unsupported mesh format")

    n_pts  = verts.shape[0]
    
    # 2) assemble global stiffness K and load b for Δφ = -1
    I = []; J = []; V = []
    b = np.zeros(n_pts, dtype=float)

    for tri in tris:
        i0, i1, i2 = tri
        p0, p1, p2 = verts[i0], verts[i1], verts[i2]
        # triangle area
        A = 0.5 * abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1]))
        # ∇N matrix (2×3)
        bmat = np.array([
            [ p1[1] - p2[1],  p2[1] - p0[1],  p0[1] - p1[1] ],
            [ p2[0] - p1[0],  p0[0] - p2[0],  p1[0] - p0[0] ],
        ]) / (2 * A)
        ke = A * (bmat.T @ bmat)  # 3×3 local stiffness

        # assemble
        for a_loc, Irow in enumerate((i0, i1, i2)):
            b[Irow] += -A/3.0   # ∫ N_i · (-1) dA = -A/3
            for b_loc, Jcol in enumerate((i0, i1, i2)):
                I.append(Irow)
                J.append(Jcol)
                V.append(ke[a_loc, b_loc])

    K = sp.coo_matrix((V, (I, J)), shape=(n_pts, n_pts)).tocsc()

    # 3) apply φ=0 on boundary nodes
    bnodes = find_boundary_nodes(tris)
    free = np.setdiff1d(np.arange(n_pts), bnodes)
    Kff  = K[free, :][:, free]
    bf   = b[free]

    # 4) solve for φ
    φ = np.zeros(n_pts, dtype=float)
    φ_free = spla.spsolve(Kff, bf)
    φ[free] = φ_free

    # 5) compute the strain energy U = ½ ∫ |∇φ|² dA
    U = 0.0
    for tri in tris:
        i0, i1, i2 = tri
        p0, p1, p2 = verts[i0], verts[i1], verts[i2]
        A = 0.5 * abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1]))
        bmat = np.array([
            [ p1[1] - p2[1],  p2[1] - p0[1],  p0[1] - p1[1] ],
            [ p2[0] - p1[0],  p0[0] - p2[0],  p1[0] - p0[0] ],
        ]) / (2 * A)
        φe = φ[[i0, i1, i2]]
        gradφ = bmat @ φe
        U += 0.5 * A * (gradφ @ gradφ)

    # 6) total area
    A_total = geometry.calculate_area()

    # 7) shear‐correction factor κ = A / (G·U) for V=1
    kappa = A_total / (G * U)
    return kappa