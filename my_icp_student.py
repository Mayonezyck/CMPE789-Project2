import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree, distance, transform
import time
def load_ply(file_path):
    # Load a .ply file using open3d as an numpy.array
    loaded_pc = o3d.io.read_point_cloud(file_path)
    #print(f"Loaded {file_path}: {len(loaded_pc.points)} points")
    #pc_downsampled = loaded_pc.voxel_down_sample(voxel_size=4)
    #print(f"Downsampled: {len(pc_downsampled.points)} points")
    point_cloud = np.asarray(loaded_pc.points)
    #print(f"Downsampled: {len(pc_downsampled.points)} points")
    # return np.array
    return point_cloud

def find_closest_points(source_points, target_points):
    # Align points in the souce and target point cloud data
    kdtree = cKDTree(target_points)
    distances, indices = kdtree.query(source_points)
    return indices

def estimate_normals(points, k_neighbors=30):
    """
    Use open3d to do it, e.g. estimate_normals()
    k_neighbors: The number of nearest neighbors to consider when estimating normals (you can change the value)
    """
    #kdtree = cKDTree(points)
    # have to convert to o3d point cloud since it doesnt work with kdtree
    source_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(points)

    source_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))
    #add this to orient nomral
    source_cloud.orient_normals_consistent_tangent_plane(k_neighbors)
    norms = np.asarray(source_cloud.normals)
    return norms

def normal_shooting(source_points, source_normals, target_points, target_normals):
    # project along normal, intersect other point set to find a correspondence
    # TODO: this diesn't use the target normals because i dont know how to use them
    # making ckdtree of points
    target_tree = cKDTree(target_points)
    # start with normal array since dont know how many points it will be
    matched_target_points = []
    step_size = 0.01
    max_dist  = 2
    num_steps = int(max_dist / step_size) # i.e. 2 / 0.01 = 200

    # have to loop through the points and normals at the same time
    for pt, norm in zip(source_points, source_normals):
        # for every point and normal, find a bunch of points along their norms
        points = [pt + step_size * currentStep * norm for currentStep in range(num_steps)]
        npPoints = np.array(points)
        # find the distance from target tree to these points
        distances, indices = target_tree.query(npPoints)
        # this gets index at closest point
        # you cant actually use indices object for this (i tried)
        closest_point_index = np.argmin(distances)
        # gets the closest point and adds it to the array
        closest_point = target_points[indices[closest_point_index]]
        # matched_target_points.append(closest_point)
        # after finding closest_point_index
        closest_point = target_points[indices[closest_point_index]]
        closest_normal = target_normals[indices[closest_point_index]]

        # check normal consistency
        if np.dot(norm, closest_normal) > 0.8:  
            matched_target_points.append(closest_point)
        else:
            _, j2 = target_tree.query(pt, k=1)
            matched_target_points.append(target_points[j2])
    # returns closest points
    return np.array(matched_target_points)


def point_to_plane(source_points, target_points, target_normals):
    # still finding closest points, but now using point to plane as the metric
    #target_tree = cKDTree(target_points)
    #distances, indices = target_tree.query(source_points)
    # closest_target_points = target_points[indices]

    # oh wait we can just use the function we already have for this
    indices = find_closest_points(source_points, target_points)
    closest_target_points = target_points[indices]
    closest_target_normals = target_normals[indices]

    return closest_target_points, closest_target_normals
 
def compute_transformation(source_points, target_points):
    # Compute the optimal rotation matrix R and translation vector t that align source_points with matched_target_points
    
    #print(source_points.shape)
    #print(target_points.shape)
    # get X0 and y0, theres no weights so its basically just the mean
    
    #x_0 = source_points.mean(axis=0)
    #y_0 = target_points.mean(axis=0)

    # weighting based on centroid might help
    centroid = np.mean(source_points, axis=0)
    weights = np.linalg.norm(source_points-centroid, axis=1)**2
    weights = weights / weights.sum()
    
    x_0 = np.average(source_points, axis=0, weights=weights)
    y_0 = np.average(target_points, axis=0, weights=weights)
    x_n = source_points - x_0
    y_n = target_points - y_0

    # computing cross covariance matrix based on mean reduced coordinates (according to slides)
    # use @ operator for matrix multiplication
    # solve for H
    #H = x_n.T @ y_n
    H = (x_n * weights[:, None]).T @ y_n

    # now do singular value decomposition
    # thank goodness numpy does this for us lol
    # svd(H) = USVT
    U, S, VT = np.linalg.svd(H)

    # R = V * UT
    R = VT.T @ U.T

    # now that we have rotation matrix, get translation vector and should be done
    t = y_0 - (R @ x_0)
    #print(t.shape)
    return R, t

def compute_transformation_point_to_plane(source_points, target_points, target_normals):
    # project point-to-point onto the direction of the normal, shot from the found point
    # trying to minimize point to plane error
    # gonna form the linear system Ax = b.
    A = []
    b = []

    for source_pt, target_pt, norm in zip(source_points, target_points, target_normals):
        # X x n
        crossProd = np.cross(source_pt, norm) # p_i x n_i T

        A.append(np.hstack((crossProd, norm))) # [p_i x n_i, n_i T] 

        b.append(-np.dot(norm, source_pt - target_pt)) # -n_i*(p_i-q_i)

    # have to solve Ax = b now. solve for x
    # thank god for numpy 
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # this gives the angle and translation in a 6 dimensional vector
    # [angle_x, angle_y, angle_z, move_x, move_y, move_z]
    w = x[:3]
    t = x[3:]

    # to make R we have to put it in the matrix manually
    # R = identity matrix plus the skew matrix
    R = np.array([[1, -w[2], w[1]],
                 [w[2], 1, -w[0]],
                 [-w[1], w[0], 1]])
    
    return R, t


def apply_transformation(source_points, R, t):
    # Apply the rotation R and translation t to the source points
    # assuming its a numpy array
    # print(type(source_points))
    # print(source_points)
    #print(source_points.shape)
    #print(R.shape)
    #print(t.shape)
    # use numpy matrix multiplication or dot product
    new_source_points = source_points @ R.T + t
    return new_source_points

def compute_mean_distance(source, target):
    # Compute the mean Euclidean distance between the source and the target
    # scipy spatial has a function to take care of this which is very nice
    # dist = distance.cdist(source, target, metric="euclidean")
    # mean_distance = np.mean(dist)

    # or just use CKDTree the same as the others
    tree = cKDTree(target)
    distances, indices = tree.query(source, k=1)
    mean_distance = np.mean(distances)
    print(mean_distance)
    return mean_distance

def calculate_mse(source_points, target_points):
    # Follow the equation in slides 
    # You may find cKDTree.query function helpful to calculate distance between point clouds with different number of points

    # if its the same number of points this should be enough
    if len(source_points) == len(target_points):
        #print("I'm in here!")
        mse = np.mean(np.sum((target_points-source_points)**2, axis=1))
    else:
        # if its not then we got work to do
        print("I'm in here!")
        targetTree = cKDTree(target_points)
        distances, indices = targetTree.query(source_points, k=1) 
        mse = np.mean(distances**2)

    return mse

def calculate_ptop_error(source_points, target_points, target_norms):
    diffs = source_points - target_points
    dot = np.sum(diffs * target_norms, axis=1)
    ptop_error = np.mean(dot ** 2)
    print(ptop_error)
    return ptop_error

def filter_by_z(points, z_min=None, z_percentile=None, min_points=50):
    """
    Returns (filtered_points, mask, z_thr). If the filter leaves too few points,
    it falls back to the original points and returns z_thr=None.
    """
    if points.size == 0:
        return points, np.ones((0,), dtype=bool), None

    z = points[:, 2]
    if z_percentile is not None:
        z_thr = float(np.percentile(z, z_percentile))
    elif z_min is not None:
        z_thr = float(z_min)
    else:
        # no filtering
        return points, np.ones((len(points),), dtype=bool), None

    mask = z > z_thr
    if np.count_nonzero(mask) < min_points:
        # too aggressive → fallback
        return points, np.ones((len(points),), dtype=bool), None
    return points[mask], mask, z_thr


def _make_pcd(points, color=None, voxel=None):
    pcd = o3d.geometry.PointCloud()
    if len(points) == 0:
        return pcd
    pcd.points = o3d.utility.Vector3dVector(points)
    if voxel is not None and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    if color is not None:
        pcd.paint_uniform_color(color)
    return pcd

def visualize_filter(points_full, mask, z_thr=None, title="Filter preview", voxel=0.0):
    """
    Show kept points vs discarded points.
    - kept:    blue-ish
    - removed: light gray
    """
    if points_full.size == 0:
        print(f"{title}: no points to visualize.")
        return

    if mask is None or mask.shape[0] != points_full.shape[0]:
        # no filter applied or invalid mask → show everything as 'kept'
        mask = np.ones(points_full.shape[0], dtype=bool)

    kept = points_full[mask]
    removed = points_full[~mask]

    pcd_kept = _make_pcd(kept, color=[0.0, 0.6, 1.0], voxel=voxel)
    pcd_removed = _make_pcd(removed, color=[0.8, 0.8, 0.8], voxel=voxel)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

    win = f"{title} (z>{z_thr:.3f})" if z_thr is not None else f"{title} (no threshold)"
    o3d.visualization.draw_geometries([pcd_removed, pcd_kept, frame], window_name=win)


'''def icp(source_points, target_points, max_iterations=100, tolerance=1e-6, R_init=None, t_init=None, strategy="closest_point"):
    # Apply initial guess if provided
    if R_init is not None and t_init is not None:
        source_points = apply_transformation(source_points, R_init, t_init)
        
    # ICP algorithm, return the rotation R and translation t
    for i in range(max_iterations):
        # The number of points may be different in the source and the target, so align them first
        print("Iteration ", i)
        if strategy == 'closest_point':
            indices = find_closest_points(source_points, target_points)
            matched_target_points = target_points[indices]
            pass
        elif strategy == 'normal_shooting':
            source_norms = estimate_normals(source_points)
            target_norms = estimate_normals(target_points)
            matched_target_points = normal_shooting(source_points, source_norms, target_points, target_norms)
            pass
        elif strategy == "point-to-plane":
            target_norms = estimate_normals(target_points)
            matched_target_points, matched_target_normals = point_to_plane(source_points, target_points, target_norms)
            pass
        else:
            raise ValueError("Invalid strategy. Choose 'closest_point', 'normal_shooting', or 'point_to_plane'")
        
        # Complete the rest of code using source_points and matched_target_points
        # Step 2: Compute the best transformation R and t
        if strategy == "point-to-plane":        
            R, t = compute_transformation_point_to_plane(source_points, matched_target_points, matched_target_normals)
        else:
            R, t = compute_transformation(source_points, matched_target_points)
        # Step 3: Apply the computed transformation (R, t) to the source points
        new_source_points = apply_transformation(source_points, R, t)
        # Step 4: Calculate the “error”

        if strategy == "point-to-plane":
            mean_distance = calculate_ptop_error(new_source_points, matched_target_points, matched_target_normals)
        else:
            mean_distance = compute_mean_distance(new_source_points, matched_target_points)
        # If the mean distance is less than the specified tolerance, the algorithm has converged
        if mean_distance < tolerance:
            print("ICP converged after", i + 1, "iterations.")
            # Return the final rotation matrix, translation vector, and aligned source points
            return R, t, new_source_points
        source_points = new_source_points
        
    print("ICP did not converge after", max_iterations, "iterations.") 
    
    aligned_source_points = new_source_points
    return R, t, aligned_source_points'''


def icp(source_points, target_points, max_iterations=100, tolerance=1e-6,
        R_init=None, t_init=None, strategy="closest_point"):
        
    # Keep an untouched copy of the original source
    src0 = source_points.copy()

    # Initialize cumulative transform
    R_total = np.eye(3)
    t_total = np.zeros(3)

    # Apply initial guess if provided AND include it in the cumulative transform
    if R_init is not None and t_init is not None:
        R_total = R_init.copy()
        t_total = t_init.copy()
        source_points = apply_transformation(source_points, R_total, t_total)

    # (Optional) precompute target normals once for point-to-plane
    target_norms_cache = None
    if strategy == "point-to-plane":
        target_norms_cache = estimate_normals(target_points)

    prev_err = np.inf
    time_elpase = 0
    for i in range(max_iterations):

        print("Iteration", i)
        start = time.time()
        # 1) Correspondences
        if strategy == 'closest_point':
            indices = find_closest_points(source_points, target_points)
            matched_target_points = target_points[indices]
            matched_target_normals = None

        elif strategy == 'normal_shooting':
            source_norms = estimate_normals(source_points)
            target_norms = estimate_normals(target_points)
            matched_target_points = normal_shooting(source_points, source_norms, target_points, target_norms)
            matched_target_normals = None

        elif strategy == "point-to-plane":
            # use cached normals (target is static)
            matched_target_points, matched_target_normals = point_to_plane(
                source_points, target_points, target_norms_cache
            )
        else:
            raise ValueError("Invalid strategy. Choose 'closest_point', 'normal_shooting', or 'point_to_plane'")

        # 2) Incremental transform from current source → matched targets
        if strategy == "point-to-plane":
            R_inc, t_inc = compute_transformation_point_to_plane(
                source_points, matched_target_points, matched_target_normals
            )
        else:
            R_inc, t_inc = compute_transformation(source_points, matched_target_points)

        # 3) Apply incremental transform to current source
        source_points = apply_transformation(source_points, R_inc, t_inc)

        # 4) Accumulate into global transform (VERY IMPORTANT)
        R_total = R_inc @ R_total
        t_total = R_inc @ t_total + t_inc

        # 5) Error for convergence
        if strategy == "point-to-plane":
            err = calculate_ptop_error(source_points, matched_target_points, matched_target_normals)
        else:
            err = compute_mean_distance(source_points, matched_target_points)

        if abs(prev_err - err) < tolerance:
            print(f"ICP converged after {i+1} iterations (Δerr < tol).")
            break
        prev_err = err
        end = time.time()
        time_elpase += (end-start)
        print("time:",time_elpase/(i+1))
        
    else:
        print("ICP did not reach Δerr < tol within max_iterations.")

    # Apply global transform to ORIGINAL source for the final aligned cloud
    aligned = apply_transformation(src0, R_total, t_total)
    return R_total, t_total, aligned


if __name__ == "__main__":
    
    strategy = "closest_point"
    #strategy = "normal_shooting"
    #strategy = "point-to-plane"

    source_file = 'v1.ply'
    target_file = 'v2.ply'
    output_file = f'v1v2_{strategy}.ply'

    #source_file = 'v3.ply'
    #target_file = f'v1v2_{strategy}.ply'
    #output_file = f'v1v2v3_{strategy}_yctry.ply'
    
    source_points_full = load_ply(source_file)
    target_points_full = load_ply(target_file)  
    
    # Choose a filter (pick one of these two knobs)
    Z_MIN = 0.0           # e.g., keep points with z > 0.0
    Z_PERCENTILE = None    # or e.g., 65 to keep top 35% in z
    VOXEL_PREVIEW = 0.0
    source_points_filt, src_mask, src_thr = filter_by_z(
        source_points_full, z_min=Z_MIN, z_percentile=Z_PERCENTILE
    )
    target_points_filt, tgt_mask, tgt_thr = filter_by_z(
        target_points_full,  z_min=Z_MIN, z_percentile=Z_PERCENTILE
    )
    print(f"ICP using {len(source_points_filt)} / {len(source_points_full)} source points "
      f"and {len(target_points_filt)} / {len(target_points_full)} target points after z-filtering.")

    visualize_filter(source_points_full, src_mask, z_thr=src_thr, title="Source filter preview", voxel=VOXEL_PREVIEW)
    visualize_filter(target_points_full, tgt_mask, z_thr=tgt_thr, title="Target filter preview", voxel=VOXEL_PREVIEW)



    # Initial guess (modifiable)
    #R_init = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    #t_init = np.array([0, 0, 0])

    max_translate = 0 
    max_rotation = 0

    r = transform.Rotation.from_euler('xyz', np.random.uniform(-max_rotation, max_rotation, 3), degrees=True)
    R_init = r.as_matrix()
    t_init = np.random.uniform(-max_translate, max_translate, 3)
    print("Initial Rotation Matrix:")
    print(R_init)
    print("Initial Translation Vector:")
    print(t_init)
    print("Starting ICP...")
    #R, t, aligned_source_points = icp(source_points, target_points, R_init=R_init, t_init=t_init, strategy=strategy)
    R, t, _ = icp(source_points_filt, target_points_filt, R_init=R_init, t_init=t_init, strategy=strategy)

    print("ICP completed.")
    print("Rotation Matrix:")
    print(R)
    print("Translation Vector:")
    print(t)
    
    aligned_source_points = apply_transformation(source_points_full, R, t)
    # have to use different error metric if point to plane - see slides
    # if strategy == "point-to-plane":
    #     # # need target norms
    #     # target_norms = estimate_normals(target_points_full)
    #     # diff = target_points_full - aligned_source_points
    #     # distances = np.sum(diff * target_norms, axis=1)
    #     # mse = np.mean(distances**2)
    #     # normals on the target (compute once)
    #     target_norms_full = estimate_normals(target_points_full)

    #     # establish correspondences from aligned source → target
    #     nn_idx = find_closest_points(aligned_source_points, target_points_full)
    #     matched_tgt = target_points_full[nn_idx]
    #     matched_norms = target_norms_full[nn_idx]

    #     diffs = aligned_source_points - matched_tgt
    #     # point-to-plane squared error
    #     mse = np.mean(np.sum(diffs * matched_norms, axis=1) ** 2)

    # else:
    #     mse = calculate_mse(aligned_source_points, target_points_full)

    mse = calculate_mse(aligned_source_points, target_points_full)

    print(f"Mean Squared Error (MSE): {mse}")
    
    # Combine aligned source and target points
    combined_points = np.vstack((aligned_source_points, target_points_full))
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(combined_points)

    # Save
    o3d.io.write_point_cloud(output_file, combined_pcd)
    print(f"Combined point cloud saved to '{output_file}'")
    
    # Visualization
    source_pcd_aligned = o3d.geometry.PointCloud()
    source_pcd_aligned.points = o3d.utility.Vector3dVector(aligned_source_points)
    
    target_pcd = o3d.io.read_point_cloud(target_file)
    #target_pcd = target_pcd.voxel_down_sample(voxel_size=2)
    o3d.visualization.draw_geometries([source_pcd_aligned.paint_uniform_color([0, 0, 1]), target_pcd.paint_uniform_color([1, 0, 0])],
                                      window_name='ICP Visualization')
