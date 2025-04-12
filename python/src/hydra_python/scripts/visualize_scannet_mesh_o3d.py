import open3d as o3d

def visualize_mesh(file_path):
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(file_path)
    
    # Check if the mesh is empty
    if not mesh.has_triangles():
        print("The mesh is empty or not loaded correctly.")
        return
    
    # Compute vertex normals for better visualization
    mesh.compute_vertex_normals()
    
    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh])

if __name__ == "__main__":
    scene_id = '0131_02'
    file_path = f'/mnt/hdd1/saumyas/data/semnav/scannet/scans/scene{scene_id}/scene{scene_id}_vh_clean.ply'
    visualize_mesh(file_path)