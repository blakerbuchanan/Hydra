#pragma once

#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include <glog/logging.h>

#include <voxblox/core/layer.h>
#include <voxblox_ros/mesh_vis.h>
#include <voxblox_skeleton/skeleton.h>

#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <dynamic_reconfigure/server.h>
#include <kimera_scene_graph/kimera_scene_graphConfig.h>
#include <std_srvs/Empty.h>
#include <std_srvs/EmptyRequest.h>
#include <std_srvs/EmptyResponse.h>
#include <std_srvs/SetBool.h>

#include <kimera_semantics_ros/semantic_simulation_server.h>

#include <object_db/ObjectRegistrationAction.h>

#include "kimera_scene_graph/building_finder.h"
#include "kimera_scene_graph/common.h"
#include "kimera_scene_graph/object_finder.h"
#include "kimera_scene_graph/places_room_connectivity_finder.h"
#include "kimera_scene_graph/room_connectivity_finder.h"
#include "kimera_scene_graph/room_finder.h"
#include "kimera_scene_graph/scene_node.h"
#include "kimera_scene_graph/semantic_ros_publishers.h"
#include "kimera_scene_graph/wall_finder.h"

namespace kimera {

typedef ColorPointCloud SemanticPointCloud;
typedef std::unordered_map<SemanticLabel, SemanticPointCloud::Ptr>
    SemanticPointCloudMap;
typedef std::unordered_map<SemanticLabel, vxb::Mesh::Ptr> SemanticMeshMap;
typedef actionlib::SimpleActionClient<object_db::ObjectRegistrationAction>
    ObjectDBClient;

class SceneGraphSimulationServer : public SemanticSimulationServer {
 private:
  typedef kimera_scene_graph::kimera_scene_graphConfig RqtSceneGraphConfig;

 public:
  SceneGraphSimulationServer(const ros::NodeHandle& nh,
                             const ros::NodeHandle& nh_private);

  bool sceneGraphReconstructionServiceCall(
      std_srvs::SetBool::Request& request,
      std_srvs::SetBool::Response& response);

  void sceneGraphReconstruction(const bool& only_rooms);

  void getSemanticPointcloudsFromMesh(
      const vxb::MeshLayer::ConstPtr& mesh_layer,
      const vxb::ColorMode& color_mode,
      SemanticPointCloudMap* semantic_pointclouds);

  void getSemanticMeshesFromMesh(const vxb::MeshLayer::ConstPtr& mesh_layer,
                                 const vxb::ColorMode& color_mode,
                                 SemanticMeshMap* semantic_pointclouds);

  ObjectPointClouds objectDatabaseActionCall(
      const ObjectPointClouds& object_pcls,
      const std::string semantic_label);

 protected:
  void rqtReconfigureCallback(RqtSceneGraphConfig& config, uint32_t level);
  void reconstructMeshOutOfTsdf(vxb::MeshLayer::Ptr mesh_layer);
  void reconstructEsdfOutOfTsdf(const bool& save_to_file);

  void publishSemanticMesh(const SemanticLabel& semantic_label,
                           const vxb::Mesh& semantic_mesh);
  void publishExplodedWalls(const vxb::Mesh& segmented_walls,
                            const double& explosion_factor = 2.0);
  void extractThings(const SemanticLabel& semantic_label,
                     const SemanticPointCloud::Ptr& semantic_pcl);
  void extractThings();

  void countRooms() const;
  void publishSkeletonToObjectLinks(const ColorPointCloud::Ptr& graph_pcl);

  // TODO(Toni): perhaps put this inside the scene graph as a conversion utility
  // function
  bool fillSceneGraphWithPlaces(
      const vxb::SparseSkeletonGraph& sparse_skeleton);

 protected:
  ros::Publisher color_clustered_pcl_pub_;
  ros::Publisher walls_clustered_pcl_pub_;
  ros::Publisher room_centroids_pub_;
  ros::Publisher mesh_pub_;
  ros::Publisher polygon_mesh_pub_;
  ros::Publisher sparse_graph_pub_;

  // TODO(Toni): This guy should be in the scene graph visualization itself
  ros::Publisher edges_obj_skeleton_pub_;

  // Rebuild esdf and save to file
  bool build_esdf_batch_ = false;

  // To publish msgs to different topics according to semantic label.
  SemanticRosPublishers<SemanticLabel, ColorPointCloud> semantic_pcl_pubs_;
  SemanticRosPublishers<SemanticLabel, visualization_msgs::Marker>
      semantic_mesh_pubs_;
  SemanticRosPublishers<SemanticLabel, visualization_msgs::Marker>
      semantic_mesh_2_pubs_;

  // Dynamically change params for scene graph reconstruction
  dynamic_reconfigure::Server<RqtSceneGraphConfig> rqt_server_;
  dynamic_reconfigure::Server<RqtSceneGraphConfig>::CallbackType rqt_callback_;

  // Dynamically request a scene graph reconstruction
  ros::ServiceServer reconstruct_scene_graph_srv_;
  ros::ServiceServer load_map_srv_;

  // Finders
  std::unique_ptr<WallFinder<ColorPoint>> wall_finder_;
  std::unique_ptr<EnclosingWallFinder> enclosing_wall_finder_;
  std::unique_ptr<ObjectFinder<ColorPoint>> object_finder_;

  // Room finder
  std::unique_ptr<RoomFinder> room_finder_;
  std::unique_ptr<PlacesRoomConnectivityFinder> places_in_rooms_finder_;
  std::unique_ptr<RoomConnectivityFinder> room_connectivity_finder_;
  std::unique_ptr<BuildingFinder> building_finder_;

  // This should be inside the object finder... (but this finder is used by
  // the room finder as well).
  // Perhaps put it inside the scene graph itself!
  NodeId next_object_id_ = 0;

  // Action client for object db
  std::unique_ptr<ObjectDBClient> object_db_client_;

  // Labels of interesting things
  std::vector<int> stuff_labels_;
  std::vector<int> walls_labels_;
  std::vector<int> floor_labels_;

  // KimeraX
  SceneGraph scene_graph_;

  // TODO(Toni): remove
  float skeleton_z_level_;

  // Skeleton graph
  vxb::SparseSkeletonGraph sparse_skeleton_graph_;
};

}  // namespace kimera
