/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Research was sponsored by the United States Air Force Research Laboratory and
 * the United States Air Force Artificial Intelligence Accelerator and was
 * accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
 * and conclusions contained in this document are those of the authors and should
 * not be interpreted as representing the official policies, either expressed or
 * implied, of the United States Air Force or the U.S. Government. The U.S.
 * Government is authorized to reproduce and distribute reprints for Government
 * purposes notwithstanding any copyright notation herein.
 * -------------------------------------------------------------------------- */
#pragma once
#include <config_utilities/factory.h>

#include <Eigen/Geometry>
#include <memory>
#include <thread>

#include "hydra/common/input_queue.h"
#include "hydra/common/module.h"
#include "hydra/common/output_sink.h"
#include "hydra/places/robot_footprint_integrator.h"
#include "hydra/reconstruction/mesh_integrator_config.h"
#include "hydra/reconstruction/projective_integrator_config.h"
#include "hydra/reconstruction/reconstruction_input.h"
#include "hydra/reconstruction/reconstruction_output.h"
#include "hydra/reconstruction/sensor.h"
#include "hydra/reconstruction/volumetric_map.h"
#include "hydra/utils/log_utilities.h"
#include "hydra/utils/pose_graph_tracker.h"

namespace hydra {

class ProjectiveIntegrator;
class MeshIntegrator;

class ReconstructionModule : public Module {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<ReconstructionModule>;
  using ReconstructionInputQueue = InputQueue<ReconstructionInput::Ptr>;
  using OutputQueue = InputQueue<ReconstructionOutput::Ptr>;
  using OutputMsgStatus = std::pair<ReconstructionOutput::Ptr, bool>;
  using Sink = OutputSink<uint64_t,
                          const Eigen::Isometry3d&,
                          const voxblox::Layer<voxblox::TsdfVoxel>&,
                          const ReconstructionOutput&>;

  struct Config {
    bool show_stats = true;
    int stats_verbosity = 2;
    bool clear_distant_blocks = true;
    double dense_representation_radius_m = 5.0;
    size_t num_poses_per_update = 1;
    size_t max_input_queue_size = 0;
    float semantic_measurement_probability = 0.9;
    ProjectiveIntegratorConfig tsdf;
    MeshIntegratorConfig mesh;
    config::VirtualConfig<Sensor> sensor;
    PoseGraphTracker::Config pose_graphs;
    config::VirtualConfig<RobotFootprintIntegrator> robot_footprint;
    std::vector<Sink::Factory> sinks;
  } const config;

  ReconstructionModule(const Config& config, const OutputQueue::Ptr& output_queue);

  virtual ~ReconstructionModule();

  void start() override;

  void stop() override;

  void save(const LogSetup& log_setup) override;

  std::string printInfo() const override;

  void spin();

  bool spinOnce();

  // public for external use
  bool spinOnce(const ReconstructionInput& input);

  void addSink(const Sink::Ptr& sink);

  inline const VolumetricMap* getMap() const { return map_.get(); }

  inline ReconstructionInputQueue::Ptr queue() const { return queue_; }

 protected:
  bool update(const ReconstructionInput& msg, bool full_update);

  voxblox::BlockIndexList findBlocksToArchive(const Eigen::Vector3f& center) const;

  void fillOutput(const ReconstructionInput& input, ReconstructionOutput& output);

  OutputMsgStatus getNextOutputMessage();

 protected:
  const std::unique_ptr<Sensor> sensor_;

  std::atomic<bool> should_shutdown_{false};
  ReconstructionInputQueue::Ptr queue_;
  std::unique_ptr<std::thread> spin_thread_;
  size_t num_poses_received_;
  std::set<uint64_t> timestamp_cache_;

  pose_graph_tools_msgs::PoseGraph agent_node_measurements_;
  PoseGraphTracker::Ptr pose_graph_tracker_;
  ReconstructionOutput::Ptr pending_output_;
  OutputQueue::Ptr output_queue_;
  Sink::List sinks_;

  std::unique_ptr<VolumetricMap> map_;
  std::unique_ptr<ProjectiveIntegrator> tsdf_integrator_;
  std::unique_ptr<MeshIntegrator> mesh_integrator_;
  RobotFootprintIntegrator::Ptr footprint_integrator_;

  inline static const auto registration_ =
      config::RegistrationWithConfig<ReconstructionModule,
                                     ReconstructionModule,
                                     Config,
                                     OutputQueue::Ptr>("ReconstructionModule");
};

void declare_config(ReconstructionModule::Config& config);

}  // namespace hydra
