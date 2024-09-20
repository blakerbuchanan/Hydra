import json
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from itertools import chain
from hydra_python import is_in_plane_frontier

class SceneGraphSim:
    def __init__(self, sg_path, pipeline):
        self._sg_path = sg_path / "filtered_dsg.json"
        self.pipeline = pipeline
        self.filter_out_objects = ['wall, floor', 'ceiling', 'door_frame']
        self.update()
    
    def _load_scene_graph(self):
        with open(self._sg_path, "r") as f:
            self.scene_graph = json.load(f)
        self.netx_sg = json_graph.node_link_graph(self.scene_graph)
                
    @property
    def scene_graph_str(self):
        return json.dumps(nx.node_link_data(self.filtered_netx_graph))
    
    @property
    def visited_node_ids(self):
        return self._visited_node_ids
    
    @property
    def frontier_node_ids(self):
        return self._frontier_node_ids

    def _build_sg_from_hydra_graph(self):
        self.filtered_netx_graph = nx.DiGraph()
        self._visited_node_ids, self._frontier_node_ids = [], []

        ## Adding agent nodes
        agent_ids, agent_cat_ids = [], []
        for layer in self.pipeline.graph.dynamic_layers:
            for node in layer.nodes:
                if 'a' in node.id.category.lower():
                    attr={}
                    nodeid, node_type, node_name = self._get_node_properties(node)
                    agent_cat_ids.append(int(node.id.category_id))
                    agent_ids.append(nodeid)

                    attr['position'] = list(node.attributes.position)
                    attr['name'] = node_name
                    attr['layer'] = node.layer
                    attr['timestamp'] = float(node.timestamp/1e8)
                    self.filtered_netx_graph.add_nodes_from([(nodeid, attr)])
        
        self.curr_agent_id = agent_ids[np.argmax(agent_cat_ids)]
        self.curr_agent_pos = self.get_position_from_id(self.curr_agent_id)

        ## Adding other nodes
        for node in self.pipeline.graph.nodes:
            attr={}
            nodeid, node_type, node_name = self._get_node_properties(node)
            attr['position'] = list(node.attributes.position)
            attr['name'] = node_name
            attr['layer'] = node.layer
            if node.id.category.lower() in ['o', 'r', 'b']:
                attr['label'] = node.attributes.semantic_label
            
            if 'o' in node.id.category.lower():
                if node_name in self.filter_out_objects:
                    continue
            
            if 'p' in node.id.category.lower():
                self._visited_node_ids.append(nodeid)
                if is_in_plane_frontier(attr['position'][2], self.curr_agent_pos[2]):
                    self._visited_node_ids.append(nodeid)
            if 'f' in node.id.category.lower():
                if is_in_plane_frontier(attr['position'][2], self.curr_agent_pos[2]):
                    self._frontier_node_ids.append(nodeid)
            self.filtered_netx_graph.add_nodes_from([(nodeid, attr)])

        ## Adding edges
        for edge in chain(self.pipeline.graph.edges, self.pipeline.graph.dynamic_interlayer_edges):
            source_node = node = self.pipeline.graph.get_node(edge.source)
            sourceid, source_type, source_name = self._get_node_properties(source_node)

            target_node = node = self.pipeline.graph.get_node(edge.target)
            targetid, target_type, target_name = self._get_node_properties(target_node)
            if source_name in self.filter_out_objects or target_name in self.filter_out_objects:
                continue
            
            if 'object' in source_type and 'object' in target_type:
                continue
            if 'visited' in source_type and 'visited' in target_type:
                continue
            if 'frontier' in source_type and 'frontier' in target_type:
                continue

            self.filtered_netx_graph.add_edges_from([(
                sourceid, targetid,
                {'source_name': source_name,
                'target_name': target_name,
                'type': f'{source_type}-to-{target_type}'}
            )])
        
        self.current_room = [n for n in self.filtered_netx_graph.predecessors(self.curr_agent_id) if 'room' in n]

    def _get_node_properties(self, node):
        # print(f"layer: {node.layer}. Category: {node.id.category.lower()}{node.id.category_id}. Active Frontier: {node.attributes.active_frontier}")
        if 'p' in node.id.category.lower():
            nodeid = f'visited_{node.id.category_id}'
            node_type = 'visited'
            node_name = 'visited'
        if 'f' in node.id.category.lower(): 
            nodeid = f'frontier_{node.id.category_id}'
            node_type = 'frontier'
            node_name = 'frontier'
        if 'o' in node.id.category.lower():
            nodeid = f'object_{node.id.category_id}'
            node_type = 'object'
            node_name = node.attributes.name
        if 'r' in node.id.category.lower():
            nodeid = f'room_{node.id.category_id}'
            node_type = 'room'
            node_name = 'room'
        if 'b' in node.id.category.lower():
            nodeid = f'building_{node.id.category_id}'
            node_type = 'building'
            node_name = 'building'
        if 'a' in node.id.category.lower():
            nodeid = f'agent_{node.id.category_id}'
            node_type = 'agent'
            node_name = 'agent'
        return nodeid, node_type, node_name
    
    def test_sg(self):
        # ***********TEST NODES***********
        ## ***Process pipeline.graph***
        place_node_positions = []
        active_frontier_place_node_positions = []
        object_node_positions = []
        room_node_positions = []
        building_node_positions = []
        n_object_nodes, n_place_nodes, n_frontier_nodes, n_agent_nodes, n_room_nodes, n_building_nodes = 0, 0, 0, 0, 0, 0
        for node in self.pipeline.graph.nodes:
            if 'p' in node.id.category.lower():
                # print(f"layer: {node.layer}. Category: {node.id.category.lower()}{node.id.category_id}. Place: {node.attributes.active_frontier}")
                place_node_positions.append(node.attributes.position)
                n_place_nodes += 1
            if 'f' in node.id.category.lower():
                # print(f"layer: {node.layer}. Category: {node.id.category.lower()}{node.id.category_id}. Active Frontier: {node.attributes.active_frontier}")
                active_frontier_place_node_positions.append(node.attributes.position)
                n_frontier_nodes += 1
            if 'o' in node.id.category.lower():
                # print(f"layer: {node.layer}. Category: {node.id.category.lower()}{node.id.category_id}.")
                object_node_positions.append(node.attributes.position)
                n_object_nodes += 1
            if 'r' in node.id.category.lower():
                # print(f"layer: {node.layer}. Category: {node.id.category.lower()}{node.id.category_id}.")
                room_node_positions.append(node.attributes.position)
                n_room_nodes += 1
            if 'b' in node.id.category.lower():
                # print(f"layer: {node.layer}. Category: {node.id.category.lower()}{node.id.category_id}.")
                building_node_positions.append(node.attributes.position)
                n_building_nodes += 1
        place_node_positions = np.array(place_node_positions)
        active_frontier_place_node_positions = np.array(active_frontier_place_node_positions)
        object_node_positions = np.array(object_node_positions)
        room_node_positions = np.array(room_node_positions)
        building_node_positions = np.array(building_node_positions)
        
        agent_node_positions = []
        for layer in self.pipeline.graph.dynamic_layers:
            for node in layer.nodes:
                # print(f"layer: {node.layer}. Category: {node.id.category.lower()} {node.id.category_id}")
                if 'a' in node.id.category.lower():
                    n_agent_nodes += 1
                    agent_node_positions.append(node.attributes.position)
        agent_node_positions = np.array(agent_node_positions)
        total_nodes = n_place_nodes+n_frontier_nodes+n_object_nodes+n_agent_nodes+n_room_nodes+n_building_nodes
        print(f"Pipeline: Object nodes:{n_object_nodes}, Place nodes:{n_place_nodes}, Frontier nodes:{n_frontier_nodes}, Agent nodes:{n_agent_nodes}, Room nodes:{n_room_nodes}, Building nodes:{n_building_nodes}. Total: {total_nodes}")

        ## ***Process json graph***
        n_object_nodes, n_place_nodes, n_frontier_nodes, n_agent_nodes, n_room_nodes, n_building_nodes = 0, 0, 0, 0, 0, 0
        for n in self.netx_sg.nodes:
            node = self.netx_sg.nodes[n]
            if 'place' in node['attributes']['type'].lower() and not node['attributes']['active_frontier']:
                n_place_nodes += 1
            if 'place' in node['attributes']['type'].lower() and node['attributes']['active_frontier']:
                n_frontier_nodes += 1
            if 'object' in node['attributes']['type'].lower():
                n_object_nodes += 1
            if 'agent' in node['attributes']['type'].lower():
                n_agent_nodes += 1
            if 'room' in node['attributes']['type'].lower():
                n_room_nodes += 1
            if 'semantic' in node['attributes']['type'].lower():
                n_building_nodes += 1
        total_nodes = n_place_nodes+n_frontier_nodes+n_object_nodes+n_agent_nodes+n_room_nodes+n_building_nodes
        print(f"Json: Object nodes:{n_object_nodes}, Place nodes:{n_place_nodes}, Frontier nodes:{n_frontier_nodes}, Agent nodes:{n_agent_nodes}, Room nodes:{n_room_nodes}, Building nodes:{n_building_nodes}. Total: {total_nodes}")
        print(f"Total json nodes: {len(self.netx_sg.nodes)}")

        # ***********TEST EDGES***********
        for layer in self.pipeline.graph.layers:
            n_edges = 0
            for edge in layer.edges:
                print(f"Layer:{layer.id}. Layer edge: {edge}")
                n_edges+=1
            print(f"Edges in layer {layer.id}: {n_edges}")

        ## These are agent to agent edges (not needed)
        for layer in self.pipeline.graph.dynamic_layers:
            n_edges = 0
            for edge in layer.edges:
                print(f"Layer:{layer.id}. Dynamic layer edge: {edge}")
                n_edges+=1
            print(f"Edges in dynamic layer {layer.id}: {n_edges}")

        n_edges, n_edges_interL, n_edges_dyn_interL = 0, 0, 0
        for edge in self.pipeline.graph.edges:
            print("Graph edge", edge)
            n_edges+=1
        print(f"Edges in graph: {n_edges}")

        ## these are already included in graph.edges
        for edge in self.pipeline.graph.interlayer_edges:
            print("interlayer edges", edge)
            n_edges_interL+=1
        print(f"Edges in interlayers: {n_edges_interL}")

        for edge in self.pipeline.graph.dynamic_interlayer_edges:
            print("dynamic interlayer edges", edge)
            n_edges_dyn_interL+=1
        print(f"Edges in dynamic interlayers: {n_edges_dyn_interL}")

        print(f"Total pipeline edges: {n_edges+n_edges_dyn_interL}")
        print(f"self.pipeline.graph.num_edges(): {self.pipeline.graph.num_edges()}")
        print(f"Total json edges: {len(self.netx_sg.edges)}")
    
    def get_current_semantic_state_str(self):
        agent_pos = self.filtered_netx_graph.nodes[self.curr_agent_id]['position']
        agent_loc_str = f'The agent is currently at node {self.curr_agent_id} at position {agent_pos}'

        room_str = ''
        if len(self.current_room) > 0:
            room_str = f'in room {self.current_room[0]}'

        return f'{agent_loc_str} {room_str}'
    
    def update(self):
        self._load_scene_graph()
        self._build_sg_from_hydra_graph()

    def get_position_from_id(self, nodeid):
        return np.array(self.filtered_netx_graph.nodes[nodeid]['position'])
