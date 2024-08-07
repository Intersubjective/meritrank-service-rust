use std::{
  sync::atomic::Ordering,
  collections::HashMap,
  env::var,
  string::ToString,
};
use petgraph::{visit::EdgeRef, graph::{DiGraph, NodeIndex}};
use simple_pagerank::Pagerank;
use meritrank::{MeritRank, Graph, NodeId, Neighbors, MeritRankError, constants::EPSILON};

use crate::log_error;
use crate::log_warning;
use crate::log_info;
use crate::log_verbose;
use crate::log_trace;
use crate::log::*;
use crate::astar::*;

pub use meritrank::Weight;

//  ================================================================
//
//    Constants
//
//  ================================================================

pub const VERSION : &str = match option_env!("CARGO_PKG_VERSION") {
  Some(x) => x,
  None    => "dev",
};

lazy_static::lazy_static! {
  pub static ref ZERO_NODE : String =
    var("MERITRANK_ZERO_NODE")
      .unwrap_or("U000000000000".to_string());

  pub static ref NUM_WALK : usize =
    var("MERITRANK_NUM_WALK")
      .ok()
      .and_then(|s| s.parse::<usize>().ok())
      .unwrap_or(10000);

  pub static ref TOP_NODES_LIMIT : usize =
    var("MERITRANK_TOP_NODES_LIMIT")
      .ok()
      .and_then(|s| s.parse::<usize>().ok())
      .unwrap_or(100);
}

//  ================================================================
//
//    Basic declarations
//
//  ================================================================

#[derive(PartialEq, Eq, Clone, Copy, Default)]
pub enum NodeKind {
  #[default]
  Unknown,
  User,
  Beacon,
  Comment,
}

#[derive(PartialEq, Eq, Clone, Default)]
pub struct NodeInfo {
  pub kind : NodeKind,
  pub name : String,
}

//  Augmented multi-context graph
//
#[derive(Clone)]
pub struct AugMultiGraph {
  pub node_count : usize,
  pub node_infos : Vec<NodeInfo>,
  pub dummy_info : NodeInfo,
  pub node_ids   : HashMap<String, NodeId>,
  pub contexts   : HashMap<String, MeritRank>,
}

//  ================================================================
//
//    Utils
//
//  ================================================================

fn kind_from_name(name : &str) -> NodeKind {
  log_trace!("kind_from_name: `{}`", name);

  match name.chars().nth(0) {
    Some('U') => NodeKind::User,
    Some('B') => NodeKind::Beacon,
    Some('C') => NodeKind::Comment,
    _         => NodeKind::Unknown,
  }
}

impl Default for AugMultiGraph {
  fn default() -> AugMultiGraph {
    AugMultiGraph::new()
  }
}

impl AugMultiGraph {
  pub fn new() -> AugMultiGraph {
    log_trace!("AugMultiGraph::new");

    AugMultiGraph {
      node_count : 0,
      node_infos : Vec::new(),
      dummy_info : NodeInfo {
        kind : NodeKind::Unknown,
        name : "".to_string(),
      },
      node_ids   : HashMap::new(),
      contexts   : HashMap::new(),
    }
  }

  pub fn copy_from(&mut self, other : &AugMultiGraph) {
    self.node_count = other.node_count;
    self.node_infos = other.node_infos.clone();
    self.node_ids   = other.node_ids.clone();
    self.contexts   = other.contexts.clone();
  }

  pub fn reset(&mut self) {
    log_trace!("reset");

    self.node_count   = 0;
    self.node_infos   = Vec::new();
    self.node_ids     = HashMap::new();
    self.contexts     = HashMap::new();
  }

  pub fn node_exists(&self, node_name : &str) -> bool {
    log_trace!("node_exists");
    self.node_ids.get(node_name).is_some()
  }

  pub fn node_info_from_id(&mut self, node_id : NodeId) -> &NodeInfo {
    log_trace!("node_info_from_id: {}", node_id);

    match self.node_infos.get(node_id) {
      Some(x) => x,
      _       => {
        log_error!("(node_info_from_id) Node does not exist: `{}`", node_id);
        self.dummy_info = NodeInfo {
          kind : NodeKind::Unknown,
          name : "".to_string(),
        };
        &self.dummy_info
      },
    }
  }

  //  Get mutable graph from a context
  //
  pub fn graph_from(&mut self, context : &str) -> Result<&mut MeritRank, ()> {
    log_trace!("graph_from: `{}`", context);

    if !self.contexts.contains_key(context) {
      if context.is_empty() {
        log_verbose!("Add context: NULL");
      } else {
        log_verbose!("Add context: `{}`", context);
      }

      let mut meritrank = match MeritRank::new(Graph::new()) {
        Ok(x)  => x,
        Err(e) => {
          log_error!("(graph_from) {}", e);
          return Err(());
        },
      };

      for _ in 0..self.node_count {
        meritrank.get_new_nodeid();
      }

      self.contexts.insert(context.to_string(), meritrank);
    }

    match self.contexts.get_mut(context) {
      Some(graph) => Ok(graph),
      None        => {
        log_error!("(graph_from) Unable to add context `{}`", context);
        Err(())
      },
    }
  }

  pub fn add_edge(&mut self, context : &str, src : NodeId, dst : NodeId, amount : Weight) {
    log_trace!("add_edge: `{}` {} {} {}", context, src, dst, amount);

    match self.graph_from(context) {
      Ok(x) => x.add_edge(src, dst, amount),
      _     => {},
    }
  }

  pub fn edge_weight(&mut self, context : &str, src : NodeId, dst : NodeId) -> Weight {
    log_trace!("edge_weight: `{}` {} {}", context, src, dst);

    match self.graph_from(context) {
      Ok(x) => *x.graph.edge_weight(src, dst).unwrap_or(None).unwrap_or(&0.0),
      _     => 0.0,
    }
  }

  pub fn all_neighbors(&mut self, context : &str, node : NodeId) -> Vec<(NodeId, Weight)> {
    match self.graph_from(context) {
      Err(_) => vec![],
      Ok(x)  => {
        let mut v = vec![];

        match x.graph.get_node_data(node) {
          None => {},
          Some(data) => {
            v.reserve_exact(
              data.neighbors(Neighbors::Positive).len() +
              data.neighbors(Neighbors::Negative).len()
            );

            for x in data.neighbors(Neighbors::Positive) {
              v.push((*x.0, *x.1));
            }

            for x in data.neighbors(Neighbors::Negative) {
              v.push((*x.0, *x.1));
            }
          }
        }

        v
      },
    }
  }

  fn get_ranks_or_recalculate(
    &mut self,
    context   : &str,
    node_id   : NodeId
  ) -> Vec<(NodeId, Weight)> {
    log_trace!("get_ranks_or_recalculate");

    let graph = match self.graph_from(context) {
      Ok(x)  => x,
      Err(_) => return vec![],
    };

    match graph.get_ranks(node_id, None) {
      Ok(ranks) => ranks,
      Err(MeritRankError::NodeDoesNotExist) => {
        log_warning!("Node does not exist: {}", node_id);
        vec![]
      },
      _ => {
        log_warning!("Recalculating node: {}", node_id);
        match graph.calculate(node_id, *NUM_WALK) {
          Err(e) => {
            log_error!("(get_ranks_or_recalculate) {}", e);
            return vec![];
          },
          _ => {},
        };
        match graph.get_ranks(node_id, None) {
          Ok(ranks) => ranks,
          Err(e) => {
            log_error!("(get_ranks_or_recalculate) {}", e);
            vec![]
          }
        }
      },
    }
  }

  fn get_score_or_recalculate(
    &mut self,
    context   : &str,
    src_id    : NodeId,
    dst_id    : NodeId
  ) -> Weight {
    log_trace!("get_score_or_recalculate");

    let graph = match self.graph_from(context) {
      Ok(x)  => x,
      Err(_) => return 0.0,
    };

    match graph.get_node_score(src_id, dst_id) {
      Ok(score) => score,
      Err(MeritRankError::NodeDoesNotExist) => {
        log_warning!("Node does not exist: {}, {}", src_id, dst_id);
        0.0
      },
      _ => {
        log_warning!("Recalculating node {}", src_id);
        match graph.calculate(src_id, *NUM_WALK) {
          Err(e) => {
            log_error!("(get_score_or_recalculate) {}", e);
            return 0.0;
          },
          _ => {},
        };
        match graph.get_node_score(src_id, dst_id) {
          Ok(score) => score,
          Err(e) => {
            log_error!("(get_score_or_recalculate) {}", e);
            0.0
          }
        }
      },
    }
  }

  pub fn find_or_add_node_by_name(
    &mut self,
    node_name : &str
  ) -> NodeId {
    log_trace!("find_or_add_node_by_name: `{}`", node_name);

    let node_id;

    if let Some(&id) = self.node_ids.get(node_name) {
      node_id = id;
    } else {
      node_id = self.node_count;

      self.node_count += 1;
      self.node_infos.resize(self.node_count, NodeInfo::default());
      self.node_infos[node_id] = NodeInfo {
        kind : kind_from_name(&node_name),
        name : node_name.to_string(),
      };
      self.node_ids.insert(node_name.to_string(), node_id);
    }

    for (context, meritrank) in &mut self.contexts {
      if meritrank.graph.contains_node(node_id) {
        continue;
      }

      if !context.is_empty() {
        log_verbose!("Add node in NULL: {}", node_id);
      } else {
        log_verbose!("Add node in `{}`: {}", context, node_id);
      }

      //  HACK!!!
      while meritrank.get_new_nodeid() < node_id {}
    }

    node_id
  }

  pub fn set_edge(
    &mut self,
    context : &str,
    src     : NodeId,
    dst     : NodeId,
    amount  : f64
  ) {
    log_trace!("set_edge: `{}` `{}` `{}` {}", context, src, dst, amount);

    if context.is_empty() {
      log_verbose!("Add edge in NULL: {} -> {} for {}", src, dst, amount);
      self.add_edge(context, src, dst, amount);
    } else {
      let null_weight = self.edge_weight("",      src, dst);
      let old_weight  = self.edge_weight(context, src, dst);
      let delta       = null_weight + amount - old_weight;

      log_verbose!("Add edge in NULL: {} -> {} for {}", src, dst, delta);
      self.add_edge("", src, dst, delta);

      log_verbose!("Add edge in `{}`: {} -> {} for {}", context, src, dst, amount);
      self.add_edge(context, src, dst, amount);
    }
  }

  pub fn connected_nodes(
    &mut self,
    context   : &str,
    ego       : NodeId
  ) -> Vec<(NodeId, NodeId)> {
    log_trace!("connected_nodes: `{}` {}", context, ego);

    self.all_neighbors(context, ego)
      .into_iter()
      .map(|(dst_id, _)| (ego, dst_id))
        .collect()
  }

  pub fn connected_node_names(
    &mut self,
    context   : &str,
    ego       : &str
  ) -> Vec<(String, String)> {
    log_trace!("connected_node_names: `{}` `{}`", context, ego);

    if !self.node_exists(ego) {
      log_error!("(connected_node_names) Node does not exist: `{}`", ego);
      return vec![];
    }

    let src_id   = self.find_or_add_node_by_name(ego);
    let edge_ids = self.connected_nodes(context, src_id);

    let mut v = vec![];
    v.reserve_exact(edge_ids.len());

    for x in edge_ids {
      v.push((
        self.node_info_from_id(x.0).name.clone(),
        self.node_info_from_id(x.1).name.clone()
      ));
    }

    v
  }

  pub fn recalculate_all(&mut self, num_walk : usize) {
    log_trace!("recalculate_all: {}", num_walk);

    let infos = self.node_infos.clone();

    let graph = match self.graph_from("") {
      Ok(x) => x,
      _     => return,
    };

    for id in 0..infos.len() {
      if (id % 100) == 90 {
        log_trace!("{}%", (id * 100) / infos.len());
      }
      if infos[id].kind == NodeKind::User {
        match graph.calculate(id, num_walk) {
          Ok(_)  => {},
          Err(e) => log_error!("(recalculate_all) {}", e),
        };
      }
    }
  }
}

//  ================================================
//
//    Commands
//
//  ================================================

pub fn read_version() -> &'static str {
  log_info!("CMD read_version");
  VERSION
}

pub fn write_log_level(log_level : u32) {
  log_info!("CMD write_log_level: {}", log_level);

  ERROR  .store(log_level > 0, Ordering::Relaxed);
  WARNING.store(log_level > 1, Ordering::Relaxed);
  INFO   .store(log_level > 2, Ordering::Relaxed);
  VERBOSE.store(log_level > 3, Ordering::Relaxed);
  TRACE  .store(log_level > 4, Ordering::Relaxed);
}

impl AugMultiGraph {
  pub fn read_node_score(
    &mut self,
    context : &str,
    ego     : &str,
    target  : &str
  ) -> Vec<(String, String, f64)> {
    log_info!("CMD read_node_score: `{}` `{}` `{}`", context, ego, target);

    if !self.node_exists(ego) {
      log_error!("(read_node_score) Node does not exist: `{}`", ego);
      return [(ego.to_string(), target.to_string(), 0.0)].to_vec();
    }

    if !self.node_exists(target) {
      log_error!("(read_node_score) Node does not exist: `{}`", target);
      return [(ego.to_string(), target.to_string(), 0.0)].to_vec();
    }

    let ego_id    = self.find_or_add_node_by_name(ego);
    let target_id = self.find_or_add_node_by_name(target);
    let w         = self.get_score_or_recalculate(context, ego_id, target_id);

    [(ego.to_string(), target.to_string(), w)].to_vec()
  }

  pub fn read_scores(
    &mut self,
    context       : &str,
    ego           : &str,
    kind_str      : &str,
    hide_personal : bool,
    score_lt      : f64,
    score_lte     : bool,
    score_gt      : f64,
    score_gte     : bool,
    index         : u32,
    count         : u32
  ) -> Vec<(String, String, Weight)> {
    log_info!("CMD read_scores: `{}` `{}` `{}` {} {} {} {} {} {} {}",
              context, ego, kind_str, hide_personal,
              score_lt, score_lte, score_gt, score_gte,
              index, count);

    let kind = match kind_str {
      ""  => NodeKind::Unknown,
      "U" => NodeKind::User,
      "B" => NodeKind::Beacon,
      "C" => NodeKind::Comment,
       _  => {
         log_error!("(read_scores) Invalid node kind string: `{}`", kind_str);
         return vec![];
      },
    };

    if !self.node_exists(ego) {
      log_error!("(read_scores) Node does not exist: `{}`", ego);
      return vec![];
    }

    let node_id = self.find_or_add_node_by_name(ego);

    let ranks = self.get_ranks_or_recalculate(context, node_id);

    let mut im : Vec<(NodeId, Weight)> =
      ranks
        .into_iter()
        .map(|(n, w)| (
          n,
          self.node_info_from_id(n).kind,
          w,
        ))
        .filter(|(_, target_kind, _)| kind == NodeKind::Unknown || kind == *target_kind)
        .filter(|(_, _, score)| score_gt < *score   || (score_gte && score_gt <= *score))
        .filter(|(_, _, score)| *score   < score_lt || (score_lte && score_lt >= *score))
        .collect::<Vec<(NodeId, NodeKind, Weight)>>()
        .into_iter()
        .filter(|(target_id, target_kind, _)| {
          if !hide_personal || (*target_kind != NodeKind::Comment && *target_kind != NodeKind::Beacon) {
            return true;
          }
          match self.graph_from(context) {
            Ok(graph) => match graph.graph.edge_weight(*target_id, node_id) {
              Ok(Some(_)) => false,
              _           => true,
            },
            Err(_) => false,
          }
        })
        .map(|(target_id, _, weight)| (target_id, weight))
        .collect();

    im.sort_by(|(_, a), (_, b)| a.total_cmp(b));

    let index = index as usize;
    let count = count as usize;

    let mut page : Vec<(String, String, Weight)> = vec![];
    page.reserve_exact(if count < im.len() { count } else { im.len() });

    for i in index..count {
      if i >= im.len() {
        break;
      }
      page.push((ego.to_string(), self.node_info_from_id(im[i].0).name.clone(), im[i].1));
    }

    page
  }

  pub fn write_put_edge(
    &mut self,
    context : &str,
    src     : &str,
    dst     : &str,
    amount  : f64
  ) {
    log_info!("CMD write_put_edge: `{}` `{}` `{}` {}", context, src, dst, amount);

    let src_id = self.find_or_add_node_by_name(src);
    let dst_id = self.find_or_add_node_by_name(dst);

    self.set_edge(context, src_id, dst_id, amount);
  }

  pub fn write_delete_edge(
    &mut self,
    context : &str,
    src     : &str,
    dst     : &str,
  ) {
    log_info!("CMD write_delete_edge: `{}` `{}` `{}`", context, src, dst);

    if !self.node_exists(src) || !self.node_exists(dst) {
      return;
    }

    let src_id = self.find_or_add_node_by_name(src);
    let dst_id = self.find_or_add_node_by_name(dst);

    self.set_edge(context, src_id, dst_id, 0.0);
  }

  pub fn write_delete_node(
    &mut self,
    context : &str,
    node    : &str,
  ) {
    log_info!("CMD write_delete_node: `{}` `{}`", context, node);

    if !self.node_exists(node) {
      return;
    }

    let id = self.find_or_add_node_by_name(node);

    for (n, _) in self.all_neighbors(context, id) {
      self.set_edge(context, id, n, 0.0);
    }
  }

  pub fn read_graph(
    &mut self,
    context       : &str,
    ego           : &str,
    focus         : &str,
    positive_only : bool,
    index         : u32,
    count         : u32
  ) -> Vec<(String, String, Weight)> {
    log_info!("CMD read_graph: `{}` `{}` `{}` {} {} {}",
              context, ego, focus, positive_only, index, count);

    if !self.node_exists(ego) {
      log_error!("(read_graph) Node does not exist: `{}`", ego);
      return vec![];
    }

    if !self.node_exists(focus) {
      log_error!("(read_graph) Node does not exist: `{}`", focus);
      return vec![];
    }

    let ego_id   = self.find_or_add_node_by_name(ego);
    let focus_id = self.find_or_add_node_by_name(focus);

    let mut indices  = HashMap::<NodeId, NodeIndex>::new();
    let mut ids      = HashMap::<NodeIndex, NodeId>::new();
    let mut im_graph = DiGraph::<NodeId, Weight>::new();

    {
      let index = im_graph.add_node(focus_id);
      indices.insert(focus_id, index);
      ids.insert(index, focus_id);
    }

    log_trace!("enumerate focus neighbors");

    let focus_neighbors = self.all_neighbors(context, focus_id);

    for (dst_id, focus_dst_weight) in focus_neighbors {
      let dst_kind = self.node_info_from_id(dst_id).kind;

      if dst_kind == NodeKind::User {
        if positive_only && self.edge_weight(context, ego_id, dst_id) <= 0.0 {
          continue;
        }

        if !indices.contains_key(&dst_id) {
          let index = im_graph.add_node(focus_id);
          indices.insert(dst_id, index);
          ids.insert(index, dst_id);
        }

        if let (Some(focus_idx), Some(dst_idx)) = (indices.get(&focus_id), indices.get(&dst_id)) {
          im_graph.add_edge(*focus_idx, *dst_idx, focus_dst_weight);
        } else {
          log_error!("(read_graph) Got invalid node id");
        }
      } else if dst_kind == NodeKind::Comment || dst_kind == NodeKind::Beacon {
        let dst_neighbors = self.all_neighbors(context, dst_id);

        for (ngh_id, dst_ngh_weight) in dst_neighbors {
          if (positive_only && dst_ngh_weight <= 0.0) || ngh_id == focus_id || self.node_info_from_id(ngh_id).kind != NodeKind::User {
            continue;
          }

          let focus_ngh_weight = focus_dst_weight * dst_ngh_weight * if focus_dst_weight < 0.0 && dst_ngh_weight < 0.0 { -1.0 } else { 1.0 };

          if !indices.contains_key(&ngh_id) {
            let index = im_graph.add_node(ngh_id);
            indices.insert(ngh_id, index);
            ids.insert(index, ngh_id);
          }

          if let (Some(focus_idx), Some(ngh_idx)) = (indices.get(&focus_id), indices.get(&ngh_id)) {
            im_graph.add_edge(*focus_idx, *ngh_idx, focus_ngh_weight);
          } else {
            log_error!("(read_graph) Got invalid node id");
          }
        }
      }
    }

    if ego_id == focus_id {
      log_trace!("ego is same as focus");
    } else {
      log_trace!("search shortest path");

      let graph_cloned = match self.graph_from(context) {
        Ok(x)  => x.graph.clone(),
        Err(_) => return vec![],
      };

      //  ================================
      //
      //    A* search
      //

      let mut open   : Vec<Node<NodeId, Weight>> = vec![];
      let mut closed : Vec<Node<NodeId, Weight>> = vec![];

      open  .resize(1024, Node::default());
      closed.resize(1024, Node::default());

      let mut astar_state = init(&mut open, ego_id, focus_id, 0.0);

      let mut steps    = 0;
      let mut neighbor = None;
      let mut status   = Status::PROGRESS;

      //  Do 10000 iterations max

      for _ in 0..10000 {
        steps += 1;
        
        status = iteration(&mut open, &mut closed, &mut astar_state, neighbor.clone());

        match status.clone() {
          Status::NEIGHBOR(request) => {
            match graph_cloned.get_node_data(request.node) {
              None       => neighbor = None,
              Some(data) => {
                let kv : Vec<_> = data.neighbors(Neighbors::Positive).iter().skip(request.index).take(1).collect();

                if kv.is_empty() {
                  neighbor = None;
                } else {
                  let n = kv[0].0;
                  let w = kv[0].1;

                  neighbor = Some(Link::<NodeId, Weight> {
                    neighbor       : *n,
                    exact_distance : if w.abs() < EPSILON { 1_000_000.0 } else { 1.0 / w },
                    estimate       : 0.0,
                  });
                }
              },
            }
          },
          Status::OUT_OF_MEMORY => {
            open  .resize(open  .len() * 2, Node::default());
            closed.resize(closed.len() * 2, Node::default());
          },
          Status::SUCCESS  => break,
          Status::FAIL     => break,
          Status::PROGRESS => {},
        };
      }

      log_trace!("did {} A* iterations", steps);

      if status == Status::SUCCESS {
        log_trace!("path found");
      } else if status == Status::FAIL {
        log_error!("(read_graph) Path does not exist from {} to {}", ego_id, focus_id);
      } else {
        log_error!("(read_graph) Unable to find a path from {} to {}", ego_id, focus_id);
      }

      let mut ego_to_focus : Vec<NodeId> = vec![];
      ego_to_focus.resize(astar_state.num_closed, 0);
      let n = path(&closed, &astar_state, &mut ego_to_focus);
      ego_to_focus.resize(n, 0);

      for node in ego_to_focus.iter() {
        log_trace!("path: {}", self.node_info_from_id(*node).name);
      }

      //  ================================

      let mut edges = Vec::<(NodeId, NodeId, Weight)>::new();
      edges.reserve_exact(ego_to_focus.len() - 1);

      log_trace!("process shortest path");

      for k in 0..ego_to_focus.len()-1 {
        let a = ego_to_focus[k];
        let b = ego_to_focus[k + 1];

        let a_kind = self.node_info_from_id(a).kind;
        let b_kind = self.node_info_from_id(b).kind;

        let a_b_weight = self.edge_weight(context, a, b);

        if k + 2 == ego_to_focus.len() {
          if a_kind == NodeKind::User {
            edges.push((a, b, a_b_weight));
          } else {
            log_trace!("ignore node {}", self.node_info_from_id(a).name);
          }
        } else if b_kind != NodeKind::User {
          log_trace!("ignore node {}", self.node_info_from_id(b).name);
          let c = ego_to_focus[k + 2];
          let b_c_weight = self.edge_weight(context, b, c);
          let a_c_weight = a_b_weight * b_c_weight * if a_b_weight < 0.0 && b_c_weight < 0.0 { -1.0 } else { 1.0 };
          edges.push((a, c, a_c_weight));
        } else if a_kind == NodeKind::User {
          edges.push((a, b, a_b_weight));
        } else {
          log_trace!("ignore node {}", self.node_info_from_id(a).name);
        }
      }

      log_trace!("add path to the graph");

      for (src, dst, weight) in edges {
        if !indices.contains_key(&src) {
          let index = im_graph.add_node(src);
          indices.insert(src, index);
          ids.insert(index, src);
        }

        if !indices.contains_key(&dst) {
          let index = im_graph.add_node(dst);
          indices.insert(dst, index);
          ids.insert(index, dst);
        }

        if let (Some(src_idx), Some(dst_idx)) = (indices.get(&src), indices.get(&dst)) {
          im_graph.add_edge(*src_idx, *dst_idx, weight);
        } else {
          log_error!("(read_graph) Got invalid node id");
        }
      }
    }

    log_trace!("remove self references");

    for (_, src_index) in indices.iter() {
      let neighbors : Vec<_> =
        im_graph.edges(*src_index)
          .map(|edge| (edge.target(), edge.id()))
          .collect();

      for (dst_index, edge_id) in neighbors {
        if *src_index == dst_index {
          im_graph.remove_edge(edge_id);
        }
      }
    }

    let mut edge_ids = Vec::<(NodeId, NodeId, Weight)>::new();
    edge_ids.reserve_exact(indices.len() * 2); // ad hok

    log_trace!("build final array");

    for (_, src_index) in indices {
      for edge in im_graph.edges(src_index) {
        if let (Some(src_id), Some(dst_id)) = (ids.get(&src_index), ids.get(&edge.target()))  {
          let w = *edge.weight();
          if w > -EPSILON && w < EPSILON {
            log_error!(
              "(read_graph) Got zero edge weight: {} -> {}",
              self.node_info_from_id(*src_id).name.clone(),
              self.node_info_from_id(*dst_id).name.clone()
            );
          } else {
            edge_ids.push((*src_id, *dst_id, w));
          }
        } else {
          log_error!("(read_graph) Got invalid node index");
        }
      }
    }

    edge_ids
      .into_iter()
      .map(|(src_id, dst_id, weight)| {(
        self.node_info_from_id(src_id).name.clone(),
        self.node_info_from_id(dst_id).name.clone(),
        weight
      )})
      .skip(index as usize)
      .take(count as usize)
      .collect()
  }

  pub fn read_connected(
    &mut self,
    context   : &str,
    ego       : &str
  ) -> Vec<(String, String)> {
    log_info!("CMD read_connected: `{}` `{}`", context, ego);
    self.connected_node_names(context, ego)
  }

  pub fn read_node_list(&self) -> Vec<(String,)> {
    log_info!("CMD read_node_list");

    self.node_infos
      .iter()
      .map(|info| (info.name.clone(),))
      .collect()
  }

  pub fn read_edges(&mut self, context : &str) -> Vec<(String, String, Weight)> {
    log_info!("CMD read_edges: `{}`", context);

    let infos = self.node_infos.clone();

    let mut v : Vec<(String, String, Weight)> = vec![];
    v.reserve(infos.len() * 2); // ad hok

    for src_id in 0..infos.len() {
      let src_name = infos[src_id].name.as_str();

      for (dst_id, weight) in self.all_neighbors(context, src_id) {
        match infos.get(dst_id) {
          Some(x) => v.push((src_name.to_string(), x.name.clone(), weight)),
          None    => log_error!("(read_edges) Node does not exist: {}", dst_id),
        }
      }
    }

    v
  }

  pub fn read_mutual_scores(
    &mut self,
    context   : &str,
    ego       : &str
  ) -> Vec<(String, Weight, Weight)> {
    log_info!("CMD read_mutual_scores: `{}` `{}`", context, ego);

    if !self.node_exists(ego) {
      log_error!("(read_mutual_scores) Node does not exist: `{}`", ego);
      return vec![];
    }

    let ego_id = self.find_or_add_node_by_name(ego);
    let ranks  = self.get_ranks_or_recalculate(context, ego_id);
    let mut v  = Vec::<(String, Weight, Weight)>::new();

    v.reserve_exact(ranks.len());

    for (node, score) in ranks {
      let info = self.node_info_from_id(node).clone();
      if score > 0.0 && info.kind == NodeKind::User
      {
        v.push((
          info.name,
          score,
          self.get_score_or_recalculate(context, node, ego_id)
        ));
      }
    }

    v
  }

  pub fn write_reset(&mut self) {
    log_info!("CMD write_reset");
    self.reset();
  }
}

//  ================================================
//
//    Zero node recalculation
//
//  ================================================

impl AugMultiGraph {
  fn reduced_graph(&mut self) -> Vec<(NodeId, NodeId, Weight)> {
    log_trace!("reduced_graph");

    let zero = self.find_or_add_node_by_name(ZERO_NODE.as_str());

    let users : Vec<NodeId> =
      self.node_infos
        .iter()
        .enumerate()
        .filter(|(id, info)|
          *id != zero && info.kind == NodeKind::User
        )
        .map(|(id, _)| id)
        .collect();

    if users.is_empty() {
      return vec![];
    }

    for id in users.iter() {
      match self.graph_from("") {
        Ok(x)  => match x.calculate(*id, *NUM_WALK) {
          Ok(_)  => {},
          Err(e) => log_error!("(reduced_graph) {}", e),
        },
        Err(_) => {},
      }
    }

    let edges : Vec<(NodeId, NodeId, Weight)> =
      users.into_iter()
        .map(|id| -> Vec<(NodeId, NodeId, Weight)> {
          self.get_ranks_or_recalculate("", id)
            .into_iter()
            .map(|(node_id, score)| (id, node_id, score))
            .filter(|(ego_id, node_id, score)| {
              let kind = self.node_info_from_id(*node_id).kind;

              (kind == NodeKind::User || kind == NodeKind::Beacon) &&
                *score > 0.0 &&
                ego_id != node_id
            })
            .collect()
        })
        .flatten()
        .collect();

    let result : Vec<(NodeId, NodeId, f64)> =
      edges
        .into_iter()
        .map(|(ego_id, dst_id, weight)| {
          let ego_kind = self.node_info_from_id(ego_id).kind;
          let dst_kind = self.node_info_from_id(dst_id).kind;
          (ego_id, ego_kind, dst_id, dst_kind, weight)
        })
        .filter(|(ego_id, ego_kind, dst_id, dst_kind, _)| {
          if *ego_id == zero || *dst_id == zero {
            false
          } else {
            ego_id != dst_id &&
            *ego_kind == NodeKind::User &&
            (*dst_kind == NodeKind::User || *dst_kind == NodeKind::Beacon)
          }
        })
        .map(|(ego_id, _, dst_id, _, weight)| {
          (ego_id, dst_id, weight)
        })
        .collect();

    result
  }

  fn delete_from_zero(&mut self) {
    log_trace!("delete_from_zero");

    let src_id = self.find_or_add_node_by_name(ZERO_NODE.as_str());
    let edges  = self.connected_nodes("", src_id);

    for (src, dst) in edges {
      self.set_edge("", src, dst, 0.0);
    }
  }

  fn top_nodes(&mut self) -> Vec<(NodeId, f64)> {
    log_trace!("top_nodes");

    let reduced = self.reduced_graph();

    if reduced.is_empty() {
      log_error!("(top_nodes) Reduced graph is empty");
      return vec![];
    }

    let mut pr   = Pagerank::<NodeId>::new();
    let     zero = self.find_or_add_node_by_name(ZERO_NODE.as_str());

    reduced
      .iter()
      .filter(|(source, target, _weight)|
        *source != zero && *target != zero
      )
      .for_each(|(source, target, _weight)| {
        // TODO: check weight
        pr.add_edge(*source, *target);
      });

    log_verbose!("Calculate page rank");
    pr.calculate();

    let (nodes, scores): (Vec<NodeId>, Vec<f64>) =
      pr
        .nodes()  // already sorted by score
        .into_iter()
        .take(*TOP_NODES_LIMIT)
        .into_iter()
        .unzip();

    let res = nodes
      .into_iter()
      .zip(scores)
      .collect::<Vec<_>>();

    if res.is_empty() {
      log_error!("(top_nodes) No top nodes");
    }

    return res;
  }

  pub fn write_recalculate_zero(&mut self) {
    log_info!("CMD write_recalculate_zero");

    self.recalculate_all(0); // FIXME Ad hok hack
    self.delete_from_zero();

    let nodes = self.top_nodes();

    self.recalculate_all(0); // FIXME Ad hok hack
    {
      let zero = self.find_or_add_node_by_name(ZERO_NODE.as_str());

      for (k, (node_id, amount)) in nodes.iter().enumerate() {
        if (k % 100) == 90 {
          log_trace!("{}%", (k * 100) / nodes.len());
        }
        self.set_edge("", zero, *node_id, *amount);
      }
    }
    self.recalculate_all(*NUM_WALK); // FIXME Ad hok hack
  }
}
