//  FIXME
//  Floating-point arithmetic for weight calculation will
//  break invariance.
//

mod errors;
mod mrgraph;

use std::thread;
use std::time::Duration;
use std::env::var;
use std::string::ToString;
use itertools::Itertools;
use std::collections::HashMap;
use std::sync::MutexGuard;
use petgraph::graph::{EdgeIndex, NodeIndex};
use nng::{Aio, AioResult, Context, Message, Protocol, Socket};
use simple_pagerank::Pagerank;
use errors::GraphManipulationError;
use mrgraph::{GraphSingleton, GRAPH};
use mrgraph::NodeId;
use meritrank::{MeritRank, MyGraph, MeritRankError, Weight};
use ctrlc;

#[cfg(test)]
mod tests;

lazy_static::lazy_static! {
    static ref SERVICE_URL: String =
        var("MERITRANK_SERVICE_URL")
            .unwrap_or("tcp://127.0.0.1:10234".to_string());

    static ref THREADS : usize =
        var("MERITRANK_SERVICE_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1);

    static ref NUM_WALK: usize =
        var("MERITRANK_NUM_WALK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(10000);

    static ref ZERO_NODE: String =
        var("MERITRANK_ZERO_NODE")
            .unwrap_or("U000000000000".to_string());

    static ref TOP_NODES_LIMIT: usize =
        var("MERITRANK_TOP_NODES_LIMIT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(100);

    static ref EMPTY_RESULT: Vec<u8> = {
        const EMPTY_ROWS_VEC: Vec<(&str, &str, f64)> = Vec::new();
        rmp_serde::to_vec(&EMPTY_ROWS_VEC).unwrap()
    };
}

const VERSION: Option<&str> = option_env!("CARGO_PKG_VERSION");

fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
    ctrlc::set_handler(move || {
        println!("");
        std::process::exit(0)
    })?;

    if *THREADS > 1 {
        main_async(*THREADS)
    } else {
        main_sync()
    }
}

fn main_sync() -> Result<(), Box<dyn std::error::Error + 'static>> {
    println!("Starting server {} at {}", VERSION.unwrap_or("unknown"), *SERVICE_URL);
    println!("NUM_WALK={}", *NUM_WALK);

    let s = Socket::new(Protocol::Rep0)?;
    s.listen(&SERVICE_URL)?;

    loop {
        let request: Message = s.recv()?;
        let reply: Vec<u8> = process(request);
        let _ = s.send(reply.as_slice()).map_err(|(_, e)| e)?;
    }
    // Ok(())
}

fn main_async(threads : usize) -> Result<(), Box<dyn std::error::Error + 'static>> {
    println!("Starting server {} at {}, {} threads", VERSION.unwrap_or("unknown"), *SERVICE_URL, threads);
    println!("NUM_WALK={}", *NUM_WALK);

    let s = Socket::new(Protocol::Rep0)?;

    // Create all of the worker contexts
    let workers: Vec<_> = (0..threads)
        .map(|_| {
            let ctx = Context::new(&s)?;
            let ctx_clone = ctx.clone();
            let aio = Aio::new(move |aio, res| worker_callback(aio, &ctx_clone, res))?;
            Ok((aio, ctx))
        })
        .collect::<Result<_, nng::Error>>()?;

    // Only after we have the workers do we start listening.
    s.listen(&SERVICE_URL)?;

    // Now start all of the workers listening.
    for (a, c) in &workers {
        c.recv(a)?;
    }

    thread::sleep(Duration::from_secs(60 * 60 * 24 * 365)); // 1 year

    Ok(())
}

/// Callback function for workers.
fn worker_callback(aio: Aio, ctx: &Context, res: AioResult) {
    match res {
        // We successfully sent the message, wait for a new one.
        AioResult::Send(Ok(_)) => ctx.recv(&aio).unwrap(),

        // We successfully received a message.
        AioResult::Recv(Ok(req)) => {
            let msg: Vec<u8> = process(req);
            ctx.send(&aio, msg.as_slice()).unwrap();
        }

        AioResult::Sleep(_) => {},

        // Anything else is an error and we will just panic.
        AioResult::Send(Err(e)) =>
            panic!("Error: {}", e.1),

        AioResult::Recv(Err(e)) =>
            panic!("Error: {}", e)
    }
}

fn process(req: Message) -> Vec<u8> {
    let slice = req.as_slice();

    let ctx = GraphContext::null();

    ctx.process(slice)
        .map(|msg| msg)
        .unwrap_or_else(|e| {
            let s: String = e.to_string();
            rmp_serde::to_vec(&s).unwrap()
        })
}

fn mr_service() -> Result<Vec<u8>, Box<dyn std::error::Error + 'static>> {
    let s: String = VERSION.unwrap_or("unknown").to_string();
    Ok(rmp_serde::to_vec(&s)?)
}

fn mr_node_score_null(ego: &str, target: &str) -> Result<Vec<u8>, Box<dyn std::error::Error + 'static>> {
    let w: Weight =
        GraphSingleton::contexts()?
            .iter()
            .filter_map(|context| {
                let mut rank = GraphSingleton::get_rank1(&context).ok()?;
                let ego_id: NodeId = GraphSingleton::node_name_to_id(ego).ok()?; // thread safety?
                let target_id: NodeId = GraphSingleton::node_name_to_id(target).ok()?; // thread safety?
                /*
                let _ = rank.calculate(ego_id, *NUM_WALK).ok()?;
                rank.get_node_score(ego_id, target_id).ok()
                */
                match rank.get_node_score(ego_id, target_id) {
                    Err(MeritRankError::NodeDoesNotCalculated) => {
                        let _ = rank.calculate(ego_id, *NUM_WALK).ok()?;
                        rank.get_node_score(ego_id, target_id).ok()
                    },
                    other => other.ok()
                }
            }) // just skip errors in contexts
            .sum();
    let result: Vec<(&str, &str, f64)> = [(ego, target, w)].to_vec();
    let v: Vec<u8> = rmp_serde::to_vec(&result)?;
    Ok(v)
}

fn node_id2string(node_id: &NodeId) -> String {
    // check if NodeId is not None and not default (== 0)
    // !self.is_none() && *self != NodeId::default()
    match node_id {
        NodeId::None => "false".to_string(),
        NodeId::Int(id) => id.to_string(),
        NodeId::UInt(id) => id.to_string(),
    }
}

fn mr_scores_null(ego: &str) -> Result<Vec<u8>, Box<dyn std::error::Error + 'static>> {
    let result: Vec<_> =
        GraphSingleton::contexts()? // .push(null)
            .iter()
            .filter_map(|context| {
                let mut rank = GraphSingleton::get_rank1(&context).ok()?;
                let ego_id: NodeId = GraphSingleton::node_name_to_id(ego).ok()?; // thread safety?

                /*
                let _ = rank.calculate(node_id, *NUM_WALK).ok()?;
                let rank_result0 = rank.get_ranks(node_id, None).ok()?;
                */
                let rank_result = match rank.get_ranks(ego_id, None) {
                    Err(MeritRankError::NodeDoesNotCalculated) => {
                        let _ = rank.calculate(ego_id, *NUM_WALK).ok()?;
                        rank.get_ranks(ego_id, None).ok()
                    },
                    other => other.ok()
                };
                let rows: Vec<_> =
                    rank_result?
                        .into_iter()
                        .map(|(n, s)| {
                            (
                                (
                                    ego,
                                    GraphSingleton::node_id_to_name(n)
                                        .unwrap_or(node_id2string(&n))
                                ),
                                s,
                            )
                        })
                        .collect();
                Some(rows)
            })
            .flatten()
            .into_iter()
            .group_by(|(nodes, _)| nodes.clone())
            .into_iter()
            .map(|((src, target), rows)|
                (src, target, rows.map(|(_, score)| score).sum::<Weight>())
            )
            .collect();

    let v: Vec<u8> = rmp_serde::to_vec(&result)?;
    Ok(v)
}


pub struct GraphContext {
    context: Option<String>,
}

impl GraphContext {
    pub fn null() -> GraphContext {
        GraphContext {
            context: None
        }
    }
    pub fn new(context_init: &str) -> GraphContext {
        if context_init.is_empty() {
            GraphContext {
                context: None
            }
        } else {
            GraphContext {
                context: Some(context_init.to_string())
            }
        }
    }

    pub fn process_context(context: &str, payload: Vec<u8>)  -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        GraphContext::new(&context).process(payload.as_slice())
    }

    pub fn process(&self, slice: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        if let Ok("ver") = rmp_serde::from_slice(slice) {
            mr_service()
        } else if let Ok(((("src", "=", ego), ("dest", "=", target)), (), "null")) = rmp_serde::from_slice(slice) {
            mr_node_score_null(ego, target)
        } else if let Ok(((("src", "=", ego), ), (), "null")) = rmp_serde::from_slice(slice) {
            mr_scores_null(ego)
        } else if let Ok(("context", context, payload)) = rmp_serde::from_slice(slice) { // rmp_serde::from_slice::<(&str, &str, Vec<u8>)>(slice) {
            Self::process_context(context, payload)
        } else if let Ok(((("src", "=", ego), ("dest", "=", target)), ())) = rmp_serde::from_slice(slice) {
            self.mr_node_score(ego, target)
        } else if let Ok(((("src", "=", ego), ), ())) = rmp_serde::from_slice(slice) {
            self.mr_scores(ego, "", false, f64::MIN, true, f64::MAX, true, None)
        } else if let Ok(((("src", "=", ego), ("target", "like", target_like), ("score", ">", score_gt), ("score", "<", score_lt)), ())) = rmp_serde::from_slice(slice) {
            self.mr_scores(ego, target_like, false, score_gt, false, score_lt, false, None)
        } else if let Ok(((("src", "=", ego), ("target", "like", target_like), ("score", ">=", score_gte), ("score", "<", score_lt)), ())) = rmp_serde::from_slice(slice) {
            self.mr_scores(ego, target_like, false, score_gte, true, score_lt, false, None)
        } else if let Ok(((("src", "=", ego), ("target", "like", target_like), ("score", ">", score_gt), ("score", "<=", score_lt)), ())) = rmp_serde::from_slice(slice) {
            self.mr_scores(ego, target_like, false, score_gt, false, score_lt, true, None)
        } else if let Ok(((("src", "=", ego), ("target", "like", target_like), ("score", ">=", score_gte), ("score", "<=", score_lt)), ())) = rmp_serde::from_slice(slice) {
            self.mr_scores(ego, target_like, false, score_gte, true, score_lt, true, None)
        } else if let Ok(((("src", "=", ego), ("target", "like", target_like), ("hide_personal", hide_personal), ("score", ">", score_gt), ("score", "<", score_lt), ("limit", limit)), ())) = rmp_serde::from_slice(slice) {
            self.mr_scores(ego, target_like, hide_personal, score_gt, false, score_lt, false, limit)
        } else if let Ok(((("src", "=", ego), ("target", "like", target_like), ("hide_personal", hide_personal), ("score", ">=", score_gte), ("score", "<", score_lt), ("limit", limit)), ())) = rmp_serde::from_slice(slice) {
            self.mr_scores(ego, target_like, hide_personal, score_gte, true, score_lt, false, limit)
        } else if let Ok(((("src", "=", ego), ("target", "like", target_like), ("hide_personal", hide_personal), ("score", ">", score_gt), ("score", "<=", score_lt), ("limit", limit)), ())) = rmp_serde::from_slice(slice) {
            self.mr_scores(ego, target_like, hide_personal, score_gt, false, score_lt, true, limit)
        } else if let Ok(((("src", "=", ego), ("target", "like", target_like), ("hide_personal", hide_personal), ("score", ">=", score_gte), ("score", "<=", score_lt), ("limit", limit)), ())) = rmp_serde::from_slice(slice) {
            self.mr_scores(ego, target_like, hide_personal, score_gte, true, score_lt, true, limit)
        } else if let Ok((((subject, object, amount), ), ())) = rmp_serde::from_slice(slice) {
            self.mr_edge(subject, object, amount)
        } else if let Ok(((("src", "delete", ego), ("dest", "delete", target)), ())) = rmp_serde::from_slice(slice) {
            self.mr_delete_edge(ego, target)
        } else if let Ok(((("src", "delete", ego), ), ())) = rmp_serde::from_slice(slice) {
            self.mr_delete_node(ego)
        } else if let Ok((((ego, "gravity", focus), positive_only, limit), ())) = rmp_serde::from_slice(slice) {
            self.mr_gravity_graph(ego, focus, positive_only/* true */, limit/* 3 */)
        } else if let Ok((((ego, "gravity_nodes", focus), positive_only, limit), ())) = rmp_serde::from_slice(slice) {
            self.mr_gravity_nodes(ego, focus, positive_only /* false */, limit /* 3 */)
        } else if let Ok((((ego, "connected"), ), ())) = rmp_serde::from_slice(slice) {
            self.mr_connected(ego)
        } else if let Ok(("for_beacons_global", ())) = rmp_serde::from_slice(slice) {
            self.mr_beacons_global()
        } else if let Ok(("nodes", ())) = rmp_serde::from_slice(slice) {
            self.mr_nodes()
        } else if let Ok(("edges", ())) = rmp_serde::from_slice(slice) {
            self.mr_edges()
        } else if let Ok(("zerorec", ())) = rmp_serde::from_slice(slice) {
            self.mr_zerorec()
        } else {
            let err: String = format!("Error: Cannot understand request {:?}", slice);
            Err(err.into())
        }
    }

    fn get_rank(&self) -> Result<MeritRank, GraphManipulationError> {
        match &self.context {
            // TODO: it's thread safe as get_rank/get_rank1 do safe copy all the graph now
            None => GraphSingleton::get_rank(),
            Some(ctx) if ctx.is_empty() => GraphSingleton::get_rank(),
            Some(ctx) => GraphSingleton::get_rank1(&ctx),
        }
    }

    fn get_node_id(
        &self,
        graph : &mut MutexGuard<GraphSingleton>,
        name  : &str
    ) -> NodeId {
        match &self.context {
            None => graph.get_node_id(name),
            Some(ctx) if ctx.is_empty() =>  graph.get_node_id(name),
            Some(ctx) => graph.get_node_id1(ctx.as_str(), name)
        }
    }

    fn mr_node_score(&self, ego: &str, target: &str) -> Result<Vec<u8>, Box<dyn std::error::Error + 'static>> {
        let mut rank = self.get_rank()?;
        let ego_id: NodeId = GraphSingleton::node_name_to_id(ego)?; // thread safety?
        let target_id: NodeId = GraphSingleton::node_name_to_id(target)?; // thread safety?
        let _ = rank.calculate(ego_id, *NUM_WALK)?;
        let w: Weight = rank.get_node_score(ego_id, target_id)?;
        let result: Vec<(&str, &str, f64)> = [(ego, target, w)].to_vec();
        let v: Vec<u8> = rmp_serde::to_vec(&result)?;
        Ok(v)
    }

    fn mr_scores(&self, ego: &str,
                 target_like: &str,
                 hide_personal: bool,
                 score_lt: f64, score_lte: bool,
                 score_gt: f64, score_gte: bool,
                 limit: Option<i32>) ->
        Result<Vec<u8>, Box<dyn std::error::Error + 'static>>
    {
        let mut rank = self.get_rank()?;
        let node_id: NodeId = GraphSingleton::node_name_to_id(ego)?; // thread safety?
        let _ = rank.calculate(node_id, *NUM_WALK)?;

        let result = rank
            .get_ranks(node_id, None)?
            .into_iter()
            .map(|(n, w)| {
                (
                    ego,
                    GraphSingleton::node_id_to_name(n).unwrap_or(node_id2string(&n)),
                    w,
                )
            })
            .filter(|(_, target, _)| target.starts_with(target_like))
            .filter(|(_, _, score)| score_gt < *score || (score_gte && score_gt == *score))
            .filter(|(_, _, score)| *score < score_lt || (score_lte && score_lt == *score))
            .filter(|(_ego, target, _)|
                if hide_personal {
                    match GraphSingleton::node_name_to_id(target) { // thread safety?
                        Ok(target_id) =>
                            !((target.starts_with("C") || target.starts_with("B")) &&
                                rank.get_edge(target_id, node_id).is_some()),
                        _ => true
                    }
                } else { true }
            );

        let limited: Vec<(&str, String, Weight)> =
            match limit {
                Some(limit) => result.take(limit.try_into().unwrap()).collect(),
                None => result.collect(),
            };

        let v: Vec<u8> = rmp_serde::to_vec(&limited)?;
        Ok(v)
    }

    fn mr_edge(
        &self,
        subject: &str,
        object: &str,
        amount: f64,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + 'static>> {
        // "result" isn't depend on operation here
        let result: Vec<(&str, &str, f64)> = [(subject, object, amount)].to_vec();
        let v: Vec<u8> = rmp_serde::to_vec(&result)?;

        // meritrank_add(subject, object, amount)?;
        let mut graph = GRAPH.lock()?;
        let subject_id = self.get_node_id(&mut graph, subject);
        let object_id = self.get_node_id(&mut graph, object);

        if let Some(context) = &self.context {
            let contexted_graph = graph.borrow_graph_mut1(context);
            let old_weight =
                contexted_graph
                    .edge_weight(subject_id.into(), object_id.into())
                    .unwrap_or(0.0);

            contexted_graph
                .upsert_edge_with_nodes(subject_id.into(), object_id.into(), amount)?;

            let null_graph = graph.borrow_graph_mut();
            match null_graph.edge_weight(subject_id.into(), object_id.into()) {
                Some(null_weight) =>
                    //*null_weight = *null_weight + amount - old_weight; // todo: check
                    null_graph.upsert_edge(subject_id.into(), object_id.into(), null_weight + amount - old_weight)?,
                _ => {
                    let _ = null_graph.upsert_edge(subject_id.into(), object_id.into(), amount)?;
                }
            }
        } else {
            graph
                .borrow_graph_mut()
                .upsert_edge(subject_id.into(), object_id.into(), amount)?;
        }

        Ok(v)
    }

    fn delete_edge_locked(
        &self,
        graph  : &mut MutexGuard<GraphSingleton>,
        src_id : NodeId,
        dst_id : NodeId
    ) -> Result<(), Box<dyn std::error::Error + 'static>> {
        if let Some(context) = &self.context {
            let contexted_graph = graph.borrow_graph_mut1(context);
            let old_weight =
                contexted_graph
                    .edge_weight(src_id.into(), dst_id.into())
                    .unwrap_or(0.0);

            contexted_graph
                .remove_edge(src_id.into(), dst_id.into());

            let null_graph  = graph.borrow_graph_mut();
            let null_weight = null_graph.edge_weight(src_id.into(), dst_id.into()).unwrap_or(0.0);
            let new_weight  = null_weight - old_weight;

            let _ = null_graph.upsert_edge(src_id.into(), dst_id.into(), new_weight)?;

            //  TODO
            //  Count all countexts.
        } else {
            graph
                .borrow_graph_mut()
                .remove_edge(src_id.into(), dst_id.into());
        }

        // TODO: use node garbage collection

        Ok(())
    }

    fn mr_delete_edge(
        &self,
        src : &str,
        dst : &str,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + 'static>> {
        let mut graph = GRAPH.lock()?;

        let src_id = self.get_node_id(&mut graph, src);
        let dst_id = self.get_node_id(&mut graph, dst);

        self.delete_edge_locked(&mut graph, src_id, dst_id)?;

        Ok(EMPTY_RESULT.to_vec())
    }

    fn mr_delete_node(
        &self,
        ego : &str,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + 'static>> {
        let mut graph  = GRAPH.lock()?;
        let     ego_id = self.get_node_id(&mut graph, ego);

        let my_graph = graph.borrow_graph_mut();

        for n in my_graph.neighbors(ego_id).iter() {
            self.delete_edge_locked(&mut graph, ego_id, *n)?;
        }

        Ok(EMPTY_RESULT.to_vec())
    }

    fn gravity_graph(
        &self,
        ego: &str,
        focus: &str,
        positive_only: bool,
        limit: i32,
    ) -> Result<
            (Vec<(String, String, Weight)>, HashMap<String, Weight>),
            Box<dyn std::error::Error + 'static>
    > {
        let mut rank = self.get_rank()?;

        match GRAPH.lock() {
            Ok(graph) => {
                //let mut rank: MeritRank = self.get_rank()?; // ***
                // MeritRank::new(graph.borrow_graph().clone())?;
                // ? should we change weight/scores in GRAPH ?

                let focus_id = graph.node_name_to_id_unsafe(focus)?;

                let mut copy = MyGraph::new();

                let source_graph =
                    if let Some(context) = &self.context {
                        graph.borrow_graph0(context)? // todo // ??
                    } else {
                        graph.borrow_graph()
                    };

                let focus_vector: Vec<(NodeId, NodeId, Weight)> =
                    source_graph.edges(focus_id).into_iter().flatten().collect();

                for (a_id, b_id, w_ab) in focus_vector {
                    //let a: String = graph.node_id_to_name_unsafe(a_id)?;
                    let b: String = graph.node_id_to_name_unsafe(b_id)?;

                    if b.starts_with("U") {
                        if positive_only {
                            let score = match rank.get_node_score(a_id, b_id) {
                                Ok(x) => x,
                                Err(MeritRankError::NodeDoesNotCalculated) => {
                                    rank.calculate(a_id, *NUM_WALK)?;
                                    rank.get_node_score(a_id, b_id)?
                                },
                                Err(x) => {
                                    return Err(x.into());
                                }
                            };
                            if score <= 0f64 {
                                continue;
                            }
                        }
                        // assert!( get_edge(a, b) != None);

                        let _ = copy.upsert_edge_with_nodes(a_id, b_id, w_ab)?;
                    } else if b.starts_with("C") || b.starts_with("B") {
                        // ? # For connections user-> comment | beacon -> user,
                        // ? # convolve those into user->user

                        let v_b: Vec<(NodeId, NodeId, Weight)> =
                            source_graph.edges(b_id).into_iter().flatten().collect();

                        for (_, c_id, w_bc) in v_b {
                            if positive_only && w_bc <= 0.0f64 {
                                continue;
                            }
                            if c_id == a_id || c_id == b_id { // note: c_id==b_id not in Python version !?
                                continue;
                            }

                            let c: String = graph.node_id_to_name_unsafe(c_id)?;

                            if !c.starts_with("U") {
                                continue;
                            }
                            // let w_ac = self.get_transitive_edge_weight(a, b, c);
                            // TODO: proper handling of negative edges
                            // Note that enemy of my enemy is not my friend.
                            // Though, this is pretty irrelevant for our current case
                            // where comments can't have outgoing negative edges.
                            // return w_ab * w_bc * (-1 if w_ab < 0 and w_bc < 0 else 1)
                            let w_ac: f64 =
                                w_ab * w_bc * (if w_ab < 0.0f64 && w_bc < 0.0f64 { -1.0f64 } else { 1.0f64 });

                            let _ = copy.upsert_edge_with_nodes(a_id, c_id, w_ac)?;
                        }
                    }
                }

                // self.remove_outgoing_edges_upto_limit(G, ego, focus, limit or 3):
                // neighbours = list(dest for src, dest in G.out_edges(focus))

                let neighbours: Vec<(EdgeIndex, NodeIndex, NodeId)> = copy.outgoing(focus_id);

                // ego_id in graph
                let ego_id: NodeId = graph.node_name_to_id_unsafe(ego)?;

                let mut sorted: Vec<(Weight, (&EdgeIndex, &NodeIndex))> =
                    neighbours
                        .iter()
                        .map(|(edge_index, node_index, node_id)| {
                            let w: f64 = rank.get_node_score(ego_id, *node_id).unwrap_or(0f64);
                            (w, (edge_index, node_index))
                        })
                        .collect::<Vec<_>>();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                //sort by weight

                // for dest in sorted(neighbours, key=lambda x: self.get_node_score(ego, x))[limit:]:
                let limited: Vec<&(&EdgeIndex, &NodeIndex)> =
                    sorted.iter()
                        .map(|(_, tuple)| tuple)
                        .take(limit.try_into().unwrap())
                        .collect();

                for (_edge_index, node_index) in limited {
                    let node_id = copy.index2node(**node_index);
                    copy.remove_edge(ego_id, node_id);
                    //G.remove_node(dest) // ???
                }

                // add_path_to_graph(G, ego, focus)
                let path: Vec<NodeId> =
                    copy
                        .shortest_path(ego_id, focus_id)
                        .unwrap_or(Vec::new());
                // add_path_to_graph(G, ego, focus)
                // Note: no loops or "self edges" are expected in the path
                let ok: Result<(), GraphManipulationError> = {
                    //  FIXME
                    //  limit.unwrap() can panic
                    let v3: Vec<&NodeId> = path.iter().take(limit.try_into().unwrap()).collect::<Vec<&NodeId>>(); // was: (3)
                    if let Some((&a, &b, &c)) = v3.clone().into_iter().collect_tuple() {
                        // # merge transitive edges going through comments and beacons

                        // ???
                        /*
                        if c is None and not (a.startswith("C") or a.startswith("B")):
                            new_edge = (a, b, self.get_edge(a, b))
                        elif ... */

                        let a_name = graph.node_id_to_name_unsafe(a)?;
                        let b_name = graph.node_id_to_name_unsafe(b)?;
                        let c_name = graph.node_id_to_name_unsafe(c)?;
                        if b_name.starts_with("C") || b_name.starts_with("B") {
                            let w_ab =
                                copy.edge_weight(a, b)
                                    .ok_or(GraphManipulationError::WeightExtractionFailure(
                                        format!("Cannot extract weight from {} to {}",
                                                a_name, b_name
                                        )
                                    ))?;
                            let w_bc =
                                copy.edge_weight(b, c)
                                    .ok_or(GraphManipulationError::WeightExtractionFailure(
                                        format!("Cannot extract weight from {} to {}",
                                                a_name, c_name
                                        )
                                    ))?;
                            // get_transitive_edge_weight
                            let w_ac: f64 =
                                w_ab * w_bc * (if w_ab < 0.0f64 && w_bc < 0.0f64 { -1.0f64 } else { 1.0f64 });
                            copy.upsert_edge(a, c, w_ac)?;
                            Ok(())
                        } else if a_name.starts_with("U") {
                            let weight =
                                copy.edge_weight(a, b)
                                    .ok_or(GraphManipulationError::WeightExtractionFailure(
                                        format!("Cannot extract weight from {} to {}",
                                                a_name, b_name
                                        )
                                    ))?;
                            copy.upsert_edge(a, b, weight)?;
                            Ok(())
                        } else {
                            Ok(())
                        }
                    } else if let Some((&a, &b)) = v3.clone().into_iter().collect_tuple()
                    {
                        /*
                        # Add the final (and only)
                        final_nodes = ego_to_focus_path[-2:]
                        final_edge = (*final_nodes, self.get_edge(*final_nodes))
                        edges.append(final_edge)
                        */
                        // ???
                        let a_name = graph.node_id_to_name_unsafe(a)?;
                        let b_name = graph.node_id_to_name_unsafe(b)?;
                        let weight =
                            copy.edge_weight(a, b)
                                .ok_or(GraphManipulationError::WeightExtractionFailure(
                                    format!("Cannot extract weight from {} to {}",
                                            a_name, b_name
                                    )
                                ))?;
                        copy.upsert_edge(a, b, weight)?;
                        Ok(())
                    } else if v3.len() == 1 {
                        // ego == focus ?
                        // do nothing
                        Ok(())
                    } else if v3.is_empty() {
                        // No path found, so add just the focus node to show at least something
                        //let node = mrgraph::meritrank::node::Node::new(focus_id);
                        let node = meritrank::node::Node::new(focus_id);
                        copy.add_node(node);
                        Ok(())
                    } else {
                        Err(errors::GraphManipulationError::DataExtractionFailure(
                            "Should never be here (v3)".to_string()
                        ))
                    }
                };
                let _ = ok?;

                // self.remove_self_edges(copy);
                // todo: just not let them pass into the graph

                let (nodes, edges) = copy.all();

                let table: Vec<(String, String, f64)> =
                    edges
                        .iter()
                        .map(|(n1, n2, weight)| {
                            let name1 = graph.node_id_to_name_unsafe(*n1)?;
                            let name2 = graph.node_id_to_name_unsafe(*n2)?;
                            Ok::<(String, String, f64), GraphManipulationError>((name1, name2, *weight))
                        })
                        .collect::<Vec<_>>()
                        .into_iter()
                        .flatten()
                        .collect::<Vec<_>>();

                let nodes_dict: HashMap<String, Weight> =
                    nodes
                        .iter()
                        .map(|node_id| {
                            let name = graph.node_id_to_name_unsafe(*node_id)?;

                            if !rank.get_personal_hits().contains_key(&ego_id) {
                                let _ = rank.calculate(ego_id, *NUM_WALK)?;
                            }
                            let score =
                                rank.get_node_score(ego_id, *node_id)?;
                            Ok::<(String, Weight), GraphManipulationError>((name, score))
                        })
                        .collect::<Vec<_>>()
                        .into_iter()
                        .flatten()
                        .collect::<HashMap<String, Weight>>();

                Ok((table, nodes_dict))
            }
            Err(e) => Err(
                Box::new(
                    GraphManipulationError::MutexLockFailure(format!(
                        "Mutex lock error: {}",
                    e
            ))))
        }
    }

    fn mr_gravity_graph(
        &self,
        ego: &str,
        focus: &str,
        positive_only: bool,
        limit: Option<i32>
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + 'static>> {
        let (result, _) = self.gravity_graph(ego, focus, positive_only, limit.unwrap_or(i32::MAX))?;
        let v: Vec<u8> = rmp_serde::to_vec(&result)?;
        Ok(v)
    }

    fn mr_gravity_nodes(
        &self,
        ego: &str,
        focus: &str,
        positive_only: bool,
        limit: Option<i32>
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + 'static>> {
        // TODO: change HashMap to string pairs here!?
        let (_, hash_map) = self.gravity_graph(ego, focus, positive_only, limit.unwrap_or(i32::MAX))?;
        let result: Vec<_> = hash_map.iter().collect();
        let v: Vec<u8> = rmp_serde::to_vec(&result)?;
        Ok(v)
    }

    fn get_connected(&self, ego : &str) -> Result<Vec<(String, String)>, Box<dyn std::error::Error + 'static>> {
        let mut graph = GRAPH.lock()?;

        let node_id = graph.node_name_to_id_unsafe(ego);

        match node_id {
            Err(_) => return Ok(Vec::<(String, String)>::new()),
            _      => {}
        };

        let my_graph : &MyGraph =
            match &self.context {
                None      => graph.borrow_graph(),
                Some(ctx) => graph.borrow_graph1(ctx)
            };

        let result: Vec<(String, String)> =
            my_graph
                .connected(node_id?)
                .iter()
                .map(|(_edge_index, from, to)|
                    (
                        graph.node_id_to_name_unsafe(*from).unwrap_or(node_id2string(from)),
                        graph.node_id_to_name_unsafe(*to).unwrap_or(node_id2string(to))
                    )
                )
                .collect();

        return Ok(result);
    }

    fn mr_connected(
        &self,
        ego: &str
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + 'static>> {
        let edges = self.get_connected(ego)?;
        if edges.is_empty() {
            return Err("No edges".into());
        }
        return Ok(rmp_serde::to_vec(&edges)?);
    }

    fn get_reduced_graph(&self) -> Result<Vec<(String, String, f64)>, Box<dyn std::error::Error + 'static>> {
        let mut rank  = self.get_rank()?;
        let     graph = GRAPH.lock()?;

        let node_names : HashMap<String, NodeId> =
            graph.borrow_node_names().clone();

        let node_ids : HashMap<NodeId, String> =
            node_names
                .clone() // ?
                .into_iter()
                .map(|(name, id)| (id, name))
                .collect();

        let users : Vec<(String, NodeId)> =
            node_names
                .clone() // ?
                .into_iter()
                .filter(|(name, _)| name.starts_with("U")) // filter zero user?
                .collect();

        if users.is_empty() {
            return Ok(Vec::new());
        }

        for (_, node_id) in users.iter() {
            rank.calculate(*node_id, *NUM_WALK)?;
        }

        let edges : Vec<(NodeId, NodeId, Weight)> =
            users.into_iter()
                .map(|(_name, ego_id)| {
                    let result: Vec<(NodeId, NodeId, Weight)> =
                        rank.get_ranks(ego_id, None)?
                        .into_iter()
                        .map(|(node_id, score)| (ego_id, node_id, score))
                        .filter(|(ego_id, node_id, score)|
                            // graph.node_id_to_name_unsafe(*node_id)
                            node_ids.get(node_id)
                                .map(|node| (node.starts_with("U") || node.starts_with("B")) &&
                                    *score > 0.0 &&
                                    ego_id != node_id)
                                .unwrap_or(false) // todo: log
                        ).collect();
                    Ok::<Vec<(NodeId, NodeId, Weight)>, MeritRankError>(result)
                })
                .filter_map(|res| res.ok())
                .flatten()
                .collect::<Vec<(NodeId, NodeId, Weight)>>();

        //let (_, edges) = my_graph.all(); // not optimal
        // Note:
        // Just eat errors in node_id_to_name_unsafe bellow.
        // Should we pass them out?
        let result : Vec<(String, String, f64)> =
            edges
                .iter()
                .filter(|(ego_id, dest_id, _)|
                            ego_id != dest_id
                        // Todo: filter if ego or dest is Zero here (?)
                )
                .flat_map(|(ego_id, dest_id, weight)| {
                    let ego = graph.node_id_to_name_unsafe(*ego_id)?;
                    Ok::<(String, &NodeId, &Weight), GraphManipulationError>((ego, dest_id, weight))
                })
                .filter(|(ego, _dest_id, _weight)|
                    ego.starts_with("U")
                )
                .flat_map(|(ego, dest_id, weight)| {
                    let dest = graph.node_id_to_name_unsafe(*dest_id)?;
                    Ok::<(String, String, Weight), GraphManipulationError>((ego, dest, *weight))
                })
                .filter(|(_ego, dest, _weight)|
                    dest.starts_with("U") || dest.starts_with("B")
                )
                .collect();

        return Ok(result);
    }

    fn mr_beacons_global(&self) -> Result<Vec<u8>, Box<dyn std::error::Error + 'static>> {
        Ok(rmp_serde::to_vec(&self.get_reduced_graph()?)?)
    }

    fn mr_nodes(&self) -> Result<Vec<u8>, Box<dyn std::error::Error + 'static>> {
        let mut graph = GRAPH.lock()?;
        let my_graph = // self.borrow_graph(graph);
            match &self.context {
                None => graph.borrow_graph(),
                Some(ctx) => graph.borrow_graph1(ctx)
            };

        let (nodes, _) = my_graph.all(); // not optimal

        let result: Vec<String> =
            nodes
                .iter()
                .map(|&node_id|
                    graph.node_id_to_name_unsafe(node_id)
                )
                .into_iter()
                .collect::<Result<Vec<String>, GraphManipulationError>>()?;

        let v: Vec<u8> = rmp_serde::to_vec(&result)?;
        Ok(v)
    }

    fn mr_edges(&self) -> Result<Vec<u8>, Box<dyn std::error::Error + 'static>> {
        let mut graph    = GRAPH.lock()?;
        let     my_graph =
            match &self.context {
                None      => graph.borrow_graph(),
                Some(ctx) => graph.borrow_graph1(ctx)
            };

        let (_, edges) = my_graph.all(); // not optimal

        let result: Vec<(String, String, Weight)> =
            edges
                .iter()
                .map(|&(from_id, to_id, w)| {
                    let from = graph.node_id_to_name_unsafe(from_id)?;
                    let to = graph.node_id_to_name_unsafe(to_id)?;
                    Ok((from, to, w))
                })
                .collect::<Result<Vec<(String, String, Weight)>, GraphManipulationError>>()?;

        let v: Vec<u8> = rmp_serde::to_vec(&result)?;
        Ok(v)
    }

    fn delete_from_zero(&self) -> Result<(), Box<dyn std::error::Error + 'static>> {
        let edges = self.get_connected(&ZERO_NODE)?;

        for (src, dst) in edges.iter() {
            let _ = self.mr_delete_edge(src, dst)?;
        }

        return Ok(());
    }

    fn top_nodes(&self) -> Result<Vec<(String, f64)>, Box<dyn std::error::Error + 'static>> {
        let reduced = self.get_reduced_graph()?;

        if reduced.is_empty() {
            return Err("Reduced graph empty".into());
        }

        let mut pr = Pagerank::<&String>::new();

        reduced
            .iter()
            .filter(|(source, target, _weight)|
                *source!=*ZERO_NODE && *target!=*ZERO_NODE
            )
            .for_each(|(source, target, _weight)| {
                // TODO: check weight
                pr.add_edge(source, target);
            });

        pr.calculate();

        let (nodes, scores): (Vec<&&String>, Vec<f64>) =
            pr
                .nodes()    // already sorted by score
                .into_iter()
                .take(*TOP_NODES_LIMIT)
                .into_iter()
                .unzip();

        let res = nodes
            .into_iter()
            .cloned()
            .cloned()
            .zip(scores)
            .collect::<Vec<_>>();

        if res.is_empty() {
            return Err("No top nodes".into());
        }

        return Ok(res);
    }

    fn mr_zerorec(&self) -> Result<Vec<u8>, Box<dyn std::error::Error + 'static>> {
        //  NOTE
        //  This func is not thread-safe.

        self.delete_from_zero()?;

        let nodes = self.top_nodes()?;

        for (name, amount) in nodes.iter() {
            let _ = self.mr_edge(ZERO_NODE.as_str(), name.as_str(), *amount)?;
        }

        return Ok(rmp_serde::to_vec(&"Ok".to_string())?);
    }
}
