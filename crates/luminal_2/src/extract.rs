use crate::Kernel;
use crate::symbolic::{Expression, Term};
use crate::{GPUArch, GraphTerm, run::run_graph};
use colored::Colorize;
use egraph_serialize::{ClassId, EGraph, NodeId};
use petgraph::algo::toposort;
use petgraph::prelude::{NodeIndex, StableGraph};
use petgraph::{Directed, Direction};
use rustc_hash::{FxHashMap, FxHashSet};

const WARMUP_TRIALS: usize = 1;
const TRIALS: usize = 3;
const MAX_SEARCHED_GRAPHS: usize = 10_000;
const MAX_CYCLES: usize = 1;
const INVALID_IR: &[&str] = &["SwapLoops", "Unary", "Binary"];

type Cost = u128; // Execution time in microseconds

use std::collections::HashMap;
use std::u128;

/// Enumerate every valid extraction, measure its runtime, and keep the fastest.
/// Cycles in the e-graph are handled by refusing to revisit a class already
/// on the current DFS path.
pub fn search(egraph: &EGraph, inputs: &[Vec<f32>]) -> Option<StableGraph<Kernel, (u8, u8)>> {
    // ───────────────────────── classes → legal node lists
    let mut class_nodes: FxHashMap<ClassId, Vec<NodeId>> = FxHashMap::default();
    for (nid, node) in &egraph.nodes {
        if INVALID_IR.contains(&node.op.as_str()) {
            continue;
        }
        class_nodes
            .entry(node.eclass.clone())
            .or_default()
            .push(nid.clone());
    }
    class_nodes.retain(|_, v| !v.is_empty());
    if class_nodes.is_empty() {
        println!("No legal nodes after applying INVALID_IR.");
        return None;
    }

    // ───────────────────────── fingerprint helpers
    fn sig(er: &ExtractionResult) -> String {
        let mut v: Vec<_> = er
            .choices
            .iter()
            .map(|(c, n)| format!("{:?}:{:?}", c, n))
            .collect();
        v.sort();
        v.join("|")
    }

    // ───────────────────────── DFS
    fn dfs<'a>(
        stack: Vec<&'a ClassId>,
        class_nodes: &'a FxHashMap<ClassId, Vec<NodeId>>,
        current: &mut FxHashMap<&'a ClassId, &'a NodeId>,
        on_path: &mut FxHashSet<&'a ClassId>, // cycle guard
        printed: &mut usize,
        egraph: &'a EGraph,
        inputs: &[Vec<f32>],
        seen: &mut FxHashSet<String>, // dedup graphs
        ref_outputs: &mut Vec<Vec<f32>>,
        best_graph: &mut StableGraph<Kernel, (u8, u8)>,
        best_time: &mut u128,
    ) {
        if *printed >= MAX_SEARCHED_GRAPHS {
            return;
        }
        if stack.is_empty() {
            let er = ExtractionResult {
                choices: current.clone(),
                ..Default::default()
            };
            if !seen.insert(sig(&er)) {
                return;
            }
            let Some(graph) = extraction_to_graph(egraph, &er, &egraph.root_eclasses) else {
                return;
            };
            let key = serde_json::to_string(&graph).unwrap();
            if !seen.insert(key) {
                return;
            }
            let root = graph.externals(Direction::Outgoing).next().unwrap();
            let Some(kernels) = crate::codegen::codegen(
                graph.clone(),
                root,
                GPUArch::Metal(HashMap::new()),
                *printed,
            ) else {
                return;
            };
            if let Some((us, outs)) = cost(&kernels, inputs) {
                println!(
                    "{}{}",
                    format!("Graph {printed} ").bold(),
                    format!("{us}µs").bright_green().bold()
                );
                if ref_outputs.is_empty() {
                    *ref_outputs = outs;
                } else {
                    for (a, b) in ref_outputs.iter().zip(&outs) {
                        for (x, y) in a.iter().zip(b) {
                            assert!((x - y).abs() <= 1e-3, "{x} | {y}");
                        }
                    }
                    println!("{}", "Outputs Validated".on_bright_green());
                }
                *printed += 1;
                if us < *best_time {
                    *best_time = us;
                    *best_graph = kernels;
                }
            }
            return;
        }

        let mut stack = stack;
        let cid = stack.pop().unwrap();

        // cycle breaker: skip if this class is already on the current path
        if !on_path.insert(cid) {
            return;
        }

        if let Some(nodes) = class_nodes.get(&cid) {
            for nid in nodes {
                current.insert(cid, nid);

                // push child classes not yet decided and not already on path
                let mut next = stack.clone();
                for child in &egraph.nodes[nid].children {
                    let ccid = egraph.nid_to_cid(child);
                    if class_nodes.contains_key(&ccid)
                        && !current.contains_key(&ccid)
                        && !on_path.contains(&ccid)
                    {
                        next.push(ccid);
                    }
                }

                dfs(
                    next,
                    class_nodes,
                    current,
                    on_path,
                    printed,
                    egraph,
                    inputs,
                    seen,
                    ref_outputs,
                    best_graph,
                    best_time,
                );
                current.remove(&cid);
            }
        }

        on_path.remove(&cid); // pop when unwinding
    }

    // ───────────────────────── kick off from roots
    let init: Vec<&ClassId> = egraph
        .root_eclasses
        .iter()
        .filter(|cid| class_nodes.contains_key(*cid))
        .collect();
    if init.is_empty() {
        println!("Roots contain no legal nodes.");
        return None;
    }

    let mut best_graph = StableGraph::default();
    dfs(
        init,
        &class_nodes,
        &mut FxHashMap::default(),
        &mut FxHashSet::default(), // on_path
        &mut 0,
        egraph,
        inputs,
        &mut FxHashSet::default(), // seen
        &mut vec![],
        &mut best_graph,
        #[allow(const_item_mutation)]
        &mut u128::MAX,
    );
    Some(best_graph)
}

#[derive(Default, Clone)]
pub struct ExtractionResult<'a> {
    pub choices: FxHashMap<&'a ClassId, &'a NodeId>,
}

fn cost<'a>(
    kernels: &StableGraph<Kernel, (u8, u8), Directed>,
    inputs: &[Vec<f32>],
) -> Option<(Cost, Vec<Vec<f32>>)> {
    // Print kernels
    if option_env!("PRINT_KERNELS")
        .map(|s| s.parse::<i32>().map(|i| i == 1).unwrap_or_default())
        .unwrap_or_default()
    {
        println!("Kernels: {}", kernels.node_count() - 2);
        for (i, node) in toposort(&kernels, None).unwrap().into_iter().enumerate() {
            let Kernel {
                code,
                grid,
                threadblock,
                smem,
                outputs,
            } = kernels.node_weight(node).unwrap();
            if code != "Inputs" && code != "Outputs" {
                println!("Kernel {i} Grid: {grid:?} Threadblock: {threadblock:?} Smem: {smem}");
                println!("{code}");
                println!("Outputs: {:?}", outputs);
            }
        }
    }

    // Warm up resources (buffer allocation, kernel compiler, etc.)
    for _ in 0..WARMUP_TRIALS {
        run_graph(inputs, &kernels);
    }
    // Test runtime
    let mut micros = vec![];
    let mut outputs = vec![];
    let mut m;
    for _ in 0..TRIALS {
        (outputs, m) = run_graph(inputs, &kernels);
        micros.push(m);
    }
    Some((micros.into_iter().sum::<u128>() / TRIALS as u128, outputs))
}

/// Build a StableGraph from an ExtractionResult.
///
/// * `extraction.choices` : HashMap<ClassId, NodeId>
/// * `roots`              : entry-point e-classes
///
/// Any helper like `math_from_class()` can be filled in later to turn
/// Loop ranges / strides into your own `Expression` type.
pub fn extraction_to_graph(
    egraph: &EGraph,
    extraction: &ExtractionResult,
    roots: &[ClassId],
) -> Option<StableGraph<GraphTerm, (), Directed>> {
    // display_graph(&extraction_to_petgraph(egraph, extraction), &[]);
    let mut g: StableGraph<GraphTerm, (), Directed> = StableGraph::new();

    fn emit<'a>(
        nid: &'a NodeId,
        egraph: &'a EGraph,
        extraction: &'a ExtractionResult,
        g: &mut StableGraph<GraphTerm, (), Directed>,
        seen: &mut HashMap<&'a NodeId, usize>,
    ) -> Option<NodeIndex> {
        let mut pick_child = |child| {
            let child_cid = egraph.nid_to_cid(child);
            let Some(mut child_nid) = extraction.choices.get(child_cid).copied() else {
                return None;
            };
            if seen
                .get(child_nid)
                .map(|s| *s > MAX_CYCLES)
                .unwrap_or_default()
            {
                // Pick another one we haven't seen more tha max cycles (THIS SHOULD BE PART OF EXTRACTION)
                let Some(c) = egraph.classes()[child_cid]
                    .nodes
                    .iter()
                    .find(|n| seen.get(*n).map(|s| *s <= MAX_CYCLES).unwrap_or(true))
                else {
                    return None;
                };
                child_nid = c;
            }
            *seen.entry(child_nid).or_default() += 1;
            let r = emit(child_nid, egraph, extraction, g, seen);
            *seen.get_mut(child_nid).unwrap() -= 1;
            r
        };
        let enode = &egraph.nodes[nid];
        match enode.op.as_str() {
            "GMEM" => Some(
                g.add_node(GraphTerm::GMEM {
                    label: Some(
                        egraph.nodes[&enode.children[0]]
                            .op
                            .replace("Boxed(\"", "")
                            .replace("\")", ""),
                    ),
                }),
            ),
            "SMEM" => Some(g.add_node(GraphTerm::SMEM)),
            "SMEMLoad" => Some(g.add_node(GraphTerm::SMEMLoad)),
            "SMEMRead" => Some(g.add_node(GraphTerm::SMEMRead)),

            // LoopIn  = (LoopIn <expr> <LoopType> <Math>)
            "LoopIn" => {
                let range =
                    convert_math(egraph.nid_to_cid(&enode.children[1]), egraph, extraction)?;
                let stride =
                    convert_math(egraph.nid_to_cid(&enode.children[2]), egraph, extraction)?;
                let child = pick_child(&enode.children[0])?;
                let n = g.add_node(GraphTerm::LoopIn {
                    range,
                    stride,
                    marker: "".to_string(),
                });
                g.add_edge(child, n, ());
                Some(n)
            }

            // LoopOut = same child layout as LoopIn
            "LoopOut" => {
                let range =
                    convert_math(egraph.nid_to_cid(&enode.children[1]), egraph, extraction)?;
                let stride =
                    convert_math(egraph.nid_to_cid(&enode.children[2]), egraph, extraction)?;

                let child = pick_child(&enode.children[0])?;
                let n = g.add_node(GraphTerm::LoopOut {
                    range,
                    stride,
                    marker: "".to_string(),
                });
                g.add_edge(child, n, ());
                Some(n)
            }

            "Add" | "Mul" | "Max" => {
                let a = pick_child(&enode.children[0])?;
                let b = pick_child(&enode.children[1])?;
                let n = g.add_node(match enode.op.as_str() {
                    "Add" => GraphTerm::Add,
                    "Mul" => GraphTerm::Mul,
                    "Max" => GraphTerm::Max,
                    _ => panic!(),
                });
                g.add_edge(a, n, ());
                g.add_edge(b, n, ());
                Some(n)
            }
            "Exp" | "Sin" | "Recip" | "Neg" => {
                let a = pick_child(&enode.children[0])?;
                let n = g.add_node(match enode.op.as_str() {
                    "Exp" => GraphTerm::Exp,
                    "Sin" => GraphTerm::Sin,
                    "Recip" => GraphTerm::Recip,
                    "Neg" => GraphTerm::Neg,
                    _ => panic!(),
                });
                g.add_edge(a, n, ());
                Some(n)
            }
            "Unary" | "Binary" | "SwapLoops" => {
                println!("expr {}", enode.op);
                return None;
            }
            _ => unreachable!("unsupported op {}", enode.op),
        }
    }

    for root_cid in roots {
        if let Some(root_nid) = extraction.choices.get(root_cid) {
            emit(root_nid, egraph, extraction, &mut g, &mut HashMap::new())?;
        }
    }
    Some(g)
}

fn convert_math(
    cid: &ClassId,
    egraph: &EGraph,
    extraction: &ExtractionResult,
) -> Option<Expression> {
    // Memoise by NodeId so we don’t re-convert shared sub-expressions.
    fn build<'a>(
        nid: &'a NodeId,
        egraph: &'a EGraph,
        extraction: &'a ExtractionResult,
        seen: &mut HashMap<&'a NodeId, usize>,
    ) -> Option<Expression> {
        let enode = &egraph.nodes[nid];
        let mut make_child = |child_cid: &ClassId| -> Option<Expression> {
            let Some(mut child_nid) = extraction.choices.get(child_cid).copied() else {
                return None;
            };
            if seen
                .get(child_nid)
                .map(|s| *s > MAX_CYCLES)
                .unwrap_or_default()
            {
                // Pick another one we haven't seen more tha max cycles (THIS SHOULD BE PART OF EXTRACTION)
                child_nid = egraph.classes()[child_cid]
                    .nodes
                    .iter()
                    .find(|n| seen.get(*n).map(|s| *s <= MAX_CYCLES).unwrap_or(true))
                    .unwrap();
            }
            *seen.entry(child_nid).or_default() += 1;
            let r = build(child_nid, egraph, extraction, seen);
            *seen.get_mut(child_nid).unwrap() -= 1;
            r
        };
        let term = match enode.op.as_str() {
            // ----------- literals & vars -----------
            op if op.starts_with("MNum:") => {
                let num: i64 = op["MNum:".len()..].parse().expect("invalid MNum literal");
                Expression::from(num as usize)
            }
            op if op.starts_with("MVar:") => {
                let name = op["MVar:".len()..].to_owned();
                Expression::from(name.chars().next().unwrap())
            }
            op if op.starts_with("Boxed(\"") => {
                let name = op.replace("Boxed(\"", "").replace("\")", "");
                // if name.len() == 1 {
                Expression::from(name.chars().next().unwrap())
                // } else {
                //     panic!("Variable name too long: {name}")
                // }
            }

            // ----------- unary ops -----------
            "MNeg" | "MRecip" => {
                let c0 = make_child(&egraph.nid_to_cid(&enode.children[0]))?;
                match enode.op.as_str() {
                    "MNeg" => c0 * -1,
                    "MRecip" => 1 / c0,
                    _ => unreachable!(),
                }
            }

            // ----------- binary ops -----------
            "MAdd" | "MSub" | "MMul" | "MDiv" | "MMod" | "MMin" | "MMax" | "MAnd" | "MOr"
            | "MGte" | "MLt" | "MFloorTo" => {
                let lhs = make_child(&egraph.nid_to_cid(&enode.children[0]))?;
                let rhs = make_child(&egraph.nid_to_cid(&enode.children[1]))?;
                match enode.op.as_str() {
                    "MAdd" => lhs + rhs,
                    "MSub" => lhs - rhs,
                    "MMul" => lhs * rhs,
                    "MDiv" => lhs / rhs,
                    "MMod" => lhs % rhs,
                    "MMin" => lhs.min(rhs),
                    "MMax" => lhs.max(rhs),
                    "MAnd" => lhs & rhs,
                    "MOr" => lhs | rhs,
                    "MFloorTo" => lhs / rhs * rhs, // NOT CORRECT, NEED FLOORTO IN EXPRESSIONS
                    _ => unreachable!(),
                }
            }

            // ----------- ternary -----------
            "MReplace" => {
                // println!("replace");
                return None;
            }

            // ----------- accumulator marker -----------
            "MAccum" => {
                let name = if enode.children.is_empty() {
                    "<acc>".to_owned()
                } else {
                    "<acc>".to_owned()
                };
                Expression::from(Term::Acc(name.chars().next().unwrap()))
            }
            "Loop" => make_child(&egraph.nid_to_cid(&enode.children[1]))?,
            "MNum" | "MVar" => make_child(&egraph.nid_to_cid(&enode.children[0]))?,
            _ => {
                if let Ok(n) = enode.op.parse::<usize>() {
                    Expression::from(n)
                } else {
                    panic!("unsupported Math op '{}'", enode.op)
                }
            }
        };

        Some(term)
    }

    if !extraction.choices.contains_key(cid) {
        return None;
    }
    let nid = &extraction.choices[cid];
    build(nid, egraph, extraction, &mut HashMap::new())
}
