use std::collections::HashMap;
use std::mem::size_of;

use crate::{
    codegen::{codegen, stitch_meta_graph_together},
    debug::display_graph,
    extract::{make_test_inputs, search},
    run::{assign_buffers, compile_kernels, run_graph},
    translate::{translate_graph_meta, InitData},
    utils::build_search_space,
    GPUArch, GraphTerm,
};
use itertools::Itertools;
use luminal::prelude::{
    petgraph::{visit::EdgeRef, Direction},
    *,
};
use metal_rs::{objc::rc::autoreleasepool, Buffer, Device, MTLResourceOptions};
use rand::{rng, Rng};
use rustc_hash::FxHashMap;

// confirm we generate 1 fused kernel and it runs successfully
#[test]
fn e2e_naive_matmul() {
    autoreleasepool(|| {
        let mut cx = Graph::new();
        let (m, k, n) = (512, 512, 512);
        let a_data = (0..(m * k)).map(|_| 1.0).collect_vec();
        let b_data = (0..(k * n)).map(|_| 2.0).collect_vec();
        let a = cx.named_tensor("A", (m, k)).set(a_data.clone());
        let mut b = cx.named_tensor("B", (k, n)).set(b_data.clone());
        b = b.transpose(1, 0);
        let c = a.matmul(b).retrieve();
        cx.set_dyn_dim('a', 3);
        cx.set_dyn_dim('b', 2);
        let (mut new_graph, mut old_to_new_mapping, mut accs) = translate_graph_meta(&cx);

        for graph_node in new_graph.node_indices().collect_vec() {
            let graph = new_graph.node_weight_mut(graph_node).unwrap();
            let search_space = build_search_space(graph, 3);
            let inputs = make_test_inputs(graph, &cx.dyn_map, &accs);
            let searched_graph = search(
                graph,
                3,
                &inputs,
                GPUArch::Metal(HashMap::default()),
                &cx.dyn_map,
            )
            .unwrap();
            let old_output = graph.externals(Direction::Outgoing).next().unwrap();
            let new_output = searched_graph
                .externals(Direction::Outgoing)
                .next()
                .unwrap();
            let old_inputs = graph
                .node_indices()
                .filter_map(|n| {
                    if let GraphTerm::GMEM { label } = graph.node_weight(n).unwrap() {
                        Some((n, label.clone()))
                    } else {
                        None
                    }
                })
                .collect::<HashMap<_, _>>();
            let new_inputs = searched_graph
                .node_indices()
                .filter_map(|n| {
                    if let GraphTerm::GMEM { label } = searched_graph.node_weight(n).unwrap() {
                        Some((label.clone(), n))
                    } else {
                        None
                    }
                })
                .collect::<HashMap<_, _>>();
            *graph = searched_graph;
            for edge in new_graph
                .edges_directed(graph_node, Direction::Outgoing)
                .map(|e| e.id())
                .collect_vec()
            {
                let (input, _) = new_graph.edge_weight_mut(edge).unwrap();
                *input = new_output;
            }
            for (_, (meta, v)) in &mut old_to_new_mapping {
                if *meta != graph_node {
                    continue;
                }
                if *v == old_output {
                    *v = new_output;
                }
                if let Some(gmem_label) = old_inputs.get(v) {
                    *v = new_inputs[gmem_label];
                }
            }
        }
        let outputs = vec![old_to_new_mapping[&c.id]];
        let (new_graph, meta_to_unified) = stitch_meta_graph_together(new_graph);
        let mut unified_map = FxHashMap::default();
        for (k, v) in old_to_new_mapping {
            if let Some(m) = meta_to_unified.get(&v) {
                unified_map.insert(k, *m);
            }
        }
        let (kernels, gmem_mapping) = codegen(
            new_graph.clone(),
            GPUArch::Metal(HashMap::default()),
            &cx.dyn_map,
        )
        .unwrap();

        assert!(kernels.node_count() == 3, "More than 1 kernel");

        let device = match Device::system_default() {
            Some(d) => d,
            None => {
                println!("❌ No Metal-enabled device found. Exiting.");
                std::process::exit(1);
            }
        };
        let mut inputs = FxHashMap::default();
        inputs.insert(
            gmem_mapping[&unified_map[&a.id]],
            (copy_metal_buffer(&a_data, &device), true),
        );
        inputs.insert(
            gmem_mapping[&unified_map[&b.id]],
            (copy_metal_buffer(&b_data, &device), true),
        );
        for (label, val) in &accs {
            match val {
                InitData::Expr(e) => {
                    let val = e.exec(&cx.dyn_map).unwrap();
                    if let Some(idx) = unified_map.get(label).and_then(|l| gmem_mapping.get(l)) {
                        inputs.insert(*idx, {
                            let v = vec![val as f32];
                            (copy_metal_buffer(&v, &device), true)
                        });
                    }
                }
                InitData::Data(d) => {
                    if let Some(idx) = unified_map.get(label).and_then(|l| gmem_mapping.get(l)) {
                        inputs.insert(*idx, (copy_metal_buffer(d, &device), true));
                    }
                }
            }
        }

        let compiled_kernels = compile_kernels(&kernels);
        let (int_buffers, int_buffer_map) = assign_buffers(&kernels);
        let (outputs, _) = run_graph(
            &new_graph,
            &mut inputs,
            &kernels,
            &cx.dyn_map,
            &compiled_kernels,
            &int_buffers,
            &int_buffer_map,
        );
        let result = &copy_metal_buffer_back(&outputs[0])[..10];
        for element in result {
            assert!(
                *element == 1024.0,
                "Logical Error All Values Should be 1024.0"
            );
        }
    });
}

pub fn copy_metal_buffer(v: &Vec<f32>, device: &Device) -> Buffer {
    let buf = device.new_buffer_with_data(
        v.as_ptr() as *const _,
        (v.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    buf
}

pub fn copy_metal_buffer_back(v: &Buffer) -> Vec<f32> {
    let mut data = vec![0f32; v.length() as usize / size_of::<f32>()];
    let ptr = v.contents() as *mut f32;
    for (i, d) in data.iter_mut().enumerate() {
        *d = unsafe { *ptr.add(i) };
    }
    data
}
