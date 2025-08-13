use std::{fs::File, io::Read};

use itertools::Itertools;
use luminal::{
    prelude::{
        NodeIndex,
        petgraph::{
            Direction,
            algo::toposort,
            prelude::StableGraph,
            visit::{EdgeRef, IntoEdgeReferences},
        },
    },
    shape::Expression,
};
use metal_rs::Device;
use rustc_hash::FxHashMap;

use crate::Kernel;

pub fn assign_buffers(
    graph: &StableGraph<Kernel, (usize, usize)>,
) -> (Vec<Expression>, FxHashMap<NodeIndex, Vec<usize>>) {
    // Count consumers only for producer outputs we manage (exclude "Inputs")
    let mut use_count: FxHashMap<(NodeIndex, usize), usize> = FxHashMap::default();
    for e in graph.edge_references() {
        let src = e.source();
        if graph[src].code != "Inputs" {
            let (src_out, _) = *e.weight();
            *use_count.entry((src, src_out)).or_default() += 1;
        }
    }

    let mut master = vec![]; // capacities by global buffer index
    let mut buf_map = FxHashMap::default(); // node -> output_idx -> buffer_idx
    let mut free_by_cap = FxHashMap::<Expression, Vec<usize>>::default(); // exact-size reuse

    for node in toposort(graph, None).unwrap() {
        let k = &graph[node];
        if k.code == "Inputs" {
            continue; // user-provided; ignore
        }

        // Allocate exact-size buffers for this node's outputs
        let mut outs = vec![];
        for &cap in &k.outputs {
            let buf_idx = if let Some(idx) = free_by_cap.get_mut(&cap).map(|l| l.pop()).flatten() {
                // reuse
                idx
            } else {
                // allocate new buffer
                master.push(cap);
                master.len() - 1
            };
            outs.push(buf_idx);
        }
        buf_map.insert(node, outs);

        // Free producer buffers whose last consumer just ran (exclude "Inputs")
        for e in graph.edges_directed(node, Direction::Incoming) {
            let src = e.source();
            if graph[src].code == "Inputs" {
                continue;
            }
            let (src_out_idx, _) = *e.weight();
            if let Some(c) = use_count.get_mut(&(src, src_out_idx)) {
                *c -= 1;
                if *c == 0 {
                    let buf_idx = buf_map[&src][src_out_idx];
                    free_by_cap
                        .entry(master[buf_idx])
                        .or_default()
                        .push(buf_idx);
                }
            }
        }
    }

    (master, buf_map)
}

#[cfg(feature = "cuda")]
pub fn compile_kernels(
    kernels: &StableGraph<Kernel, (usize, usize)>,
) -> FxHashMap<String, CudaFunction> {
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let mut compiled = FxHashMap::default();
    for kernel in kernels.node_weights() {
        if !compiled.contains_key(&kernel.code)
            && kernel.code != "Inputs"
            && kernel.code != "Outputs"
        {
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(
                &kernel.code,
                CompileOptions {
                    include_paths: vec!["/usr/include".into()],
                    options: vec![
                        "--gpu-architecture=compute_75".into(),
                        "--relocatable-device-code=false".into(),
                        "--std=c++14".into(),
                    ],
                    ..Default::default()
                },
            )
            .unwrap();
            let module = ctx.load_module(ptx).unwrap();
            let k = module.load_function("kernel_name").unwrap();
            compiled.insert(kernel.code.clone(), k);
        }
    }
    compiled
}

#[cfg(feature = "metal")]
pub fn compile_kernels(
    kernels: &StableGraph<Kernel, (usize, usize)>,
) -> FxHashMap<String, metal_rs::Function> {
    let device = Device::system_default().unwrap();
    let options = metal_rs::CompileOptions::new();
    options.set_fast_math_enabled(true);

    let mut compiled = FxHashMap::default();
    for kernel in kernels.node_weights() {
        if !compiled.contains_key(&kernel.code)
            && kernel.code != "Inputs"
            && kernel.code != "Outputs"
        {
            let lib = device
                .new_library_with_source(&kernel.code, &options)
                .unwrap();
            let f = lib.get_function("kernel_name", None).unwrap();
            compiled.insert(kernel.code.clone(), f);
        }
    }
    compiled
}

#[cfg(feature = "cuda")]
pub fn run_graph(
    inputs: &mut FxHashMap<usize, (CudaSlice<f32>, bool)>,
    kernels: &StableGraph<Kernel, (usize, usize)>,
    dyn_vars: &FxHashMap<char, usize>,
    compiled_kernels: &FxHashMap<String, CudaFunction>,
    intermediate_buffers: &Vec<Expression>,
    intermediate_buffer_map: &FxHashMap<NodeIndex, Vec<usize>>,
) -> (Vec<CudaSlice<f32>>, u128) {
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let start = std::time::Instant::now();

    // Allocate intermediate buffers
    let mut buffers = intermediate_buffers
        .iter()
        .map(|e| unsafe { stream.alloc(e.exec(dyn_vars).unwrap()).unwrap() })
        .collect_vec();
    let input_node = kernels
        .node_indices()
        .find(|n| kernels[*n].code == "Inputs")
        .unwrap();
    for node in toposort(kernels, None).unwrap() {
        let kernel = &kernels[node];
        if kernel.code == "Inputs" {
            // Inputs should already be in the buffer map
        } else if kernel.code == "Outputs" {
            // Run
            stream.synchronize().unwrap(); // There shouldn't be any other syncs from dispatch till here
            let outputs = kernels
                .edges_directed(node, Direction::Incoming)
                .map(|e| {
                    (
                        e.weight().1,
                        intermediate_buffer_map[&e.source()][e.weight().0],
                    )
                })
                .sorted_by_key(|(_, b)| *b)
                .rev()
                .map(|(a, b)| (a, buffers.remove(b)))
                .sorted_by_key(|(a, _)| *a)
                .map(|(_, a)| a)
                .collect_vec();
            return (outputs, start.elapsed().as_micros());
        } else if kernel.code.starts_with("Diff") {
            // Load file and diff numbers
            let diff_name = kernel.code.replace("Diff", "");
            let (input, input_index) = kernels
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|n| n.weight().1)
                .map(|n| (n.source(), n.weight().0))
                .next()
                .unwrap();
            let buffer = &buffers[intermediate_buffer_map[&input][input_index]];
            let data: Vec<f32> = stream.memcpy_dtov(buffer).unwrap();
            let mut file = File::open(format!("{diff_name}.bin")).unwrap();
            let mut file_buffer = Vec::new();
            file.read_to_end(&mut file_buffer).unwrap();
            assert_eq!(file_buffer.len() % std::mem::size_of::<f32>(), 0);

            let num_floats = file_buffer.len() / std::mem::size_of::<f32>();
            let floats: Vec<f32> = unsafe {
                let ptr = file_buffer.as_ptr() as *const f32;
                Vec::from_raw_parts(ptr as *mut f32, num_floats, num_floats)
            };
            let mut matched = true;
            println!("Diff {} | {}", data.len(), floats.len());
            for (ind, (i, j)) in data.iter().zip(floats).enumerate() {
                if (i - j).abs() > 1e-5 {
                    matched = false;
                    println!("Diff {diff_name} failed: curr: {i} != file: {j}, index {ind}");
                    break;
                }
            }
            std::mem::forget(file_buffer);
            if matched {
                println!("DIFF {diff_name} MATCHED");
            }
            let dest_buffer = &mut buffers[intermediate_buffer_map[&node][0]];
            stream.memcpy_htod(&data, dest_buffer).unwrap();
        } else {
            let mut builder = stream.launch_builder(&compiled_kernels[&kernel.code]);

            // set inputs
            for (input, input_index) in kernels
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|n| n.weight().1)
                .map(|n| (n.source(), n.weight().0))
            {
                if input == input_node {
                    builder.arg(&inputs[&input_index].0);
                } else {
                    builder.arg(&buffers[intermediate_buffer_map[&input][input_index]]);
                }
            }
            // set output
            let mut output_views = (0..kernel.outputs.len())
                .map(|o| buffers[intermediate_buffer_map[&node][o]].as_view_mut())
                .collect_vec();
            for o in &mut output_views {
                builder.arg(o);
            }
            // set dynamic dimensions
            for (_, v) in dyn_vars.iter().sorted_by_key(|(k, _)| **k) {
                builder.arg(v);
            }

            // Set dispatch
            let grid = (
                kernel.grid.0.exec(dyn_vars).unwrap() as u32,
                kernel.grid.1.exec(dyn_vars).unwrap() as u32,
                kernel.grid.2.exec(dyn_vars).unwrap() as u32,
            );
            let tb = (
                kernel.threadblock.0.exec(dyn_vars).unwrap() as u32,
                kernel.threadblock.1.exec(dyn_vars).unwrap() as u32,
                kernel.threadblock.2.exec(dyn_vars).unwrap() as u32,
            );
            assert!(
                tb.0 * tb.1 * tb.2 <= 1024,
                "threadblock is too big: {tb:?} > 1024"
            );
            assert!(grid.1 <= 65535, "grid.y > 65535");
            assert!(grid.2 <= 65535, "grid.z > 65535");
            assert!(grid.0 <= 2147483647, "grid.x > 2147483647");
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: grid,
                    block_dim: tb,
                    shared_mem_bytes: kernel.smem.exec(dyn_vars).unwrap() as u32,
                })
            }
            .unwrap();
        }
    }
    panic!("No output kernel detected in graph!");
}

#[cfg(feature = "metal")]
pub fn run_graph(
    inputs: &mut FxHashMap<usize, (metal_rs::Buffer, bool)>,
    kernels: &StableGraph<Kernel, (usize, usize)>,
    dyn_vars: &FxHashMap<char, usize>,
    compiled_kernels: &FxHashMap<String, metal_rs::Function>,
    intermediate_buffers: &Vec<Expression>,
    intermediate_buffer_map: &FxHashMap<NodeIndex, Vec<usize>>,
) -> (Vec<metal_rs::Buffer>, u128) {
    use metal_rs::objc::rc::autoreleasepool;

    autoreleasepool(|| {
        use metal_rs::MTLResourceOptions;

        // println!("deep down in the mines");

        let device = metal_rs::Device::system_default().unwrap();
        let queue = device.new_command_queue();
        let command_buffer = queue.new_command_buffer();
        let start = std::time::Instant::now();

        // Allocate intermediate buffers
        let mut buffers = intermediate_buffers
            .iter()
            .map(|e| {
                device.new_buffer(
                    (e.exec(dyn_vars).unwrap() * size_of::<f32>()) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .collect_vec();
        let input_node = kernels
            .node_indices()
            .find(|n| kernels[*n].code == "Inputs")
            .unwrap();
        for node in toposort(kernels, None).unwrap() {
            let kernel = &kernels[node];
            // println!("Our wonderful kernel: {:?}", kernel);
            if kernel.code == "Inputs" {
                // Inputs should already be in the buffer map
            } else if kernel.code == "Outputs" {
                // Run
                command_buffer.commit();
                command_buffer.wait_until_completed();
                let outputs = kernels
                    .edges_directed(node, Direction::Incoming)
                    .map(|e| {
                        (
                            e.weight().1,
                            intermediate_buffer_map[&e.source()][e.weight().0],
                        )
                    })
                    .sorted_by_key(|(_, b)| *b)
                    .rev()
                    .map(|(a, b)| (a, buffers.remove(b)))
                    .sorted_by_key(|(a, _)| *a)
                    .map(|(_, a)| a)
                    .collect_vec();
                return (outputs, start.elapsed().as_micros());
            } else if kernel.code.starts_with("Diff") {
                // Load file and diff numbers
                let diff_name = kernel.code.replace("Diff", "");
                let (input, input_index) = kernels
                    .edges_directed(node, Direction::Incoming)
                    .sorted_by_key(|n| n.weight().1)
                    .map(|n| (n.source(), n.weight().0))
                    .next()
                    .unwrap();
                let buffer = &buffers[intermediate_buffer_map[&input][input_index]];
                let mut data = vec![0_f32; buffer.length() as usize / size_of::<f32>()];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        buffer.contents() as *const _,
                        &mut data,
                        data.len(),
                    );
                }
                let mut file = File::open(format!("{diff_name}.bin")).unwrap();
                let mut file_buffer = Vec::new();
                file.read_to_end(&mut file_buffer).unwrap();
                assert_eq!(file_buffer.len() % std::mem::size_of::<f32>(), 0);

                let num_floats = file_buffer.len() / std::mem::size_of::<f32>();
                let floats: Vec<f32> = unsafe {
                    let ptr = file_buffer.as_ptr() as *const f32;
                    Vec::from_raw_parts(ptr as *mut f32, num_floats, num_floats)
                };
                let mut matched = true;
                println!("Diff {} | {}", data.len(), floats.len());
                for (ind, (i, j)) in data.iter().zip(floats).enumerate() {
                    if (i - j).abs() > 1e-5 {
                        matched = false;
                        println!("Diff {diff_name} failed: curr: {i} != file: {j}, index {ind}");
                        break;
                    }
                }
                std::mem::forget(file_buffer);
                if matched {
                    println!("DIFF {diff_name} MATCHED");
                }
                let dest_buffer = &mut buffers[intermediate_buffer_map[&node][0]];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        &data,
                        dest_buffer.contents() as *mut _,
                        data.len(),
                    );
                }
            } else {
                use metal_rs::{ComputePassDescriptor, ComputePipelineDescriptor, MTLSize};
                let encoder = command_buffer
                    .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
                let pipeline_state_descriptor = ComputePipelineDescriptor::new();
                pipeline_state_descriptor
                    .set_compute_function(Some(&compiled_kernels[&kernel.code]));
                let pipeline = device
                    .new_compute_pipeline_state_with_function(
                        pipeline_state_descriptor.compute_function().unwrap(),
                    )
                    .unwrap();
                encoder.set_compute_pipeline_state(&pipeline);

                // set inputs
                let mut buffer_count = 0;
                for (input, input_index) in kernels
                    .edges_directed(node, Direction::Incoming)
                    .sorted_by_key(|n| n.weight().1)
                    .map(|n| (n.source(), n.weight().0))
                {
                    println!("({:?} , {:?})", input, input_index);
                    if input == input_node {
                        println!("index: {:?}", &input_index);
                        println!("inputs: {:?}", inputs.keys());
                        println!("inuts: {:?}", &inputs[&input_index].0);
                        encoder.set_buffer(buffer_count, Some(&inputs[&input_index].0), 0);
                        println!("did not panic");
                    } else {
                        encoder.set_buffer(
                            buffer_count,
                            Some(&buffers[intermediate_buffer_map[&input][input_index]]),
                            0,
                        );
                    }
                    buffer_count += 1;
                }
                // set output
                for o in 0..kernel.outputs.len() {
                    encoder.set_buffer(
                        buffer_count,
                        Some(&buffers[intermediate_buffer_map[&node][o]]),
                        0,
                    );
                    buffer_count += 1;
                }
                // set dynamic dimensions
                for (_, v) in dyn_vars.iter().sorted_by_key(|(k, _)| **k) {
                    let val: u64 = *v as u64;
                    let buf = device.new_buffer_with_data(
                        &val as *const _ as *const _,
                        std::mem::size_of::<u64>() as u64,
                        MTLResourceOptions::StorageModeShared,
                    );
                    encoder.set_buffer(buffer_count, Some(&buf), 0);
                    buffer_count += 1;
                }

                // Set dispatch
                let grid = (
                    kernel.grid.0.exec(dyn_vars).unwrap() as u64,
                    kernel.grid.1.exec(dyn_vars).unwrap() as u64,
                    kernel.grid.2.exec(dyn_vars).unwrap() as u64,
                );
                let tb = (
                    kernel.threadblock.0.exec(dyn_vars).unwrap() as u64,
                    kernel.threadblock.1.exec(dyn_vars).unwrap() as u64,
                    kernel.threadblock.2.exec(dyn_vars).unwrap() as u64,
                );
                assert!(
                    tb.0 * tb.1 * tb.2 <= 1024,
                    "threadblock is too big: {tb:?} > 1024"
                );
                assert!(grid.1 <= 65535, "grid.y > 65535");
                assert!(grid.2 <= 65535, "grid.z > 65535");
                assert!(grid.0 <= 2147483647, "grid.x > 2147483647");
                encoder.dispatch_thread_groups(
                    MTLSize::new(grid.0, grid.1, grid.2),
                    MTLSize::new(tb.0, tb.1, tb.2),
                );
                encoder.end_encoding();
            }
        }
        panic!("No output kernel detected in graph!");
    })
}
