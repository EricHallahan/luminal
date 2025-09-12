#[cfg(feature = "metal")]
use crate::{Buffer, Device, Function};
#[cfg(feature = "cuda")]
use cudarc::{driver::*, nvrtc::CompileOptions};
use itertools::Itertools;
#[cfg(feature = "cuda")]
use std::fs::OpenOptions;
#[cfg(feature = "cuda")]
use std::io::Write;

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
#[cfg(feature = "metal")]
use objc2_metal::{MTLBuffer, MTLDevice};
#[cfg(feature = "opencl")]
use opencl3::{
    command_queue::CommandQueue,
    context::Context,
    kernel::{ExecuteKernel, Kernel as OclKernel},
    memory::{Buffer as OclBuffer, CL_MEM_READ_WRITE},
    program::Program,
};
use rustc_hash::FxHashMap;

#[cfg(any(feature = "cuda", feature = "metal"))]
use std::{fs::File, io::Read};

#[cfg(feature = "metal")]
use std::{ffi::c_void, ptr::NonNull};

use crate::{GraphTerm, Kernel};

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

    // Open (or create) the log file, appending logs to it
    let log_path = "kernel_log.txt";
    let mut log_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true) // overwrite on each run
        .open(log_path)
        .expect("Failed to open kernel log file");

    for kernel in kernels.node_weights() {
        if !compiled.contains_key(&kernel.code)
            && kernel.code != "Inputs"
            && kernel.code != "Outputs"
        {
            writeln!(log_file, "Compiling kernel:\n{}\n", kernel.code)
                .expect("Failed to write to kernel log file");

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
) -> FxHashMap<String, Function> {
    use objc2_metal::MTLCreateSystemDefaultDevice;

    let device = MTLCreateSystemDefaultDevice().unwrap();
    let mut compiled = FxHashMap::default();
    for kernel in kernels.node_weights() {
        if !compiled.contains_key(&kernel.code)
            && kernel.code != "Inputs"
            && kernel.code != "Outputs"
        {
            use objc2_foundation::{NSString, ns_string};
            use objc2_metal::{MTLDevice, MTLLibrary};
            let lib = device
                .newLibraryWithSource_options_error(&NSString::from_str(&kernel.code), None)
                .unwrap();
            let f = lib.newFunctionWithName(ns_string!("kernel_name")).unwrap();
            compiled.insert(kernel.code.clone(), f);
        }
    }
    compiled
}

#[cfg(feature = "opencl")]
pub fn compile_kernels(
    kernels: &StableGraph<Kernel, (usize, usize)>,
    context: &Context,
) -> FxHashMap<String, OclKernel> {
    let mut compiled = FxHashMap::default();
    for kernel in kernels.node_weights() {
        if !compiled.contains_key(&kernel.code)
            && kernel.code != "Inputs"
            && kernel.code != "Outputs"
        {
            let program =
                Program::create_and_build_from_source(context, &kernel.code, "-cl-std=CL1.2")
                    .unwrap();
            let ocl_kernel = OclKernel::create(&program, "kernel_name").unwrap();
            compiled.insert(kernel.code.clone(), ocl_kernel);
        }
    }
    compiled
}

#[cfg(feature = "cuda")]
pub fn run_graph(
    _graph: &StableGraph<GraphTerm, ()>,
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
        .map(|e| unsafe {
            stream
                .alloc(e.exec(dyn_vars).unwrap() * size_of::<f32>())
                .unwrap()
        })
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
            let dest_buffer: &mut CudaSlice<f32> = &mut buffers[intermediate_buffer_map[&node][0]];
            stream.memcpy_htod(&data, dest_buffer).unwrap();
        } else {
            let mut builder = stream.launch_builder(&compiled_kernels[&kernel.code]);
            // println!("Code to run: {}", kernel.code);

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
    graph: &StableGraph<GraphTerm, ()>,
    inputs: &mut FxHashMap<usize, (Buffer, bool)>,
    kernels: &StableGraph<Kernel, (usize, usize)>,
    dyn_vars: &FxHashMap<char, usize>,
    compiled_kernels: &FxHashMap<String, Function>,
    intermediate_buffers: &Vec<Expression>,
    intermediate_buffer_map: &FxHashMap<NodeIndex, Vec<usize>>,
) -> (Vec<Buffer>, u128) {
    objc2::rc::autoreleasepool(|_| {
        use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice};

        let device = MTLCreateSystemDefaultDevice().unwrap();
        let queue = device.newCommandQueue().expect("No command queue");
        let command_buffer = queue.commandBuffer().unwrap();
        let start = std::time::Instant::now();

        // Allocate intermediate buffers
        let mut buffers = intermediate_buffers
            .iter()
            .map(|e| {
                use objc2_metal::MTLResourceOptions;

                device
                    .newBufferWithLength_options(
                        (e.exec(dyn_vars).unwrap() * size_of::<f32>()) as u64,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .unwrap()
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
                use objc2_metal::MTLCommandBuffer;
                command_buffer.commit();
                unsafe {
                    command_buffer.waitUntilCompleted();
                }
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

                use objc2_metal::MTLBuffer;
                let diff_name = kernel.code.replace("Diff", "");
                let (input, input_index) = kernels
                    .edges_directed(node, Direction::Incoming)
                    .sorted_by_key(|n| n.weight().1)
                    .map(|n| (n.source(), n.weight().0))
                    .next()
                    .unwrap();
                let buffer = &buffers[intermediate_buffer_map[&input][input_index]];
                let mut data = vec![0_f32; buffer.length() as usize / std::mem::size_of::<f32>()];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        buffer.contents().as_ptr() as *const f32,
                        data.as_mut_ptr(),
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
                        data.as_ptr(),
                        dest_buffer.contents().as_ptr() as *mut f32,
                        data.len(),
                    );
                }
            } else {
                use objc2_metal::{
                    MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize,
                };
                let encoder = command_buffer.computeCommandEncoder().unwrap();
                let Ok(c) = device
                    .newComputePipelineStateWithFunction_error(&compiled_kernels[&kernel.code])
                else {
                    println!("failed to compile {}", kernel.code);
                    crate::debug::display_graph(graph);
                    panic!();
                };
                encoder.setComputePipelineState(&c);

                // set inputs
                let mut buffer_count = 0;

                for (input, input_index) in kernels
                    .edges_directed(node, Direction::Incoming)
                    .sorted_by_key(|n| n.weight().1)
                    .map(|n| (n.source(), n.weight().0))
                {
                    if input == input_node {
                        unsafe {
                            encoder.setBuffer_offset_atIndex(
                                Some(&inputs[&input_index].0),
                                0,
                                buffer_count,
                            );
                        }
                    } else {
                        unsafe {
                            encoder.setBuffer_offset_atIndex(
                                Some(&buffers[intermediate_buffer_map[&input][input_index]]),
                                0,
                                buffer_count,
                            );
                        }
                    }
                    buffer_count += 1;
                }
                // set output
                for o in 0..kernel.outputs.len() {
                    unsafe {
                        encoder.setBuffer_offset_atIndex(
                            Some(&buffers[intermediate_buffer_map[&node][o]]),
                            0,
                            buffer_count,
                        );
                    }
                    buffer_count += 1;
                }
                // set dynamic dimensions
                for (_, v) in dyn_vars.iter().sorted_by_key(|(k, _)| **k) {
                    let val: u64 = *v as u64;
                    let buf = unsafe {
                        use objc2_metal::MTLResourceOptions;

                        device
                            .newBufferWithBytes_length_options(
                                NonNull::new(&val as *const _ as *mut c_void).unwrap(),
                                std::mem::size_of::<u64>(),
                                MTLResourceOptions::StorageModeShared,
                            )
                            .unwrap()
                    };
                    unsafe { encoder.setBuffer_offset_atIndex(Some(&buf), 0, buffer_count) };
                    buffer_count += 1;
                }

                // Set dispatch
                let grid = (
                    kernel.grid.0.exec(dyn_vars).unwrap(),
                    kernel.grid.1.exec(dyn_vars).unwrap(),
                    kernel.grid.2.exec(dyn_vars).unwrap(),
                );
                let tb = (
                    kernel.threadblock.0.exec(dyn_vars).unwrap(),
                    kernel.threadblock.1.exec(dyn_vars).unwrap(),
                    kernel.threadblock.2.exec(dyn_vars).unwrap(),
                );
                assert!(
                    tb.0 * tb.1 * tb.2 <= 1024,
                    "threadblock is too big: {tb:?} > 1024"
                );
                assert!(grid.1 <= 65535, "grid.y > 65535");
                assert!(grid.2 <= 65535, "grid.z > 65535");
                assert!(grid.0 <= 2147483647, "grid.x > 2147483647");
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: grid.0,
                        height: grid.1,
                        depth: grid.2,
                    },
                    MTLSize {
                        width: tb.0,
                        height: tb.1,
                        depth: tb.2,
                    },
                );
                encoder.endEncoding();
            }
        }
        panic!("No output kernel detected in graph!");
    })
}

#[cfg(feature = "opencl")]
pub fn run_graph(
    _graph: &StableGraph<GraphTerm, ()>,
    inputs: &mut FxHashMap<usize, (OclBuffer<f32>, bool)>,
    kernels: &StableGraph<Kernel, (usize, usize)>,
    dyn_vars: &FxHashMap<char, usize>,
    compiled_kernels: &FxHashMap<String, OclKernel>,
    intermediate_buffers: &Vec<Expression>,
    intermediate_buffer_map: &FxHashMap<NodeIndex, Vec<usize>>,
    context: &Context,
    queue: &CommandQueue,
) -> (Vec<OclBuffer<f32>>, u128) {
    let start = std::time::Instant::now();

    // Allocate intermediate buffers
    let mut buffers = intermediate_buffers
        .iter()
        .map(|e| {
            let size = e.exec(dyn_vars).unwrap();
            unsafe {
                OclBuffer::<f32>::create(context, CL_MEM_READ_WRITE, size, std::ptr::null_mut())
            }
            .unwrap()
        })
        .collect_vec();
    let input_node = kernels
        .node_indices()
        .find(|n| kernels[*n].code == "Inputs")
        .unwrap();

    for node in toposort(kernels, None).unwrap() {
        let kernel = &kernels[node];
        if kernel.code == "Inputs" {
            // Inputs provided
        } else if kernel.code == "Outputs" {
            queue.finish().unwrap();
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
            // Not implemented for OpenCL yet, but this should be fine
        } else {
            let kernel_fn = &compiled_kernels[&kernel.code];
            let mut exec_kernel = ExecuteKernel::new(kernel_fn);

            // set inputs
            for (input, input_index) in kernels
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|n| n.weight().1)
                .map(|n| (n.source(), n.weight().0))
            {
                let buffer: &OclBuffer<f32> = if input == input_node {
                    &inputs[&input_index].0
                } else {
                    &buffers[intermediate_buffer_map[&input][input_index]]
                };
                unsafe { exec_kernel.set_arg(buffer) };
            }

            // set output
            for o in 0..kernel.outputs.len() {
                let buffer = &buffers[intermediate_buffer_map[&node][o]];
                unsafe { exec_kernel.set_arg(buffer) };
            }

            // set dynamic dimensions
            for (_, v) in dyn_vars.iter().sorted_by_key(|(k, _)| **k) {
                let val: i32 = *v as i32;
                unsafe { exec_kernel.set_arg(&val) };
            }

            // set shared mem
            if kernel.code.contains("__local float* sm") {
                unsafe {
                    exec_kernel.set_arg_local_buffer(
                        kernel.smem.exec(dyn_vars).unwrap() * std::mem::size_of::<f32>(),
                    );
                }
            }

            // Set dispatch
            let grid = [
                kernel.grid.0.exec(dyn_vars).unwrap(),
                kernel.grid.1.exec(dyn_vars).unwrap(),
                kernel.grid.2.exec(dyn_vars).unwrap(),
            ];
            let tb = [
                kernel.threadblock.0.exec(dyn_vars).unwrap(),
                kernel.threadblock.1.exec(dyn_vars).unwrap(),
                kernel.threadblock.2.exec(dyn_vars).unwrap(),
            ];
            let global_work_size = [grid[0] * tb[0], grid[1] * tb[1], grid[2] * tb[2]];

            unsafe {
                exec_kernel
                    .set_global_work_sizes(&global_work_size)
                    .set_local_work_sizes(&tb)
                    .enqueue_nd_range(queue)
                    .unwrap();
            }
        }
    }
    panic!("No output kernel detected in graph!");
}

#[cfg(feature = "metal")]
pub fn copy_metal_buffer(v: &Vec<f32>, device: &Device) -> Buffer {
    use std::{ffi::c_void, ptr::NonNull};
    let buf = unsafe {
        device
            .newBufferWithBytes_length_options(
                NonNull::new(v.as_ptr() as *mut c_void).unwrap(),
                (v.len() * std::mem::size_of::<f32>()) as u64,
                objc2_metal::MTLResourceOptions::StorageModeShared,
            )
            .unwrap()
    };
    buf
}

#[cfg(feature = "metal")]
pub fn copy_metal_buffer_back(v: &Buffer) -> Vec<f32> {
    let mut data = vec![0f32; v.length() as usize / size_of::<f32>()];
    let ptr = v.contents().as_ptr() as *mut f32;
    for (i, d) in data.iter_mut().enumerate() {
        *d = unsafe { *ptr.add(i) };
    }
    data
}
