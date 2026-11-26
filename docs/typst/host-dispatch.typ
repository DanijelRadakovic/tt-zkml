#set page(paper: "a4", margin: (x: 1.7cm, y: 1.8cm), numbering: "1")
#set text(size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.")
#show raw: set text(size: 8.5pt)
#show raw.where(block: true): it => block(fill: luma(247), stroke: 0.4pt + luma(200), inset: 6pt, radius: 3pt, width: 100%, it)

#let note(t) = block(fill: rgb("#fff7e0"), stroke: 0.5pt + rgb("#d9b44a"), inset: 7pt, radius: 3pt, width: 100%, t)
#let warn(t) = block(fill: rgb("#fdecec"), stroke: 0.5pt + rgb("#d98a8a"), inset: 7pt, radius: 3pt, width: 100%, t)
#let good(t) = block(fill: rgb("#eef7ee"), stroke: 0.5pt + rgb("#8ab98a"), inset: 7pt, radius: 3pt, width: 100%, t)
#let tbl(cols, ..args) = table(columns: cols, inset: 4pt, stroke: 0.4pt + luma(170), ..args)
#let bar(w, fill, label) = box(width: w, height: 14pt, fill: fill, stroke: 0.4pt, align(center + horizon, text(size: 7pt, label)))

#align(center)[
  #text(size: 18pt, weight: "bold")[Host dispatch: what the host must finish before the device can start] \
  #v(2pt)
  #text(size: 11pt)[Why one fused `execute_network` call can be slower in wall time than a loop of `execute_layer` calls]
]
#v(6pt)

This note explains a problem we measured in `execute_network`: on the chip the fused network is *faster* than running the layers one by one, but in wall-clock time it can be *slower*. The reason is not on the device. It is the work the host does inside one call, in particular allocating the output tensors, and the fact that this work must be finished before the device gets the program. Everything below is from this project's code and measurements (`benchmarks/nn_network`). The first sections describe the problem as we found it; the last two show the fix (packed outputs) and the numbers after it.

#outline(indent: 1.2em, depth: 2)
#pagebreak()

= What happens in one device-operation call

A call such as `zkml::prim::execute_layer(x, w)` is host code that calls `ttnn::device_operation::launch<Op>`. In order, on the host thread:

#tbl((auto, 1fr),
  [*1. Validate*], [Check that the inputs are device tensors and allocated.],
  [*2. Create the output tensors*], [`Op::create_output_tensors`: for every output, `ttnn::create_device_tensor(spec, device)`. This asks the device memory allocator for a DRAM buffer and returns a `Tensor` whose buffer has an *address*. No data is written; the buffer is only reserved.],
  [*3. Place the outputs*], [Bookkeeping of where the outputs live on the mesh.],
  [*4. Program-cache lookup*], [Hash of the attributes and tensor arguments, and an exact key. On a hit the compiled program is reused.],
  [*5. Override the runtime arguments*], [Write the *addresses* of the inputs and the outputs into the program's runtime-argument buffers (see the next section).],
  [*6. Enqueue*], [Push the program (a command) to the device command queue and return. The call returns here; the device runs the program on its own.],
)

Steps 1 to 6 run one after the other on one host thread. The device does nothing for this call until step 6 has pushed the command. So *the host does not wait for the device; the device waits for the host*, for as long as steps 1 to 5 take.

= Why the output tensors must exist first: addresses as arguments

A kernel does not know where a tensor is. A tensor in DRAM is a buffer at some *base address* chosen by the allocator when the tensor is created. The kernel gets that address as a runtime argument and builds a `TensorAccessor` from it, which turns a page index into a bank and an offset:

```cpp
// writer kernel (device side)
const uint32_t acc_addr = get_arg_val<uint32_t>(3 + 4 * I);      // base address, written by the host
const auto acc = TensorAccessor(acc_args, acc_addr);              // page_id -> (DRAM bank, offset)
noc.async_write(cb_acc, acc, tile_bytes, {}, {.page_id = tile});  // write one tile of the output
```

So the chain for every output tensor is: *allocate the buffer* (the address now exists) #sym.arrow.r *write the address into the runtime args of the kernel that writes it* #sym.arrow.r *enqueue the program*. The runtime arguments are stored in the kernel-config region of the core's L1, next to the kernel code, and the host sends them with the launch command. If an output is not allocated, there is no address to put in the arguments, so the command cannot be built. That is why allocation cannot be postponed or overlapped with the device running this same program.

#note[The allocation does *not* transfer data and the device does *not* wait for "all tensors to be stored". The output buffers are empty until the kernel writes them. What the device waits for is only the host finishing steps 1 to 6, which includes the allocation of every output.]

On a program-cache *hit* the program is already compiled and its argument buffers already exist; only the addresses change from call to call (the tensors are new), so step 5 just overwrites them. This is `override_runtime_arguments` in our factories.

= Two examples

== Example 1: one output tensor

An operation with a single output (an illustrative sketch, not a copy of one of our operations): one allocation, one address in the writer arguments.

```cpp
// output allocation: one tensor
Op::tensor_return_value_t Op::create_output_tensors(const operation_attributes_t& a, const tensor_args_t& t) {
    return ttnn::create_device_tensor(compute_output_specs(a, t), t.x.device());   // 1 allocation
}

// program factory: the address goes into the runtime args of the writer
SetRuntimeArgs(program, writer_id, core, {out.buffer()->address(), num_tiles});

// on a cache hit: only the address changes
void Op::SingleCore::override_runtime_arguments(cached_program_t& cp, const operation_attributes_t&,
                                                const tensor_args_t&, tensor_return_value_t& out) {
    GetRuntimeArgs(cp.program, cp.shared_variables.writer_id, CoreCoord{0, 0})[0] = out.buffer()->address();
}
```

Host cost of the call: one allocation (about 3.6 us, see the table below) plus the fixed part of a call.

== Example 2: `execute_network` as first written, many outputs

`execute_network` must return everything the proof needs: the output, the input scaler, and for every layer the accumulator and two scalers: `2 + 3L` tensors (20 for six layers). Each one is a separate allocation and each address is a separate runtime argument:

```cpp
// output allocation: 2 + 3L tensors, one create_device_tensor each (execute_network_device_operation.cpp)
for (const auto& spec : compute_output_specs(attributes, tensor_args)) {
    outputs.push_back(ttnn::create_device_tensor(spec, device));
}

// program factory: three output addresses per layer in the writer arguments
std::vector<uint32_t> writer_args = {out_addr, scaler_x_addr, out_tiles};
for (uint32_t l = 0; l < L; l++) {
    writer_args.push_back(outputs[acc_index(l)].buffer()->address());
    writer_args.push_back(outputs[scaler_w_index(L, l)].buffer()->address());
    writer_args.push_back(outputs[scaler_out_index(L, l)].buffer()->address());
    writer_args.push_back(plan.out_tiles(l));
}
SetRuntimeArgs(program, writer_id, core, writer_args);

// on a cache hit: all of the addresses are rewritten
for (uint32_t l = 0; l < L; l++) {
    writer_args[3 + 4 * l] = outputs[acc_index(l)].buffer()->address();
    writer_args[4 + 4 * l] = outputs[scaler_w_index(L, l)].buffer()->address();
    writer_args[5 + 4 * l] = outputs[scaler_out_index(L, l)].buffer()->address();
}
```

The device work per layer is the same as in `execute_layer`; the difference is that the host does `2 + 3L` allocations and `3L + 2` address writes before the single launch. (Section 7 shows what replaced this.)

= Timeline: why the loop hides the host cost and the fused call does not

#let h = rgb("#f4c7a1")
#let d = rgb("#a9c8e8")
#let w = 10.5cm
#align(center)[
  #tbl((auto, 1fr),
    [*Loop of 4 `execute_layer`*],
    [#stack(dir: ltr, bar(1.1cm, h, "host 1"), bar(3cm, d, "device: layer 1"), bar(3cm, d, "layer 2"), bar(3cm, d, "layer 3"), bar(3cm, d, "layer 4")) \
     #stack(dir: ltr, bar(1.1cm, white, ""), bar(2.6cm, h, "host 2, 3, 4: enqueued while layer 1 runs"), bar(1.1cm, white, ""))],
    [*One `execute_network`*],
    [#stack(dir: ltr, bar(4.4cm, h, "host: 14 allocations, args, enqueue"), bar(9.5cm, d, "device: all 4 layers"))],
  )
]
The commands are asynchronous: after `execute_layer` has pushed layer 1, the host immediately prepares layer 2 while the device runs layer 1, and so on. Only the first host step is in front of the device; the rest is hidden behind device time. The fused call is one command, so all of its host work (the allocations of all outputs) is in front of the kernel. In the figure the bars are schematic; the measured numbers follow.

= Measurements before the fix

Setup: Blackhole p100a, one core; `M = 128`, width 256 (`256 x 256` layers), bf16, program cached; `benchmarks/nn_network`.

*On the device* (profiler zones in the compute kernel), 4 layers:

#tbl((auto, auto, auto),
  [], [*fused `execute_network`*], [*4 #sym.times `execute_layer`*],
  [kernel time], [180 us], [203 us (198 + 4 between programs)],
  [quantize x], [7.7 us], [30.0 us (repeated every layer)],
  [quantize w / matmul / requant], [52 / 65 / 52 us], [51 / 64 / 51 us],
)

*Wall time of the whole network*: fused 240 us, loop 229 us. So the fused network is 11% faster on the chip and still loses in wall time.

*Host time of one call*, by piece (`zkml_nn_network_bench hostprof`, median of 100, in us):

#tbl((auto, auto, auto, auto, auto, auto),
  [*call*], [*outputs*], [*create outputs*], [*hash + key*], [*rest*], [*total*],
  [`execute_layer`], [5], [17.4], [0.5], [6.3], [24.2],
  [`execute_network` L=1], [5], [17.9], [0.5], [6.5], [25.0],
  [`execute_network` L=4], [14], [53.6], [1.0], [7.5], [62.2],
  [`execute_network` L=6], [20], [76.4], [1.4], [10.7], [88.5],
)

"Rest" is placement, `override_runtime_arguments` and the enqueue. Validation is below 0.1 us. Reading the table:

- Creating the outputs costs about *3.6 us per tensor* and is about 85% of a call. It grows linearly with `2 + 3L`.
- The program hash, the cache key and the validation are each about 1 us or less, so the program cache is not the problem.
- A single `execute_layer` call (24 us, five outputs) is cheaper than one fused call at L=4 (62 us), but the loop's four calls overlap with the device and the fused call does not.

= The fix: packed outputs, and an address that is a base plus an offset

The intermediate results (`acc_l` and the scalers) are needed for the trace, so they cannot be dropped. What can change is how many tensors they are spread over. The fix is to allocate *three* output tensors however many layers there are, and to compute the position of every layer's data inside them:

#tbl((auto, 1fr),
  [`out`], [UINT8 `[M, N_last]`, the quantized result of the last layer.],
  [`scalers`], [FLOAT32 `[32 * S, 32]`, S tiles: tile 0 is `scaler_x`, then (when the weights are quantized in the kernel) the weight scalers of every layer, then the requantization scaler of every layer. Scaler `i` is page `i`.],
  [`acc`], [UINT32 `[M, N_0 + ... + N_{L-1}]`: the raw accumulators of all layers side by side. Layer `l` is the column range starting at `N_0 + ... + N_{l-1}`.],
)

The writer no longer receives one address per output tensor; it receives *three base addresses* and per layer a small offset, and builds the page index itself. A tile of the TILE-layout tensor `acc` is one page, in row-major tile order, so tile `(mt, c)` of layer `l` is page `mt * total_nt + nt_off_l + c`:

```cpp
// program factory: three addresses, then (Nt, first column tile) per layer
std::vector<uint32_t> writer_args = {out_addr, scalers_addr, acc_addr, out_tiles, Mt, total_nt};
for (uint32_t l = 0; l < L; l++) {
    writer_args.push_back(plan.Nt[l]);
    writer_args.push_back(plan.nt_off[l]);          // = N_0 + ... + N_{l-1}, in tiles
}

// on a cache hit: only the three addresses change (the offsets follow from the shapes, which are in the cache key)
writer_args[0] = outputs[out_index()].buffer()->address();
writer_args[1] = outputs[scalers_index()].buffer()->address();
writer_args[2] = outputs[acc_index()].buffer()->address();

// writer kernel (device side): tile c of row-block mt of layer l -> its page in the packed tensor
noc.async_write(cb_acc, acc, acc_bytes, {.offset_bytes = i * acc_bytes},
                {.page_id = mt * total_nt + nt_off + c});
noc.async_write(cb_so, scalers, tile_bytes, {}, {.page_id = scaler_out_page(l)});   // scaler i = page i
```

So the "address" of every layer's result is still a runtime argument, but it is now a base address (one per tensor) plus an offset the kernel adds. The allocation still has to finish before the command can be built, but there are three allocations, not `2 + 3L`. The host reads one value with a helper that indexes into the packed tensor (`result.scaler_out(l)`, `result.acc_raw(l)`).

The same change removed another host cost: with pre-quantized weights (`execute_network_quantized`) there are no weight-scaler outputs at all, because the scalers are inputs.

= Measurements after the fix

*Host time of one `execute_network` call* (`zkml_nn_network_bench hostprof`, M = 32, width 128, bf16, median of 100, in us). Before the fix: 62.2 us at L = 4 (14 outputs) and 88.5 us at L = 6 (20 outputs), at M = 128, width 256 (section 5).

#tbl((auto, auto, auto, auto, auto, auto),
  [*layers*], [*output tensors*], [*create outputs*], [*hash + key*], [*rest*], [*total*],
  [1], [3], [11.0], [0.6], [6.3], [18.0],
  [4], [3], [11.0], [1.2], [6.7], [18.9],
  [8], [3], [11.0], [2.0], [11.0], [24.1],
  [16], [3], [10.9], [3.1], [11.6], [26.0],
)

Creating the outputs is now a constant 11 us (three tensors) and the call is 18 to 26 us for 1 to 16 layers, against 88 us for six layers before. What still grows with the layer count is the program hash and the argument override, slowly.

*Wall time of the whole network* (us, bf16, median of 20; the layer loop is `L` calls of `execute_layer`):

#tbl((auto, auto, auto, auto, auto),
  [*M x W, 8 layers*], [*layer loop*], [*fused, weights quantized each call*], [*fused, weights pre-quantized*], [*best CPU (VNNI)*],
  [32 x 64], [194], [58], [51], [58],
  [32 x 128], [195], [73], [49], [92],
  [128 x 128], [204], [130], [100], [125],
  [128 x 256], [423], [315], [217], [252],
  [128 x 512], [1198], [1021], [608], [685],
)

Before the fix the fused call lost to the loop for wide layers (for example 32 x 256, 8 layers: 262 us against 217 us). Now the fused call is faster than the loop in every configuration we ran (the non-quantized one too: 182 us against 217 us for 32 x 256): 1.9x to 4.0x at 8 layers, and 2.0x to 8.0x at 32 layers (pre-quantized weights). The device kernel itself also got faster in this work (float requantizer, no weight quantization), which is why these numbers are lower than the host cost alone would explain; the host cost is no longer what decides the wall time.

= Where this leaves the design

- The host cost of an output tensor is the lever, and it is solved by packing: three allocations and three base addresses, offsets computed in the factory and the kernel.
- Two further options remain, neither needed now: *reuse the outputs* (pass preallocated tensors in, so a call has no allocation) and *trace replay* (capture the enqueue once and replay it for a fixed network, which removes the per-call host work).
- For a model that is run repeatedly with the same weights, quantize the weights once (`nn::Sequential::load_parameters` does this) and use `execute_network_quantized`: it removes 30% to 55% of the kernel time and needs no weight scaler outputs.

#good[With packed outputs the fused network is the faster choice at every size we measured, against the layer loop. With pre-quantized weights it is also faster than the best CPU baseline (including AVX-512 VNNI) for networks of 8 or more layers at most widths, and loses only for short, narrow networks where fixed per-call costs dominate.]
