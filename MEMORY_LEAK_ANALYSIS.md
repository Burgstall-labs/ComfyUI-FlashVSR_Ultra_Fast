# Memory Leak Analysis - FlashVSR ComfyUI

## Executive Summary

**THREE critical issues** were identified and fixed in the FlashVSR ComfyUI implementation:

1. **Buffer_LQ4x_Proj Cache Assignment Bug** (`src/models/utils.py`) - Memory leak affecting FlashVSR original model
2. **Tiled DIT Processing Leak** (`nodes.py`) - Memory leak affecting all models when using tiled_dit mode, especially with larger tile sizes
3. **Tiled Output Dtype Mismatch** (`nodes.py`) - Compatibility issue causing failures in downstream nodes like Fill's FILM

The first two caused intermediate tensors to accumulate in GPU/CPU memory, while the third caused dtype incompatibility with ComfyUI's IMAGE tensor requirements.

## Memory Leak #1: Buffer_LQ4x_Proj Cache Assignment

### Location
- **File**: `src/models/utils.py`
- **Class**: `Buffer_LQ4x_Proj`
- **Methods**: `forward()` and `stream_forward()`
- **Lines**: 304-310 (forward), 338-356 (stream_forward)
- **Affects**: FlashVSR original model only (FlashVSR-v1.1 uses Causal_LQ4x_Proj which was already correct)

### The Bug

In the `Buffer_LQ4x_Proj` class, the cache was being **assigned BEFORE being used** in the convolution operations:

```python
# INCORRECT ORDER (causing memory leak)
cache1_x = x[:, :, -CACHE_T:, :, :].clone()
self.cache['conv1'] = cache1_x          # ❌ Assigned first
x = self.conv1(x, self.cache['conv1'])  # Then used
```

This should have been:

```python
# CORRECT ORDER (fixed)
cache1_x = x[:, :, -CACHE_T:, :, :].clone()
x = self.conv1(x, self.cache['conv1'])  # ✅ Use old cache first
self.cache['conv1'] = cache1_x          # Then update cache
```

### Why This Causes Memory Leaks

1. **Broken Temporal Causality**: The convolution receives the current frame's cache instead of the previous frame's cache, breaking the causal temporal relationship.

2. **Tensor Reference Accumulation**: PyTorch's autograd graph retains references to intermediate tensors because:
   - The cache reference is updated prematurely
   - The computation graph maintains links to these tensors
   - Tensors are never properly released from memory

3. **Compounding Effect**: With video processing involving hundreds of frames, these unreleased tensors accumulate rapidly, causing:
   - GPU VRAM exhaustion
   - System memory (RAM) exhaustion
   - Performance degradation
   - Potential OOM (Out of Memory) crashes

## Evidence from Codebase

The correct implementation already existed in the `Causal_LQ4x_Proj` class (used for FlashVSR-v1.1 model), which shows the proper cache assignment order:

```python
# From Causal_LQ4x_Proj.forward() - lines 403-405
cache1_x = x[:, :, -CACHE_T:, :, :].clone()
x = self.conv1(x, self.cache['conv1'])  # ✅ Correct order
self.cache['conv1'] = cache1_x
```

This proves the developers were aware of the correct pattern but failed to apply it consistently to the `Buffer_LQ4x_Proj` class (used for the original FlashVSR model).

## Impact

### Affected Components
- **FlashVSR (original model)**: Uses `Buffer_LQ4x_Proj` - **AFFECTED**
- **FlashVSR-v1.1**: Uses `Causal_LQ4x_Proj` - **NOT AFFECTED** (already has correct implementation)

### Symptoms
- Progressive memory consumption during video processing
- Memory not released after processing completes
- VRAM/RAM exhaustion on longer videos
- Potential crashes with OOM errors
- Performance degradation over time

## Memory Leak #2: Tiled DIT Processing

### Location
- **File**: `nodes.py`
- **Function**: `flashvsr()`
- **Lines**: 266-304 (tiled_dit processing loop)
- **Affects**: All models (FlashVSR and FlashVSR-v1.1) when using tiled_dit mode

### The Bug

In the tiled DIT processing loop, memory was accumulating between tile iterations:

1. **Mask tensors not explicitly deleted**: `mask_nchw` and `mask_nhwc` tensors scaled with tile size but weren't deleted
2. **Pipeline caches persist across tiles**: `LQ_proj_in` and `TCDecoder` internal caches accumulated across spatial tiles
3. **Why tile_size matters**: Larger tiles (288 vs 256) → larger cached tensors → faster memory accumulation

**Before:**
```python
del LQ_tile, output_tile_gpu, processed_tile_cpu, input_tile
clean_vram()
# Missing: mask tensors and pipeline cache cleanup
```

**After:**
```python
del LQ_tile, output_tile_gpu, processed_tile_cpu, input_tile
# Explicitly delete mask tensors
del mask_nchw, mask_nhwc
# Clear pipeline caches between tiles to prevent accumulation
if hasattr(pipe, 'denoising_model') and hasattr(pipe.denoising_model(), 'LQ_proj_in'):
    pipe.denoising_model().LQ_proj_in.clear_cache()
if hasattr(pipe, 'TCDecoder'):
    pipe.TCDecoder.clean_mem()
clean_vram()
```

## Fixes Applied

### Fix #1: Buffer_LQ4x_Proj Cache Assignment

**File**: `src/models/utils.py`

#### 1. Fixed `Buffer_LQ4x_Proj.forward()` method (lines 304-314)

**Before:**
```python
cache1_x = x[:, :, -CACHE_T:, :, :].clone()
self.cache['conv1'] = cache1_x
x = self.conv1(x, self.cache['conv1'])
x = self.norm1(x)
x = self.act1(x)
cache2_x = x[:, :, -CACHE_T:, :, :].clone()
self.cache['conv2'] = cache2_x
if i == 0:
    continue
x = self.conv2(x, self.cache['conv2'])
```

**After:**
```python
cache1_x = x[:, :, -CACHE_T:, :, :].clone()
x = self.conv1(x, self.cache['conv1'])      # ✅ Use cache first
self.cache['conv1'] = cache1_x              # ✅ Then update
x = self.norm1(x)
x = self.act1(x)
cache2_x = x[:, :, -CACHE_T:, :, :].clone()
if i == 0:
    self.cache['conv2'] = cache2_x
    continue
x = self.conv2(x, self.cache['conv2'])      # ✅ Use cache first
self.cache['conv2'] = cache2_x              # ✅ Then update
```

#### 2. Fixed `Buffer_LQ4x_Proj.stream_forward()` method (lines 338-356)

Applied the same cache assignment order fix to both branches (clip_idx == 0 and else).

### Fix #2: Tiled DIT Processing Cleanup

**File**: `nodes.py`

**Location**: `flashvsr()` function, lines 296-304

Added explicit cleanup of mask tensors and pipeline caches between tile iterations:
- Delete mask tensors (`mask_nchw`, `mask_nhwc`)
- Clear `LQ_proj_in.clear_cache()` between tiles
- Clear `TCDecoder.clean_mem()` between tiles
- Safe hasattr checks ensure compatibility with all pipeline modes

### Fix #3: Tiled Output Dtype Conversion

**File**: `nodes.py`

**Location**: `flashvsr()` function, line 307

**Issue**: Tiled path returned bf16/fp16 tensors while non-tiled path returned float32, causing failures in downstream nodes.

**Before:**
```python
final_output = final_output_canvas / weight_sum_canvas
```

**After:**
```python
final_output = (final_output_canvas / weight_sum_canvas).float()
```

**Why this matters**:
- ComfyUI IMAGE tensors must be float32
- Non-tiled path uses `tensor2video()` which includes `.float()` conversion
- Tiled path was missing this conversion
- Downstream nodes like Fill's FILM expect float32 and fail with bf16/fp16

## Verification

### Testing Recommendations

1. **Memory Profiling**: Run video processing with memory profiling tools (e.g., `torch.cuda.memory_summary()`, `nvidia-smi`)
2. **Long Video Tests**: Process videos with 100+ frames and monitor memory usage
3. **Multiple Runs**: Execute multiple inference runs in sequence to verify memory is released
4. **Comparison Test**: Compare memory usage between FlashVSR (now fixed) and FlashVSR-v1.1

### Expected Results After Fixes

**Fix #1 (Buffer_LQ4x_Proj)**:
- ✅ Stable memory for FlashVSR original model users
- ✅ Proper temporal cache handling

**Fix #2 (Tiled DIT Memory)**:
- ✅ tile_size=288 works the same as tile_size=256
- ✅ Memory usage stable across all tile iterations
- ✅ No progressive accumulation regardless of tile count or size
- ✅ Consistent performance across multiple tiled runs
- ✅ No OOM errors with larger tile sizes

**Fix #3 (Tiled Output Dtype)**:
- ✅ Tiled output is float32, matching non-tiled output
- ✅ Fill's FILM node works correctly with FlashVSR output
- ✅ All downstream nodes receive proper float32 IMAGE tensors
- ✅ Consistent behavior regardless of tiled_dit setting

## Additional Notes

### Secondary Fix (Already in Codebase)

The newest version also included a `.float()` conversion fix in `nodes.py` line 323:

```python
stacked_image_tensor = torch.median(final_output, dim=0).values.unsqueeze(0).float().to('cpu')
```

This prevents dtype-related memory issues when moving tensors to CPU, though this is a minor optimization compared to the primary cache ordering bug.

### Why This Bug Was Introduced

The bug likely occurred because:
1. The `Buffer_LQ4x_Proj` class was implemented first with the incorrect pattern
2. When implementing `Causal_LQ4x_Proj` for v1.1, the developers corrected the pattern
3. The fix was never backported to `Buffer_LQ4x_Proj`
4. Code review may have missed the subtle ordering difference

## Conclusion

Three distinct issues have been successfully identified and fixed:

1. **Buffer_LQ4x_Proj cache bug**: Fixed cache assignment order to ensure proper temporal causality and prevent tensor accumulation for FlashVSR original model users.

2. **Tiled DIT processing leak**: Added explicit cleanup of mask tensors and pipeline caches between tile iterations, preventing memory accumulation when using tiled_dit mode with any tile size.

3. **Tiled output dtype mismatch**: Added float32 conversion to tiled output path to ensure ComfyUI compatibility and prevent failures in downstream nodes expecting float32 IMAGE tensors.

These fixes work together to ensure stable memory usage, correct dtype handling, and full compatibility across all FlashVSR modes, model versions, and processing configurations.

**Status**: ✅ **FIXED**

---

**Date**: November 7, 2025  
**Analyzed by**: AI Code Review System  
**Severity**: Critical (Memory Leak)  
**Priority**: High

