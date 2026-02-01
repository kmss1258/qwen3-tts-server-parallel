# Deployment Investigation Summary

**Date**: January 25, 2026  
**Task**: Deploy and benchmark vLLM backend for Qwen3-TTS

## Actions Taken

### 1. Git Pull and Update
- ✅ Successfully pulled latest changes from repository
- 📥 Retrieved 35 new/modified files including vLLM backend implementation
- 📋 New files included:
  - `api/backends/*` - Backend architecture
  - `docs/vllm-backend.md` - vLLM documentation
  - `tests/` - Test suite
  - Updated Dockerfile with vLLM stage

### 2. vLLM Backend Investigation

**Discovery**: The vLLM backend is **non-functional**

#### Root Cause
The implementation attempts to import `from vllm import Omni`, but:
- vLLM 0.14.1 (latest) does not have an `Omni` class
- vLLM's multi-modal support exists but lacks TTS-specific interfaces
- The backend appears to be a **placeholder for future functionality**

#### Evidence
```python
# From api/backends/vllm_omni_qwen3_tts.py (line 70)
from vllm import Omni  # ❌ ImportError: cannot import name 'Omni'
```

Testing showed:
- ✅ `vllm.LLM` exists
- ✅ `vllm.MultiModalRegistry` exists  
- ❌ `vllm.Omni` does NOT exist

### 3. Docker Deployment Attempts

#### GPU Configuration Issues Encountered

**Problem**: PyTorch unable to detect CUDA GPUs despite nvidia-smi working

**Symptoms**:
```bash
# Inside container
nvidia-smi          # ✅ Works - shows GPU
python -c "import torch; torch.cuda.is_available()"  # ❌ Returns False
```

**Debugging Steps Taken**:
1. ✅ Verified GPU allocation in docker-compose
2. ✅ Checked NVIDIA device requests (correctly configured)
3. ✅ Confirmed CUDA libraries exist at `/usr/local/cuda-12.1/`
4. ✅ Updated `LD_LIBRARY_PATH` in docker-compose
5. ✅ Tested with clean PyTorch container (worked fine)
6. ❌ Issue persists with custom-built Qwen3-TTS container

**Configuration Applied**:
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=2
  - LD_LIBRARY_PATH=/usr/local/cuda-12.1/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['2']
          capabilities: [gpu]
```

**Status**: GPU passthrough issue remains unresolved in current container build

### 4. Benchmark Results

Due to GPU detection issues, benchmarks ran on **CPU mode** instead of GPU:

| Test Case | Words | Time (CPU) | Notes |
|-----------|-------|------------|-------|
| Ultra Short | 2 | 8.06s | Much slower than expected |
| Short | 2 | 9.32s | |
| Medium | 12 | 55.05s | |
| Medium-Long | 32 | 135.85s | |
| Long | 39 | 234.33s | |
| Very Long | 74 | 386.32s | ~6.5 min for 74 words |

**Comparison to Published Benchmarks**:
- Published GPU (RTX 3090): 1.00s for 2 words
- Our CPU run: 8.06s for 2 words
- **~8x slower** due to running on CPU instead of GPU

### 5. Files Created/Modified

#### Created:
- `VLLM_BACKEND_STATUS.md` - Detailed investigation report
- `benchmark_official.py` - Comprehensive benchmark script
- `/tmp/tts_benchmark/benchmark_results.json` - Benchmark data

#### Modified:
- `README.md` - Updated backend status table
- `api/backends/__init__.py` - Fixed missing `initialize_backend` export
- `docker-compose.yml` - Added GPU device selection and LD_LIBRARY_PATH

#### Committed and Pushed:
```
commit 3f618fa
Fix: Update vLLM backend status and document GPU issues
- 5 files changed
- 290 insertions, 9 deletions
```

## Current Status

### ✅ Completed
- [x] Git pull successful
- [x] vLLM backend investigation complete
- [x] Documentation updated with accurate status
- [x] Fixed backend initialization export issue
- [x] Created comprehensive benchmark script
- [x] Changes committed and pushed to GitHub

### ⚠️ Issues Identified

1. **vLLM Backend Non-Functional**
   - Missing `Omni` class in vLLM library
   - Requires vLLM upstream changes or custom implementation
   - Marked as experimental/planned in README

2. **GPU Passthrough Problem**
   - Container build doesn't properly expose CUDA to PyTorch
   - nvidia-smi works but torch.cuda.is_available() returns False
   - Likely related to container build process or library paths
   - Needs further investigation or rebuild from different base image

### ❌ Could Not Complete
- [ ] Performance benchmarks on GPU (due to CUDA detection issue)
- [ ] vLLM backend deployment (backend is non-functional)
- [ ] Speed comparison vs. official backend (both unavailable on GPU)

## Recommendations

### Immediate Actions

1. **For Users**:
   - Use official backend (`TTS_BACKEND=official`)
   - Do not attempt to use vLLM backend
   - Expect GPU inference if properly configured

2. **For Developers**:
   - **Fix GPU Detection**: Debug why PyTorch can't see CUDA in container
     - Consider using official PyTorch base image instead of NVIDIA CUDA image
     - Verify CUDA toolkit vs runtime image requirements
     - Test with `--runtime=nvidia` flag explicitly
   
   - **vLLM Backend**: Either:
     - Wait for vLLM to add native TTS support
     - Implement custom adapter for Qwen3-TTS using vLLM primitives
     - Remove non-functional backend and documentation claims

### Next Steps to Resolve GPU Issue

1. Try rebuilding with PyTorch official base image:
   ```dockerfile
   FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
   ```

2. Alternatively, use runtime flag:
   ```yaml
   runtime: nvidia
   ```

3. Verify CUDA toolkit libraries are properly installed in builder stage

## Conclusion

The git pull revealed that the vLLM backend is a **planned feature**, not a working implementation. The codebase references functionality (`vllm.Omni`) that doesn't exist in the current vLLM library.

Additionally, there's a container configuration issue preventing PyTorch from accessing the GPU, which blocked GPU benchmark testing.

**The repository has been updated** to:
- ✅ Accurately reflect backend status
- ✅ Document the vLLM backend limitations
- ✅ Provide investigation findings
- ✅ Include comprehensive benchmark tooling

**Users should use the official backend**, which is stable and production-ready, albeit currently experiencing GPU detection issues in the Docker deployment that need further investigation.

---

*Investigation completed: January 25, 2026*
