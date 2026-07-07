# Long Horizon Planning: Recent Papers Digest

## Key Paper: **Cortex** (arXiv:2607.05377v1)

**Title:** *A Bidirectionally Aligned Embodied Agent Framework for Long-horizon Manipulation*  
**Authors:** Jiaqi Peng et al.

### Abstract
This paper addresses the fundamental limitation of current Vision-Language-Action (VLA) models in handling long-horizon tasks due to their Markovian nature—relying solely on current observations rather than planning across extended sequences. The authors introduce **Cortex**, a bidirectionally aligned embodied agent framework that bridges high-level planning semantics with low-level execution kinematics through a customized planning interface.

### Methodology
- Standardizes manipulation subtasks into 32 canonical skill primitives  
- Injects tractability principles (object attributes, trajectory reachability) into data generation pipelines  
- Enables automatic annotation of over 4k hours of open-source video data + 30 hours simulation data  
- Uses event-balanced sampling to handle planning ambiguity during subtask transitions  

### Results
Cortex outperforms monolithic baselines by:
- **3.1%** on Libero-long benchmark
- **4.1%** on RoboTwin

Notably, the generalist VLM enables zero-shot completion of unseen real-world long-horizon tasks (e.g., multi-stage chemistry experiments)—a capability infeasible through VLA fine-tuning alone.

### Significance
Cortex represents a significant advancement by overcoming the planning-execution gap that has limited embodied AI systems for extended task sequences, enabling capabilities previously considered impossible with standard VLA architectures.

---

*Note: Other recent papers on related topics include verification frameworks (LLM-as-a-Verifier) and world model datasets (Deform360), but Cortex is most directly focused on long-horizon planning methodology.*