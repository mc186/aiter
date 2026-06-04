# SPDX-License-Identifier: MIT

# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


@triton.jit
def remap_xcd_chunked(
    pid, GRID_MN, NUM_XCDS: tl.constexpr = 8, CHUNK_SIZE: tl.constexpr = 2
):
    # Compute current XCD and local PID
    xcd = pid % NUM_XCDS
    # distribute the modulo pids in round robin
    if pid > (GRID_MN // (NUM_XCDS * CHUNK_SIZE)) * (NUM_XCDS * CHUNK_SIZE):
        return pid
    local_pid = pid // NUM_XCDS
    # Calculate chunk index and position within chunk
    chunk_idx = local_pid // CHUNK_SIZE
    pos_in_chunk = local_pid % CHUNK_SIZE
    # Calculate new PID
    new_pid = chunk_idx * NUM_XCDS * CHUNK_SIZE + xcd * CHUNK_SIZE + pos_in_chunk
    return new_pid


@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )

    return pid


@triton.jit
def pid_grid(pid: int, num_pid_m: int, num_pid_n: int, GROUP_SIZE_M: tl.constexpr = 1):
    """
    Maps 1D pid to 2D grid coords (pid_m, pid_n).

    Args:
        - pid: 1D pid
        - num_pid_m: grid m size
        - num_pid_n: grid n size
        - GROUP_SIZE_M: tl.constexpr: default is 1
    """
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        tl.assume(group_size_m >= 0)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n


@triton.jit
def pid_grid_3d(pid: int, num_pid_m: int, num_pid_n: int, num_pid_k):
    """
    Maps 1D pid to 3D grid coords (pid_m, pid_n, pid_k).
    Args:
        - pid: 1D pid
        - num_pid_m: grid m size
        - num_pid_n: grid n size
        - num_pid_k: grid k size

    Returns:
        - pid_m, pid_n, pid_k: 3D grid coordinates
    """
    pid_m = pid % num_pid_m
    pid_n = (pid // num_pid_m) % num_pid_n
    pid_k = pid // (num_pid_m * num_pid_n) % num_pid_k

    return pid_m, pid_n, pid_k


@triton.jit
def remap_workgroup_head_first(
    wid,
    NUM_Q_HEADS,
    NUM_BLOCKS,
    BATCH,
    NUM_QUERIES_PER_KV: tl.constexpr,
    NUM_XCDS: tl.constexpr = 8,
):
    """
    GQA-aware spatial workgroup mapping for attention kernels on chiplet GPUs.

    The default kernel iteration (`off_q_head = wid % NUM_Q_HEADS`) combined with
    `remap_xcd` already provides good per-KV-head L2 locality when
    NUM_KV_HEADS == NUM_XCDS (e.g. GQA-128/8 on MI355X). This mapping makes the
    same intent EXPLICIT and remains correct + locality-preserving when the
    alignment doesn't hold (HK > NUM_XCDS or HK < NUM_XCDS).

    Strategy by regime:
      * HK == NUM_XCDS: 1 KV head per XCD (matches remap_xcd's effect exactly)
      * HK >  NUM_XCDS: each XCD owns ceil(HK/NXCD) heads, processed one fully
                        (all blocks x all Q-in-group) before moving to the next,
                        so a KV head's working set stays resident in L2 between
                        transitions.
      * HK <  NUM_XCDS: NXCD/HK XCDs share each KV head and split the per-KV
                        work between themselves.

    Args:
        wid:                  1-D workgroup id (0..BATCH*NUM_Q_HEADS*NUM_BLOCKS-1)
        NUM_Q_HEADS:          number of query heads
        NUM_BLOCKS:           number of query blocks along seqlen
        BATCH:                batch size
        NUM_QUERIES_PER_KV:   GQA ratio = NUM_Q_HEADS // NUM_KV_HEADS
                              (= 1 for MHA, will use the fallback below)
        NUM_XCDS:             number of accelerator complex dies (8 on MI300X/MI355X)

    Returns:
        (off_q_head, start_m, off_z) — same semantics as the default mapping.
    """
    xcd = wid % NUM_XCDS
    pos = wid // NUM_XCDS

    # MHA (no shared KV) falls back to head-first w/ XCD remap to retain
    # locality without GQA-specific decomposition.
    if NUM_QUERIES_PER_KV == 1:
        wgs_per_head = NUM_BLOCKS * BATCH
        local_head_idx = pos // wgs_per_head
        remainder = pos % wgs_per_head
        off_z = remainder // NUM_BLOCKS
        start_m = remainder % NUM_BLOCKS
        off_q_head = local_head_idx * NUM_XCDS + xcd
        if off_q_head >= NUM_Q_HEADS:
            off_q_head = wid % NUM_Q_HEADS
            start_m = (wid // NUM_Q_HEADS) % NUM_BLOCKS
            off_z = (wid // (NUM_BLOCKS * NUM_Q_HEADS)) % BATCH
        return off_q_head, start_m, off_z

    NUM_KV_HEADS: tl.constexpr = NUM_Q_HEADS // NUM_QUERIES_PER_KV

    if NUM_KV_HEADS >= NUM_XCDS:
        NUM_KV_REPLICAS: tl.constexpr = (NUM_KV_HEADS + NUM_XCDS - 1) // NUM_XCDS
        wgs_per_block = NUM_QUERIES_PER_KV * BATCH
        if NUM_KV_REPLICAS == 1:
            # HK == NUM_XCDS: 1 KV per XCD, identical effect to remap_xcd.
            kv_head = xcd % NUM_KV_HEADS
            start_m = (pos // wgs_per_block) % NUM_BLOCKS
            remainder = pos % wgs_per_block
            off_z = remainder // NUM_QUERIES_PER_KV
            local_q_in_group = remainder % NUM_QUERIES_PER_KV
            off_q_head = kv_head * NUM_QUERIES_PER_KV + local_q_in_group
        else:
            # HK > NUM_XCDS: each XCD owns multiple KV heads, processed one at a time.
            wgs_per_kv_group = NUM_QUERIES_PER_KV * NUM_BLOCKS * BATCH
            kv_slot = pos // wgs_per_kv_group
            kv_head = xcd + kv_slot * NUM_XCDS
            local = pos % wgs_per_kv_group
            start_m = (local // (NUM_QUERIES_PER_KV * BATCH)) % NUM_BLOCKS
            remainder = local % (NUM_QUERIES_PER_KV * BATCH)
            off_z = remainder // NUM_QUERIES_PER_KV
            local_q_in_group = remainder % NUM_QUERIES_PER_KV
            off_q_head = kv_head * NUM_QUERIES_PER_KV + local_q_in_group
    else:
        # HK < NUM_XCDS: cluster XCDs onto each KV head, splitting per-group work.
        NXCD_PER_KV: tl.constexpr = NUM_XCDS // NUM_KV_HEADS
        xcd_in_cluster = xcd // NUM_KV_HEADS
        kv_head = xcd % NUM_KV_HEADS
        wgs_per_kv_group = NUM_QUERIES_PER_KV * NUM_BLOCKS * BATCH
        wgs_per_xcd_in_cluster = wgs_per_kv_group // NXCD_PER_KV
        local = (pos % wgs_per_xcd_in_cluster) + xcd_in_cluster * wgs_per_xcd_in_cluster
        start_m = (local // (NUM_QUERIES_PER_KV * BATCH)) % NUM_BLOCKS
        remainder = local % (NUM_QUERIES_PER_KV * BATCH)
        off_z = remainder // NUM_QUERIES_PER_KV
        local_q_in_group = remainder % NUM_QUERIES_PER_KV
        off_q_head = kv_head * NUM_QUERIES_PER_KV + local_q_in_group

    # Safety fallback for non-divisible edge cases.
    if kv_head >= NUM_KV_HEADS or off_q_head >= NUM_Q_HEADS:
        off_q_head = wid % NUM_Q_HEADS
        start_m = (wid // NUM_Q_HEADS) % NUM_BLOCKS
        off_z = (wid // (NUM_BLOCKS * NUM_Q_HEADS)) % BATCH

    return off_q_head, start_m, off_z

