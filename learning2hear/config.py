from __future__ import annotations


class TransferDiTSettings:
    model_id = "facebook/sam-audio-small"
    sample_rate = 48_000
    disable_model_rankers = True
    ode_method = "midpoint"
    ode_step_size = 2 / 32
    transfer_dit_layer_residual_rank = 8
    lora_rank = 16
    lora_alpha = 32.0
    lora_dropout = 0.0
    lora_targets = ("wq", "wk", "wv", "wo", "w1", "w2", "w3", "output", "t_block")


class EgoGraphSettings:
    pe_av_model = "pe-av-large"
    pe_av_batch_size = 64
    waveform_peak = 1.0
    pca_dim = 128
    pca_random_state = 0
    seed_ratio = 0.10
    n_neighbors = 64
    alpha = 0.20
    max_iter = 100
    tol = 1e-4
    n_jobs = -1
    selection_threshold = 0.90
