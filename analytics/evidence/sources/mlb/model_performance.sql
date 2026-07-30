select
    model,
    is_baseline,
    n_scored,
    n_available,
    accuracy,
    brier,
    log_loss,
    delta_log_loss_vs_elo
from main_marts.mart_model_performance
