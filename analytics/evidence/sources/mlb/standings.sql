select
    team_id,
    team_name,
    abbrev,
    league,
    division,
    elo,
    elo_rank,
    wins,
    losses,
    projected_wins,
    p_make_playoffs,
    p_win_division,
    p_win_pennant,
    p_win_world_series,
    world_series_rank
from main_marts.mart_standings
