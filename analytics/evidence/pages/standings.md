---
title: Standings & Playoff Odds
---

Team power board from `mart_standings`: current Elo rating, current-season record, and
Monte-Carlo playoff odds.

```sql leagues
select distinct league from mlb.standings order by league
```

<Dropdown data={leagues} name=league value=league>
    <DropdownOption value="%" valueLabel="All leagues"/>
</Dropdown>

```sql standings
select
    team_name, abbrev, league, division, elo, elo_rank,
    wins, losses, projected_wins,
    p_make_playoffs, p_win_division, p_win_pennant, p_win_world_series
from mlb.standings
where league like '${inputs.league.value}'
order by p_win_world_series desc
```

## Championship & playoff odds

<DataTable data={standings} rows=15>
    <Column id=team_name title="Team"/>
    <Column id=division/>
    <Column id=wins title="W"/>
    <Column id=losses title="L"/>
    <Column id=projected_wins title="Proj W" fmt="0.0"/>
    <Column id=p_make_playoffs title="Playoffs" fmt="0.0%" contentType=colorscale/>
    <Column id=p_win_division title="Division" fmt="0.0%" contentType=colorscale/>
    <Column id=p_win_pennant title="Pennant" fmt="0.0%" contentType=colorscale/>
    <Column id=p_win_world_series title="World Series" fmt="0.0%" contentType=colorscale/>
</DataTable>

## Elo rating vs projected wins

```sql elo_scatter
select team_name, abbrev, league, elo, projected_wins, p_win_world_series
from mlb.standings
where league like '${inputs.league.value}'
```

<ScatterPlot
    data={elo_scatter}
    x=elo
    y=projected_wins
    series=league
    tooltipTitle=team_name
    title="Elo vs projected wins"
/>
