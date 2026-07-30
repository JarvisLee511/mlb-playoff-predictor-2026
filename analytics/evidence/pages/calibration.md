---
title: Calibration
---

A model is **calibrated** when games it calls at probability _p_ are won about _p_ of
the time. The diagonal is perfect calibration; points above it mean the model was
under-confident, below it over-confident. From `mart_calibration`.

```sql cal
select model, mean_predicted, fraction_won, calibration_gap
from mlb.calibration
order by model, mean_predicted
```

<ScatterPlot
    data={cal}
    x=mean_predicted
    y=fraction_won
    series=model
    xFmt="0.0%"
    yFmt="0.0%"
    xMin=0.3
    xMax=0.7
    yMin=0.3
    yMax=0.7
    title="Reliability curve — predicted vs actual win rate"
>
    <ReferenceLine data={cal} x=mean_predicted y=mean_predicted label="perfect calibration"/>
</ScatterPlot>

## Calibration gap by bin

`fraction_won − mean_predicted`; closer to zero is better.

```sql gaps
select model, mean_predicted, calibration_gap
from mlb.calibration
order by model, mean_predicted
```

<LineChart data={gaps} x=mean_predicted y=calibration_gap series=model xFmt="0.0%" title="Calibration gap across probability bins">
    <ReferenceLine y=0 label="calibrated"/>
</LineChart>
