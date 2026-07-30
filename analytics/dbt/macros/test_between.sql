-- Generic test: fails for any non-null value outside [min_value, max_value].
-- Dependency-free stand-in for dbt_utils.accepted_range.
{% test between(model, column_name, min_value, max_value) %}
select {{ column_name }}
from {{ model }}
where {{ column_name }} is not null
  and ({{ column_name }} < {{ min_value }} or {{ column_name }} > {{ max_value }})
{% endtest %}
