# GT Sample Difficulty Scoring Criteria

This document defines the rubric used to assign the `difficulty` score to entries in [benchmark_dataset.json](/home/oss/Desktop/DataAgent/evaluation/benchmark_dataset.json).

The score is meant to reflect the complexity of the SQL reasoning required to answer the prompt correctly on the current schema, not the writing style of the prompt.

## Goal

The `difficulty` field is used to separate:

- simple lookup and aggregation tasks
- medium tasks with time bucketing or richer aggregation
- harder tasks that require joins or nested aggregation
- advanced tasks that combine joins with derived metrics, comparisons, or multi-step logic

The score should answer this question:

`How hard is it for the agent to generate the correct SQL for this prompt on this schema?`

## Scope Of The Scoring

The rubric is grounded in the current tables:

- `sales`
- `products`
- `stores`

Difficulty is primarily driven by:

- number of tables involved
- need for joins
- need for time aggregation
- number and type of aggregations
- use of subqueries or derived tables
- use of conditional logic such as `CASE WHEN`
- need for ratios, shares, or year-over-year comparisons
- number of dimensions mixed in the final result

Difficulty is not based on:

- whether the prompt sounds long or short
- whether a chart is requested
- whether the result has many rows

## Scoring Rubric

### Difficulty 1

Use `1` for straightforward SQL with a single level of aggregation and no joins.

Typical characteristics:

- one table, usually `sales`
- simple `WHERE` filters
- a single `GROUP BY`
- simple `ORDER BY`
- optional `LIMIT`
- no nested queries
- no derived metrics beyond a basic aggregate

Common SQL patterns:

- total revenue by store
- top product class codes by units sold
- promo vs non-promo totals

## Difficulty 2

Use `2` for still-manageable queries that add time bucketing or multiple output aggregates, but do not require joins or deeply composed logic.

Typical characteristics:

- still usually one table
- month/day level grouping with `DATE_TRUNC` or date extraction
- two or more aggregates in the same result
- top-N over aggregated time slices
- grouped comparisons that remain flat and do not need subqueries

Common SQL patterns:

- monthly revenue and units sold
- top revenue days in a year
- monthly trend for one filtered product class
- monthly revenue split by promo flag

## Difficulty 3

Use `3` for queries that require either joins or one additional reasoning layer such as a nested aggregation.

Typical characteristics:

- one or more joins to `products` or `stores`
- grouping on business dimensions outside `sales`
- derived price metrics such as realized average selling price
- grouped comparisons across product or store metadata
- one subquery or derived table, but not a full multi-step comparison pipeline

Common SQL patterns:

- revenue by brand, category, store type, or region
- top cities by revenue
- revenue by region for organic vs non-organic products
- average selling price by brand

## Difficulty 4

Use `4` for multi-step SQL that combines joins with non-trivial analytical logic.

Typical characteristics:

- multiple tables plus one or more derived steps
- nested aggregation, usually through a subquery or CTE
- ratio/share metrics
- conditional aggregation with `CASE WHEN`
- year-over-year comparisons
- average-of-aggregates logic
- mixing multiple business dimensions in the same result

Common SQL patterns:

- average monthly revenue by region or store type across years
- promo revenue share by category
- region-category year-over-year growth
- queries that require both joins and a derived comparison metric

## Practical Rules Used When Scoring

When assigning the score, apply these rules in order:

1. Start from the SQL required by the ground truth, not from the natural-language prompt.
2. If the query is single-table and flat, it is usually `1` or `2`.
3. If the query needs a join to `products` or `stores`, it is usually at least `3`.
4. If the query needs a subquery, derived table, share/ratio, or year-over-year comparison, it is usually `4`.
5. If a prompt could be written in both a simple and a complex way, score the actual GT SQL that was chosen.

## Tie-Breaking Guidance

If a sample sits between two levels, use these defaults:

- prefer `2` over `1` when time bucketing and multiple aggregates are both present
- prefer `3` over `2` when the query joins to another table
- prefer `4` over `3` when the query combines joins with nested aggregation or comparative metrics

## Examples From The Benchmark

- `1`: `Return the top 5 stores by total revenue`
- `2`: `Return the 12 months of 2023 with total revenue and total units sold`
- `3`: `Show total revenue by product brand for 2023 as a bar chart`
- `4`: `Compare average monthly revenue between store regions for 2022 and 2023`

## Notes For Future GT Additions

When adding new GT samples:

- keep the score consistent with the SQL logic actually required
- prefer a balanced spread across the rubric instead of clustering near one level
- use schema-aware prompts so the score reflects realistic database reasoning
- avoid inflating difficulty just because the wording is more specific or verbose
