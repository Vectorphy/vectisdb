# Results summary

## TPC-H scale-factor runs

Each row covers 22 TPC-H queries in one scale factor and cache state. CPU time and best-GPU time are arithmetic means of the recorded per-query times in milliseconds. Mean speedup is the arithmetic mean of the 22 per-query ratios, calculated as CPU time divided by the lowest recorded GPU time for that query. It is not an aggregate throughput figure.

| Scale factor | Cache state | Queries | Mean CPU time (ms) | Mean best GPU time (ms) | Mean per-query speedup | GPU faster |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Cached | 22 | 104.95 | 38.83 | 3.08x | 21 / 22 |
| 1 | Cold cache | 22 | 165.31 | 130.57 | 1.26x | 17 / 22 |
| 2 | Cached | 22 | 206.97 | 74.11 | 3.31x | 21 / 22 |
| 2 | Cold cache | 22 | 306.77 | 213.25 | 1.43x | 20 / 22 |
| 4 | Cached | 22 | 420.08 | 342.20 | 1.21x | 18 / 22 |
| 4 | Cold cache | 22 | 619.59 | 406.04 | 1.52x | 20 / 22 |
| 8 | Cached | 22 | 980.02 | 663.90 | 1.47x | 22 / 22 |
| 8 | Cold cache | 22 | 1,429.30 | 767.84 | 1.94x | 22 / 22 |
| 16 | Cached | 22 | 1,845.42 | 1,368.52 | 1.37x | 22 / 22 |
| 16 | Cold cache | 22 | 2,532.81 | 1,531.90 | 1.71x | 22 / 22 |
| 32 | Cached | 22 | 3,862.51 | 3,024.11 | 1.35x | 22 / 22 |
| 32 | Cold cache | 22 | 6,206.15 | 3,452.11 | 1.92x | 22 / 22 |
| 64 | Cached | 22 | 8,680.49 | 5,955.60 | 1.50x | 22 / 22 |
| 64 | Cold cache | 22 | 12,946.29 | 6,856.15 | 2.01x | 22 / 22 |

## What the table shows

The cached runs at scale factors 1 and 2 have the highest mean per-query ratios in this set: 3.08x and 3.31x. The cached scale-factor-4 result is lower at 1.21x, despite 18 of 22 queries recording a lower best-GPU time. From scale factor 8 onward, the best recorded GPU time is lower than the CPU time for every query in both cache states.

Cold-cache measurements also favour the best recorded GPU time more consistently at larger scale factors. The mean per-query ratio rises from 1.26x at scale factor 1 to 2.01x at scale factor 64. These are observations from the supplied runs, not a claim about performance on other hardware or data distributions.

## How to interpret “best GPU”

For each query and cache state, the source data contains eight GPU configuration results. The value used here is the fastest of those eight measurements. This makes the table useful for identifying headroom in the prototype, but it does not represent a fixed GPU configuration that a deployment would select without query-specific tuning. Read [methodology and limits](methodology.md) before comparing these values with another system.
