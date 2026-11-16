# Benchmark inputs

The five ADB scenario configurations and `data/adb_terms.json` of the challenge's benchmark,
`joint-loss-baseline` v0.5.0 (author James Edward Ball; Apache License 2.0). They are byte-identical to the
copy published at github.com/NOTMORSE-PROG/disaster-liquidity-reserves (folder `benchmark/`, commit
`d1188d4`); SHA-256 of `data/adb_terms.json`:
`50bdb628d0a67d14966d16bda35b4ab9d6b2fc473bdec42c9a664db10cc2b69b`.

They are used only to compute the benchmark's `problem_id` and to test `team_solver.py`. The values match the
brief (Tables 1, 4 and 5). Replace them with the organisers' official copy when it is available, then rerun
`python experiments/benchmark_submission.py` and `pytest`.
