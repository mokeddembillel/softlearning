#!/bin/bash

# softlearning run_example_local examples.multi_goal \
# --algorithm SQL \
# --universe gym \
# --domain MultiGoal \
# --task Default-v0 \
# --exp-nam my-sql-experiment-1 \
# --checkpoint-frequency 250 \
# --local-dir ~/Desktop/Code/softlearning/ray_results



softlearning run_example_local examples.development \
--algorithm SQL \
--universe gym \
--domain HalfCheetah \
--task v2 \
--exp-nam my-sql-experiment-1 \
--checkpoint-frequency 1000 \
--local-dir ~/Desktop/Code/softlearning/ray_results \
>> ./ray_results/sql_HalfCheetah_s1.txt



