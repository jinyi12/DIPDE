python -m src.ukappa_inverse_problem \
--noise 0.001 \
--num_samples 100 \
--max_iterations 1000 \
--initial_step_size 10000 \
--initial_lambda_reg 0.00001 \
--lambda_min_factor 1 \
--lambda_schedule_iterations 100 \
--regularizer tv \
--p_norm 2 \
--epsilon 1e-3
