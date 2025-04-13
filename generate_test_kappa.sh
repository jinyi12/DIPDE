python -m dataprep.genrf_vertices_coarse \
--output /data1/jy384/research/Data/DIPDE/randomfield_test \
--seed 42 \
--n_samples 1000 \
--lx 1.0 \
--ly 1.0 \
--lc 0.2 \
--nx 32 \
--ny 32 \
--mu 2.7 \
--sigma 0.3


python -m dataprep.genrf_vertices_interpolated \
--input /data1/jy384/research/Data/DIPDE/randomfield_test/ \
--output /data1/jy384/research/Data/DIPDE/randomfield_test/ \
--nx_coarse 32 \
--ny_coarse 32 \
--nx_fine 256 \
--ny_fine 256 \
--dataset_type test