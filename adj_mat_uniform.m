function w = adj_mat_uniform(size, radius, sparsity)

rng(1);

w = sprand(size, size, sparsity);

e = eigs(w);
e = abs(e);
w = (w./max(e)) * radius;