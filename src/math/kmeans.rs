use super::kernels;
use rayon::prelude::*;
use std::{f32, vec};

pub struct KMeans {
    k: usize, // no of centroids
    dim: usize,
    max_iters: usize,
}

impl KMeans {
    pub fn new(k: usize, dim: usize, max_iters: usize) -> Self {
        Self { k, dim, max_iters }
    }

    pub fn kmeans_init(&self, vectors: &[f32], n: usize) -> Vec<f32> {
        // creation of centroids
        // n --> no of vectors , vectors --> datapoints in 1D
        let mut rng = fastrand::Rng::new();
        let mut centroids = vec![0.0f32; self.k * self.dim]; // centroid points on 1D

        let first = rng.usize(..n);
        centroids[..self.dim].copy_from_slice(&vectors[first * self.dim..(first + 1) * self.dim]);

        for c in 1..self.k {
            let mut min_dists = vec![f32::MAX; n];

            for i in 0..n {
                let vec = &vectors[i * self.dim..(i + 1) * self.dim];

                for j in 0..c {
                    let centroid = &centroids[j * self.dim..(j + 1) * self.dim];
                    let d = kernels::l2_squared(vec, centroid);
                    if d < min_dists[i] {
                        min_dists[i] = d;
                    }
                }
            }

            let threshold = rng.f32();
            let mut cum_sum = 0.0f32;
            let mut pick = 0;

            for i in 0..n {
                cum_sum += min_dists[i];
                if cum_sum > threshold {
                    pick = i;
                    break;
                }
            }

            centroids[c * self.dim..(c + 1) * self.dim]
                .copy_from_slice(&vectors[pick * self.dim..(pick + 1) * self.dim]);
        }
        return centroids;
    }

    pub fn predict(&self, vectors: &[f32], centroids: &[f32]) -> Vec<usize> {
        // each datapoint assigned a centroid
        let n = vectors.len() / self.dim;
        let mut assignments = vec![0usize; n];

        assignments
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, cluster)| {
                let vec = &vectors[i * self.dim..(i + 1) * self.dim];
                let mut best_dist = f32::MAX;
                let mut best_cluster = 0;

                for c in 0..self.k {
                    let centroid = &centroids[c * self.dim..(c + 1) * self.dim];
                    let d = kernels::l2_squared(vec, centroid);
                    if d < best_dist {
                        best_dist = d;
                        best_cluster = c;
                    }
                }

                *cluster = best_cluster;
            });
        return assignments;
    }

    pub fn fit(&self, vectors: &[f32]) -> Vec<f32> {
        let n = vectors.len() / self.dim;
        assert!(n >= self.k, "not enough vectors for k-means");

        let mut centroids = self.kmeans_init(vectors, n);

        for _ in 0..self.max_iters {
            let assignments = self.predict(vectors, &centroids);

            let mut new_centroids = vec![0.0f32; self.k * self.dim];
            let mut counts = vec![0usize; self.k];

            for (i, &cluster) in assignments.iter().enumerate() {
                counts[cluster] += 1;

                let vec = &vectors[i * self.dim..(i + 1) * self.dim];

                // Get a mutable view of this cluster's spot in the new_centroids array
                let centroid = &mut new_centroids[cluster * self.dim..(cluster + 1) * self.dim];

                // Add the point's coordinates to the cluster's running total
                for d in 0..self.dim {
                    centroid[d] += vec[d];
                }
            }

            // Average the sums to find the new mathematical center (mean)
            for c in 0..self.k {
                if counts[c] > 0 {
                    let centroid = &mut new_centroids[c * self.dim..(c + 1) * self.dim];
                    for d in 0..self.dim {
                        // Divide the total sum by the number of points to get the average
                        centroid[d] /= counts[c] as f32;
                    }
                } else {
                    // If a cluster is empty, resurrect it with a random data point
                    let idx = fastrand::usize(..n);
                    new_centroids[c * self.dim..(c + 1) * self.dim]
                        .copy_from_slice(&vectors[idx * self.dim..(idx + 1) * self.dim]);
                }
            }
            centroids = new_centroids;
        }
        centroids
    }
}
