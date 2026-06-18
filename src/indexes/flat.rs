// Stores all vectors in memory and performs linear scan for search queries.
// Time complexity: O(n*d) per query, n=vectors, d=dimension
// Space complexity: O(n*d)

use crate::math::kernels;
use crate::math::topk::TopK;
use crate::types::{Hit, Metric};

pub struct FlatIndex {
    dim: usize,
    metric: Metric,
    ids: Vec<i64>,
    vecs: Vec<f32>,
}

impl FlatIndex {
    pub fn new(dim: usize, metric: Metric) -> Self {
        // Constructor
        // IMPORTANT: All vectors are L2-normalized internally
        Self {
            dim,
            metric,
            ids: Vec::new(),
            vecs: Vec::new(),
        }
    }

    pub fn add(&mut self, ids: &[i64], vecs: &[f32]) {
        // STEP 1: Calculate number of vectors
        let n = vecs.len() / self.dim;
        assert_eq!(ids.len(), n, "ids and vectors length mismatch");

        // STEP 2: Store IDs
        self.ids.extend_from_slice(ids);

        // STEP 3: Store vectors with L2-normalization
        let mut tmp = vec![0.0f32; self.dim];
        for i in 0..n {
            // Extract vector i from input
            let src = &vecs[i * self.dim..(i + 1) * self.dim];
            tmp.copy_from_slice(src);
            // L2-normalize the vector in-place
            kernels::normalize(&mut tmp);
            // Store normalized vector
            self.vecs.extend_from_slice(&tmp);
        }
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        // STEP 1: Normalize query
        let mut q_buf = query.to_vec();
        kernels::normalize(&mut q_buf);
        let q = &q_buf;

        // STEP 2: Create TopK min-heap to track top k results
        let mut topk = TopK::new(k);

        // STEP 3: Linear scan through all vectors
        for i in 0..self.ids.len() {
            let vec = &self.vecs[i * self.dim..(i + 1) * self.dim];
            // Compute score based on metric
            // For normalized vectors:
            // - Cosine = dot(q, vec)
            // - L2 = -l2_squared(q, vec) = -(2 - 2*dot) = 2*dot - 2 (proportional to dot)
            let score = match self.metric {
                Metric::Cosine => kernels::dot(q, vec), // dot product, higher is better
                Metric::L2 => -kernels::l2_squared(q, vec), // negative L2,  higher is better
            };
            // Push to heap, will keep only top K
            topk.push(self.ids[i], score);
        }

        // STEP 4: Extract Best First
        topk.take_sorted()
    }

    // Get number of vectors in index
    pub fn size(&self) -> usize {
        self.ids.len()
    }
}
