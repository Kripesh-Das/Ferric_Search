use super::kernels;
use super::kmeans::KMeans;
use crate::error::FerricError;
use crate::Result;

pub struct ProductQuantizer {
    dim: usize,          // dimension of each datapoint
    m: usize,            // no of subspaces
    ksub: usize,         // codes per subspace
    dsub: usize,         // dimension of each subspace
    codebooks: Vec<f32>, // learned centroids
    trained: bool,
    // codebooks.len() = 24{m} * 256{ksub} * 32{dsub = dim/m} = 196,608 floats
}

impl ProductQuantizer {
    pub fn new(dim: usize, m: usize, ksub: usize) -> Result<Self> {
        // Constructor
        if dim % m != 0 {
            return Err(FerricError::DimensionNotDivisible { dim, m });
        }
        if ksub > 256 {
            return Err(FerricError::KsubTooLarge(ksub));
        }

        Ok(Self {
            dim,
            m,
            ksub,
            dsub: dim / m,
            codebooks: Vec::new(),
            trained: false,
        })
    }

    // GETTER METHODS

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn ksub(&self) -> usize {
        self.ksub
    }

    pub fn dsub(&self) -> usize {
        self.dsub
    }

    pub fn trained(&self) -> bool {
        self.trained
    }

    // TRAINING

    pub fn train(&mut self, vectors: &[f32]) {
        // Train: Learn codebooks for each subspace using K-means clustering
        // Result: self.codebooks filled with m*ksub centroids of dimension dsub each

        let n = vectors.len() / self.dim;
        self.codebooks.resize(self.m * self.ksub * self.dsub, 0.0);

        // Temporary buffer for subspace data
        let mut sub = vec![0.0f32; n * self.dsub];

        // For each subspace m
        for m in 0..self.m {
            // STEP 1: Extract the m-th subspace from all vectors
            for i in 0..n {
                let src =
                    &vectors[i * self.dim + m * self.dsub..i * self.dim + (m + 1) * self.dsub];
                sub[i * self.dsub..(i + 1) * self.dsub].copy_from_slice(src);
            }

            // STEP 2: Run K-means on this subspace to get ksub centroids
            let km = KMeans::new(self.ksub, self.dsub, 25);
            let cb = km.fit(&sub);

            // STEP 3: Store the learned centroids in the codebook
            self.codebooks[m * self.ksub * self.dsub..(m + 1) * self.ksub * self.dsub]
                .copy_from_slice(&cb);
        }
        self.trained = true;
    }

    // ENCODING

    pub fn encode_one(&self, vec: &[f32]) -> Vec<u8> {
        // Encode a single vector into m codes (one per subspace)
        // Each code is a u8 index (0..ksub) pointing to the nearest centroid in that subspace

        assert!(self.trained, "ProductQuantizer not trained");
        let mut code = vec![0u8; self.m];

        // For each subspace
        for m in 0..self.m {
            // STEP 1: Extract the subspace portion of the vector
            let sub = &vec[m * self.dsub..(m + 1) * self.dsub];

            // STEP 2: Get the codebook for this subspace
            let cb = &self.codebooks[m * self.ksub * self.dsub..(m + 1) * self.ksub * self.dsub];

            // STEP 3: Find nearest centroid and store its index
            code[m] = self.nearest_centroid(sub, cb);
        }

        code
    }

    pub fn encode(&self, vectors: &[f32]) -> Vec<u8> {
        // Encode multiple vectors into a flat array of codes
        // n vectors -> n*m bytes (each vector becomes m bytes)

        assert!(self.trained, "ProductQuantizer not trained");
        let n = vectors.len() / self.dim;
        let mut codes = vec![0u8; n * self.m];

        // Encode each vector
        for i in 0..n {
            let vec = &vectors[i * self.dim..(i + 1) * self.dim];
            let code = self.encode_one(vec);
            codes[i * self.m..(i + 1) * self.m].copy_from_slice(&code);
        }

        codes
    }

    //  DISTANCE SEARCH

    pub fn precompute_table(&self, query: &[f32]) -> Vec<f32> {
        // Precompute lookup table for fast approximate distance calculation
        // For a query vector, compute distances to all centroids in all subspaces

        assert!(self.trained, "ProductQuantizer not trained");
        let mut table = vec![0.0f32; self.m * self.ksub];

        // For each subspace
        for m in 0..self.m {
            // STEP 1: Extract query subspace portion
            let qsub = &query[m * self.dsub..(m + 1) * self.dsub];

            // STEP 2: Get the codebook for this subspace
            let cb = &self.codebooks[m * self.ksub * self.dsub..(m + 1) * self.ksub * self.dsub];

            // STEP 3: Compute distance from query subspace to all centroids in this subspace
            for k in 0..self.ksub {
                table[m * self.ksub + k] =
                    kernels::l2_squared(qsub, &cb[k * self.dsub..(k + 1) * self.dsub]);
            }
        }

        table
    }

    pub fn approx_distance(&self, table: &[f32], code: &[u8]) -> f32 {
        // Compute approximate distance between query and encoded vector
        // Using precomputed lookup table and encoded vector's codes
        //
        let mut d = 0.0f32;
        for m in 0..self.m {
            d += table[m * self.ksub + code[m] as usize];
        }
        d
    }

    //  HELPER

    fn nearest_centroid(&self, sub: &[f32], cb: &[f32]) -> u8 {
        // Find the centroid closest to a subspace vectorb
        // Returns: index (0..ksub) of nearest centroid as u8

        let mut best = f32::MAX;
        let mut index = 0u8;

        // Linear search through all centroids
        for k in 0..self.ksub {
            let d = kernels::l2_squared(sub, &cb[k * self.dsub..(k + 1) * self.dsub]);
            if d < best {
                best = d;
                index = k as u8;
            }
        }
        index
    }
}
