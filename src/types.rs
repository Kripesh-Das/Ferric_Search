pub enum Metric {
    Cosine,
    L2,
}

pub enum IndexKind {
    Flat,
    IVF,
    PQFlat,
    IVFPQ,
}

pub enum StorageType {
    F32,
    INT8,
}

pub struct IndexSpec {
    pub dim: usize,
    pub metric: Metric,
    pub normalize: bool,
    pub kind: IndexKind,
    pub storage: StorageType,
    pub nlist: usize,
    pub nprobe: usize,
    pub m: usize,
    pub ksub: usize,
}

impl IndexSpec {
    pub fn flat(dim: usize, metric: Metric) -> Self {
        Self {
            dim,
            metric,
            normalize: true,
            kind: IndexKind::Flat,
            storage: StorageType::F32,
            nlist: 0,
            nprobe: 1,
            m: 0,
            ksub: 0,
        }
    }

    pub fn ivf(dim: usize, metric: Metric, nlist: usize) -> Self {
        Self {
            dim,
            metric,
            normalize: true,
            kind: IndexKind::IVF,
            storage: StorageType::F32,
            nlist,
            nprobe: 1,
            m: 0,
            ksub: 0,
        }
    }

    pub fn pq_flat(dim: usize, metric: Metric, m: usize, ksub: usize) -> Self {
        Self {
            dim,
            metric,
            normalize: true,
            kind: IndexKind::PQFlat,
            storage: StorageType::F32,
            nlist: 0,
            nprobe: 1,
            m,
            ksub,
        }
    }

    pub fn ivf_pq(
        dim: usize,
        metric: Metric,
        nlist: usize,
        nprobe: usize,
        m: usize,
        ksub: usize,
    ) -> Self {
        Self {
            dim,
            metric,
            normalize: true,
            kind: IndexKind::IVFPQ,
            storage: StorageType::F32,
            nlist,
            nprobe,
            m,
            ksub,
        }
    }
}

pub struct Hit {
    pub id: i64,
    pub score: f32,
}
