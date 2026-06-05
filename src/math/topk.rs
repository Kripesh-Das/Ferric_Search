use crate::types::Hit;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub struct TopK {
    k: usize,
    heap: BinaryHeap<Reverse<Hit>>,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k),
        }
    }

    pub fn push(&mut self, id: i64, score: f32) {
        if self.heap.len() < self.k {
            self.heap.push(Reverse(Hit { id, score }));
        } else if let Some(Reverse(worst)) = self.heap.peek() {
            if score > worst.score {
                self.heap.pop();
                self.heap.push(Reverse(Hit { id, score }));
            }
        }
    }

    pub fn take_sorted(self) -> Vec<Hit> {
        let mut results: Vec<Hit> = self
            .heap
            .into_sorted_vec()
            .into_iter()
            .map(|Reverse(h)| h)
            .collect();

        results.reverse();
        return results;
    }
}
