use cnn_core::cnn::Cnn;
use cnn_core::tensor::Rng;
use std::time::Instant;

fn main() {
    let dir = "data";
    let train_n: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(10000);
    let epochs: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(3);

    let mut train = cnn_core::mnist::load(&dir, false).expect("train data");
    let test = cnn_core::mnist::load(&dir, true).expect("test data");
    train.truncate(train_n);
    println!("train {} test {}", train.len(), test.len());

    let mut net = Cnn::new(1);
    let mut rng = Rng::new(7);
    let batch = 32;
    let mut lr = 0.08;

    for ep in 0..epochs {
        for i in (0..train.len()).rev() {
            let j = rng.below(i + 1);
            train.swap(i, j);
        }
        let t = Instant::now();
        let mut loss = 0.0;
        let mut nb = 0;
        let mut s = 0;
        while s < train.len() {
            let end = (s + batch).min(train.len());
            loss += net.train_batch(&train[s..end], lr);
            nb += 1;
            s = end;
        }
        let mut correct = 0;
        for (x, y) in test.iter() {
            if net.predict(x) == *y as usize { correct += 1; }
        }
        let acc = 100.0 * correct as f32 / test.len() as f32;
        println!("epoch {}: loss {:.3}  test acc {:.2}%  ({:.1}s)", ep + 1, loss / nb as f32, acc, t.elapsed().as_secs_f32());
        lr *= 0.7;
    }

    net.save("weights.bin").expect("save weights");
    println!("saved weights.bin");
}
