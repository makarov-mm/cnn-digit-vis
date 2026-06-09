pub mod tensor;
pub mod layers;
pub mod cnn;
pub mod mnist;

use crate::cnn::Cnn;
use crate::tensor::Vol;

impl Cnn {
    pub fn forward_backward(&mut self, input: &Vol, label: usize) {
        self.conv1.zero_grad();
        self.conv2.zero_grad();
        self.fc1.zero_grad();
        self.fc2.zero_grad();
        self.forward(input);
        self.backward(label);
    }
}

#[cfg(test)]
mod tests {
    use crate::cnn::Cnn;
    use crate::layers::{softmax, Conv, Dense, MaxPool};
    use crate::tensor::{Rng, Vol};

    fn rand_input(rng: &mut Rng) -> Vol {
        let mut v = Vol::zeros(1, 28, 28);
        for x in v.data.iter_mut() {
            *x = rng.unit();
        }
        v
    }

    #[test]
    fn softmax_sums_to_one() {
        let s = softmax(&[1.0, 2.0, 3.0, -1.0]);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(s.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn forward_shapes() {
        let mut net = Cnn::new(1);
        let mut rng = Rng::new(2);
        let probs = net.forward(&rand_input(&mut rng)).to_vec();
        assert_eq!(probs.len(), 10);
        assert_eq!((net.a_c1.c, net.a_c1.h, net.a_c1.w), (8, 28, 28));
        assert_eq!((net.a_p1.c, net.a_p1.h, net.a_p1.w), (8, 14, 14));
        assert_eq!((net.a_c2.c, net.a_c2.h, net.a_c2.w), (16, 14, 14));
        assert_eq!((net.a_p2.c, net.a_p2.h, net.a_p2.w), (16, 7, 7));
        assert_eq!(net.a_fc1.len(), 64);
    }

    #[test]
    fn maxpool_picks_max_and_routes_grad() {
        let mut p = MaxPool::new();
        let mut x = Vol::zeros(1, 2, 2);
        x.data = vec![1.0, 2.0, 3.0, 9.0];
        let out = p.forward(&x);
        assert_eq!(out.data, vec![9.0]);
        let mut go = Vol::zeros(1, 1, 1);
        go.data = vec![5.0];
        let gi = p.backward(&go);
        assert_eq!(gi.data, vec![0.0, 0.0, 0.0, 5.0]);
    }

    #[test]
    fn save_load_roundtrip() {
        let mut a = Cnn::new(7);
        let path = std::env::temp_dir().join("cnn_test_weights.bin");
        let p = path.to_str().unwrap();
        a.save(p).unwrap();

        let mut b = Cnn::new(999);
        assert!(b.try_load(p));

        let mut rng = Rng::new(3);
        let inp = rand_input(&mut rng);
        let pa = a.forward(&inp).to_vec();
        let pb = b.forward(&inp).to_vec();
        for i in 0..10 {
            assert!((pa[i] - pb[i]).abs() < 1e-6);
        }
    }

    fn rel_err(a: f32, b: f32) -> f32 {
        let d = (a - b).abs();
        let s = a.abs() + b.abs();
        if s < 1e-6 { 0.0 } else { d / s }
    }

    fn check_param(net: &mut Cnn, sel: fn(&mut Cnn) -> &mut Vec<f32>, idx: usize,
                   input: &Vol, label: usize, analytic: f32, eps: f32) -> Option<f32> {
        let orig = sel(net)[idx];

        sel(net)[idx] = orig + eps;
        let lp = net.loss(input, label);
        sel(net)[idx] = orig - eps;
        let lm = net.loss(input, label);
        sel(net)[idx] = orig;

        let num = (lp - lm) / (2.0 * eps);
        if analytic.abs() < 5e-3 && num.abs() < 5e-3 {
            return None;
        }
        Some(rel_err(analytic, num))
    }

    fn median(mut v: Vec<f32>) -> f32 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if v.is_empty() { 0.0 } else { v[v.len() / 2] }
    }

    #[test]
    fn gradient_check_smooth_head() {
        let mut net = Cnn::new(11);
        let mut rng = Rng::new(123);
        let input = rand_input(&mut rng);
        let label = 4usize;
        net.forward_backward(&input, label);

        let mut worst = 0.0f32;
        let mut checked = 0;
        for idx in 0..net.fc2.w.len() {
            let analytic = net.fc2.dw[idx];
            if let Some(e) = check_param(&mut net, |n| &mut n.fc2.w, idx, &input, label, analytic, 1e-3) {
                worst = worst.max(e);
                checked += 1;
                assert!(e < 0.03, "fc2 grad mismatch: rel err {} (analytic {})", e, analytic);
            }
        }
        assert!(checked >= 5);
        println!("smooth-head check: {} params, worst rel err {:.5}", checked, worst);
    }

    #[test]
    fn gradient_check_conv_and_fc1() {
        // За ReLU конечные разности шумят на изломах — проверяем медиану по выборке.
        let mut net = Cnn::new(11);
        let mut rng = Rng::new(123);
        let input = rand_input(&mut rng);
        let label = 4usize;
        net.forward_backward(&input, label);

        let eps = 3e-3;
        let mut errs: Vec<f32> = Vec::new();

        macro_rules! probe {
            ($wsel:expr, $dw:expr, $len:expr) => {{
                let len = $len;
                for s in 0..24 {
                    let idx = (s * 2654435761usize) % len;
                    let analytic = $dw(&mut net)[idx];
                    if let Some(e) = check_param(&mut net, $wsel, idx, &input, label, analytic, eps) {
                        errs.push(e);
                    }
                }
            }};
        }

        probe!(|n: &mut Cnn| &mut n.conv1.w, |n: &mut Cnn| n.conv1.dw.clone(), net.conv1.w.len());
        probe!(|n: &mut Cnn| &mut n.conv2.w, |n: &mut Cnn| n.conv2.dw.clone(), net.conv2.w.len());
        probe!(|n: &mut Cnn| &mut n.fc1.w, |n: &mut Cnn| n.fc1.dw.clone(), net.fc1.w.len());

        assert!(errs.len() >= 12, "too few params validated: {}", errs.len());
        let med = median(errs.clone());
        let bad = errs.iter().filter(|&&e| e > 0.20).count();
        println!("conv/fc1 check: {} params, median rel err {:.4}, outliers {}", errs.len(), med, bad);
        assert!(med < 0.05, "median rel err {} too high", med);
        assert!((bad as f32) < 0.25 * errs.len() as f32, "too many outliers: {}", bad);
    }

    #[allow(dead_code)]
    fn _unused(_: Conv, _: Dense) {}
}
