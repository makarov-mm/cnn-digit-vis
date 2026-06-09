use cnn_core::cnn::Cnn;
use cnn_core::mnist;
use cnn_core::tensor::{argmax, Rng, Vol};
use macroquad::prelude::*;
use std::path::Path;
use std::sync::mpsc::{self, Receiver};
use std::thread;

const DRAW_RES: usize = 280;
const WEIGHTS: &str = "weights.bin";
const DATA_DIR: &str = "data";

fn window_conf() -> Conf {
    Conf {
        window_title: "CNN Feature Maps".to_owned(),
        window_width: 1320,
        window_height: 880,
        high_dpi: true,
        ..Default::default()
    }
}

enum Stage {
    NoData,
    Training(Receiver<Msg>, String),
    Ready,
}

enum Msg {
    Progress(String),
    Done(Box<Cnn>),
}

fn inferno(t: f32) -> Color {
    let stops = [
        (0.0f32, 0.0, 0.0, 0.015),
        (0.25, 0.34, 0.06, 0.43),
        (0.5, 0.74, 0.22, 0.33),
        (0.75, 0.98, 0.56, 0.035),
        (1.0, 0.99, 1.0, 0.64),
    ];
    let t = t.clamp(0.0, 1.0);
    for i in 0..stops.len() - 1 {
        let (t0, r0, g0, b0) = stops[i];
        let (t1, r1, g1, b1) = stops[i + 1];
        if t <= t1 {
            let f = (t - t0) / (t1 - t0).max(1e-6);
            return Color::new(r0 + (r1 - r0) * f, g0 + (g1 - g0) * f, b0 + (b1 - b0) * f, 1.0);
        }
    }
    Color::new(0.99, 1.0, 0.64, 1.0)
}

fn channel_max(v: &Vol, ch: usize) -> f32 {
    let mut m = 1e-6f32;
    let base = ch * v.h * v.w;
    for i in 0..v.h * v.w {
        m = m.max(v.data[base + i]);
    }
    m
}

fn draw_map(v: &Vol, ch: usize, x: f32, y: f32, px: f32) {
    let m = channel_max(v, ch);
    for yy in 0..v.h {
        for xx in 0..v.w {
            let t = v.get(ch, yy, xx) / m;
            draw_rectangle(x + xx as f32 * px, y + yy as f32 * px, px, px, inferno(t));
        }
    }
}

fn draw_layer(label: &str, v: &Vol, x0: f32, y: f32, px: f32, gap: f32) -> f32 {
    draw_text(label, x0, y - 8.0, 22.0, Color::new(0.85, 0.85, 0.9, 1.0));
    let tile = v.w as f32 * px;
    for ch in 0..v.c {
        let x = x0 + ch as f32 * (tile + gap);
        draw_map(v, ch, x, y, px);
    }
    v.h as f32 * px
}

fn paint(buf: &mut [f32], cx: f32, cy: f32) {
    let r = 11.0f32;
    let ri = r.ceil() as i32;
    let (cxi, cyi) = (cx as i32, cy as i32);
    for dy in -ri..=ri {
        for dx in -ri..=ri {
            let (x, y) = (cxi + dx, cyi + dy);
            if x < 0 || y < 0 || x >= DRAW_RES as i32 || y >= DRAW_RES as i32 {
                continue;
            }
            let d = ((dx * dx + dy * dy) as f32).sqrt();
            if d <= r {
                let v = (1.0 - d / r).min(1.0);
                let idx = y as usize * DRAW_RES + x as usize;
                buf[idx] = buf[idx].max(v);
            }
        }
    }
}

fn start_training() -> Receiver<Msg> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut train = match mnist::load(DATA_DIR, false) {
            Ok(v) => v,
            Err(e) => {
                let _ = tx.send(Msg::Progress(format!("data error: {e}")));
                return;
            }
        };
        let test = mnist::load(DATA_DIR, true).unwrap_or_default();
        train.truncate(20000);

        let mut net = Cnn::new(1);
        let mut rng = Rng::new(7);
        let batch = 32;
        let mut lr = 0.08;
        let epochs = 4;

        for ep in 0..epochs {
            for i in (1..train.len()).rev() {
                let j = rng.below(i + 1);
                train.swap(i, j);
            }
            let mut s = 0;
            while s < train.len() {
                let end = (s + batch).min(train.len());
                net.train_batch(&train[s..end], lr);
                s = end;
            }
            let mut acc = 0.0;
            if !test.is_empty() {
                let mut correct = 0;
                for (x, y) in test.iter().take(2000) {
                    if net.predict(x) == *y as usize {
                        correct += 1;
                    }
                }
                acc = 100.0 * correct as f32 / 2000.0;
            }
            let _ = tx.send(Msg::Progress(format!("epoch {}/{}  test acc {:.1}%", ep + 1, epochs, acc)));
            lr *= 0.7;
        }
        let _ = net.save(WEIGHTS);
        let _ = tx.send(Msg::Done(Box::new(net)));
    });
    rx
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut net = Cnn::new(1);
    let mut stage = if net.try_load(WEIGHTS) {
        Stage::Ready
    } else if Path::new(DATA_DIR).join("train-images-idx3-ubyte.gz").exists() {
        Stage::Training(start_training(), "starting…".to_owned())
    } else {
        Stage::NoData
    };

    let mut buf = vec![0.0f32; DRAW_RES * DRAW_RES];
    let mut dirty = true;
    let mut last_mouse: Option<(f32, f32)> = None;
    let mut input = Vol::zeros(1, 28, 28);

    let panel = Rect::new(40.0, 80.0, 280.0, 280.0);
    let clear_btn = Rect::new(40.0, 372.0, 130.0, 36.0);

    loop {
        clear_background(Color::new(0.05, 0.06, 0.08, 1.0));

        if let Stage::Training(rx, status) = &mut stage {
            let mut done: Option<Box<Cnn>> = None;
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    Msg::Progress(s) => *status = s,
                    Msg::Done(n) => done = Some(n),
                }
            }
            if let Some(n) = done {
                net = *n;
                dirty = true;
                stage = Stage::Ready;
            }
        }

        if matches!(stage, Stage::Ready) {
            let (mx, my) = mouse_position();
            if is_mouse_button_down(MouseButton::Left) && panel.contains(vec2(mx, my)) {
                let bx = (mx - panel.x) / panel.w * DRAW_RES as f32;
                let by = (my - panel.y) / panel.h * DRAW_RES as f32;
                if let Some((lx, ly)) = last_mouse {
                    let steps = 8;
                    for i in 0..=steps {
                        let f = i as f32 / steps as f32;
                        paint(&mut buf, lx + (bx - lx) * f, ly + (by - ly) * f);
                    }
                } else {
                    paint(&mut buf, bx, by);
                }
                last_mouse = Some((bx, by));
                dirty = true;
            } else {
                last_mouse = None;
            }

            if is_mouse_button_pressed(MouseButton::Left) && clear_btn.contains(vec2(mx, my)) {
                buf.iter_mut().for_each(|v| *v = 0.0);
                dirty = true;
            }

            if dirty {
                input = mnist::preprocess(&buf, DRAW_RES, DRAW_RES);
                net.forward(&input);
                dirty = false;
            }
        }

        draw_text("Draw a digit", panel.x, panel.y - 12.0, 24.0, WHITE);
        draw_rectangle(panel.x, panel.y, panel.w, panel.h, Color::new(0.1, 0.1, 0.13, 1.0));
        // штрихи
        let cell = panel.w / DRAW_RES as f32;
        let step = 3;
        for y in (0..DRAW_RES).step_by(step) {
            for x in (0..DRAW_RES).step_by(step) {
                let v = buf[y * DRAW_RES + x];
                if v > 0.05 {
                    draw_rectangle(
                        panel.x + x as f32 * cell,
                        panel.y + y as f32 * cell,
                        cell * step as f32,
                        cell * step as f32,
                        Color::new(0.95, 0.95, 1.0, v),
                    );
                }
            }
        }

        draw_rectangle_lines(panel.x, panel.y, panel.w, panel.h, 2.0, Color::new(0.4, 0.45, 0.55, 1.0));
        draw_rectangle(clear_btn.x, clear_btn.y, clear_btn.w, clear_btn.h, Color::new(0.2, 0.2, 0.25, 1.0));
        draw_rectangle_lines(clear_btn.x, clear_btn.y, clear_btn.w, clear_btn.h, 1.5, Color::new(0.5, 0.5, 0.6, 1.0));
        draw_text("Clear", clear_btn.x + 36.0, clear_btn.y + 24.0, 22.0, WHITE);

        match &stage {
            Stage::NoData => {
                draw_text("MNIST data not found.", 40.0, 470.0, 24.0, Color::new(1.0, 0.6, 0.4, 1.0));
                draw_text("See README: put the 4 .gz files in ./data,", 40.0, 500.0, 20.0, WHITE);
                draw_text("or run the training command to create cnn_weights.bin.", 40.0, 524.0, 20.0, WHITE);
            }
            Stage::Training(_, status) => {
                draw_text("Training the network…", 40.0, 470.0, 24.0, Color::new(0.5, 0.9, 1.0, 1.0));
                draw_text(status, 40.0, 500.0, 22.0, WHITE);
                draw_text("(first run only, then weights are cached)", 40.0, 528.0, 18.0, Color::new(0.7, 0.7, 0.75, 1.0));
            }
            Stage::Ready => {
                let pred = argmax(&net.probs);
                draw_text("Prediction:", 40.0, 470.0, 24.0, WHITE);
                draw_text(&pred.to_string(), 200.0, 510.0, 90.0, GOLD);

                // бары вероятностей
                for d in 0..10 {
                    let p = net.probs[d];
                    let y = 540.0 + d as f32 * 26.0;
                    draw_text(&d.to_string(), 40.0, y + 16.0, 22.0, WHITE);
                    draw_rectangle(64.0, y, 220.0 * p, 18.0, if d == pred { GOLD } else { Color::new(0.4, 0.6, 0.9, 1.0) });
                }
            }
        }

        if matches!(stage, Stage::Ready) {
            let x0 = 380.0;
            draw_layer("input 28x28", &input, x0, 90.0, 4.0, 0.0);
            draw_layer("conv1 (8 maps)", &net.a_c1, x0, 250.0, 2.0, 4.0);
            draw_layer("pool1 (8)", &net.a_p1, x0, 360.0, 3.0, 6.0);
            draw_layer("conv2 (16 maps)", &net.a_c2, x0, 470.0, 3.0, 4.0);
            draw_layer("pool2 (16)", &net.a_p2, x0, 580.0, 5.0, 4.0);

            // ядра conv1
            draw_text("conv1 filters (3x3)", x0, 672.0, 22.0, Color::new(0.85, 0.85, 0.9, 1.0));
            for oc in 0..net.conv1.out_c {
                let mut maxw = 1e-6f32;
                for i in 0..9 {
                    maxw = maxw.max(net.conv1.w[oc * 9 + i].abs());
                }
                for ky in 0..3 {
                    for kx in 0..3 {
                        let wv = net.conv1.w[oc * 9 + ky * 3 + kx] / maxw;
                        let t = 0.5 + 0.5 * wv;
                        draw_rectangle(x0 + oc as f32 * 36.0 + kx as f32 * 10.0, 684.0 + ky as f32 * 10.0, 10.0, 10.0, inferno(t));
                    }
                }
            }
        }

        let help = "Hold left mouse button to draw.  Each feature map is one channel; brighter = stronger activation.";
        draw_text(help, 40.0, screen_height() - 16.0, 18.0, Color::new(0.7, 0.7, 0.75, 1.0));

        next_frame().await;
    }
}
