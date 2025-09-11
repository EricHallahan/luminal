// egraph_debugger.rs  (compact version)

use eframe::{
    egui,
    egui::{Color32, Pos2, Vec2},
};
use egraph_serialize::{ClassId, EGraph as SEGraph, NodeId};
use indexmap::IndexMap;

/* ---------- Entry ---------- */

pub fn display_egraph(se: &SEGraph) {
    display_egraph_with_path(se, &[]);
}

pub fn display_egraph_with_path(se: &SEGraph, dfs_path: &[&NodeId]) {
    let icon = eframe::icon_data::from_png_bytes(include_bytes!("../debugger_icon.png")).unwrap();
    let view = View::from_se_with_path(se, dfs_path);
    let _ = eframe::run_native(
        "Luminal E-Graph Debugger",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1200.0, 900.0])
                .with_icon(icon),
            ..Default::default()
        },
        Box::new(move |_cc| Ok(Box::new(App { v: view }))),
    );
}

/* ---------- Model (no petgraph) ---------- */

#[derive(Clone)]
struct Enode {
    label: String,
    eclass: ClassId,
    child_classes: Vec<ClassId>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    All,
    IROnly,
}

struct View {
    enodes: Vec<Enode>,
    pos: Vec<Pos2>,
    class_members: IndexMap<ClassId, Vec<usize>>,
    cam: Cam,
    dragging: Option<usize>,
    need_fit: bool,
    mode: Mode,
    dfs_edges: Vec<(usize, ClassId)>, // source enode idx -> target class
}

impl View {
    fn from_se_with_path(se: &SEGraph, dfs_path: &[&NodeId]) -> Self {
        let mut id2idx: IndexMap<NodeId, usize> = IndexMap::new();
        let mut enodes = Vec::with_capacity(se.nodes.len());
        let mut class_members: IndexMap<ClassId, Vec<usize>> = IndexMap::new();

        for (i, (nid, n)) in se.nodes.iter().enumerate() {
            id2idx.insert(nid.clone(), i);
            let label = if n.children.is_empty() {
                n.op.clone()
            } else {
                format!("{}({})", n.op, n.children.len())
            };
            enodes.push(Enode {
                label,
                eclass: n.eclass.clone(),
                child_classes: n
                    .children
                    .iter()
                    .filter_map(|cid| se.nodes.get(cid).map(|cn| cn.eclass.clone()))
                    .collect(),
            });
            class_members.entry(n.eclass.clone()).or_default().push(i);
        }

        // Layout: BFS on a tiny bipartite graph (Class -> Enode -> Class)
        let classes: Vec<ClassId> = class_members.keys().cloned().collect();
        let c_index: IndexMap<ClassId, usize> = classes
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, c)| (c, i))
            .collect();
        let c_off = 0usize;
        let e_off = classes.len();
        let tot = classes.len() + enodes.len();
        let mut adj: Vec<Vec<usize>> = vec![vec![]; tot];

        for (c, members) in &class_members {
            let ci = c_index[c] + c_off;
            for &ei in members {
                adj[ci].push(e_off + ei);
            }
        }
        for (ei, en) in enodes.iter().enumerate() {
            let u = e_off + ei;
            for ch in &en.child_classes {
                if let Some(&ci) = c_index.get(ch) {
                    adj[u].push(c_off + ci);
                }
            }
        }

        let mut depth = vec![usize::MAX; tot];
        let mut q = std::collections::VecDeque::new();
        for r in &se.root_eclasses {
            if let Some(&ci) = c_index.get(r) {
                depth[ci] = 0;
                q.push_back(ci);
            }
        }
        if q.is_empty() {
            for ci in 0..classes.len() {
                depth[ci] = 0;
                q.push_back(ci);
                break;
            }
        }
        while let Some(u) = q.pop_front() {
            let d = depth[u];
            for &v in &adj[u] {
                if depth[v] == usize::MAX {
                    depth[v] = d + 1;
                    q.push_back(v);
                }
            }
        }
        let mut pos = vec![Pos2::ZERO; enodes.len()];
        let mut buckets: IndexMap<usize, Vec<usize>> = IndexMap::new();
        for ei in 0..enodes.len() {
            let du = depth[e_off + ei];
            buckets.entry(du.min(10_000)).or_default().push(ei);
        }
        let dx = 340.0;
        let dy = 120.0;
        for (layer, es) in buckets.into_iter().enumerate() {
            let n = es.1.len().max(1) as f32;
            let w = (n - 1.0) * dx;
            let y = (layer as f32) * dy + 80.0;
            for (k, &ei) in es.1.iter().enumerate() {
                let x = (k as f32) * dx - w * 0.5 + 100.0;
                pos[ei] = Pos2::new(x, y);
            }
        }

        use std::collections::HashSet;

        // ---- build DFS overlay edges (with backtrack via remaining-children) ----
        let mut dfs_edges = Vec::new();
        let mut seen: HashSet<(usize, ClassId)> = HashSet::new();

        #[derive(Clone, Copy)]
        struct Frame<'a> {
            id: &'a NodeId,
            remaining: usize,
        }

        let mut stack: Vec<Frame> = Vec::new();

        // helper: number of e-graph children for a node
        let child_count = |id: &NodeId| se.nodes.get(id).map(|n| n.children.len()).unwrap_or(0);

        for &nid in dfs_path {
            // backtrack: discard any finished ancestors
            while let Some(top) = stack.last() {
                if top.remaining == 0 {
                    stack.pop();
                } else {
                    break;
                }
            }

            // if we still have a parent, connect parent -> current
            if let Some(top) = stack.last_mut() {
                if let (Some(&src_idx), Some(child_node)) = (id2idx.get(top.id), se.nodes.get(nid))
                {
                    let key = (src_idx, child_node.eclass.clone());
                    if seen.insert(key.clone()) {
                        dfs_edges.push(key); // enode idx -> target class (matches your draw)
                    }
                }
                // consume one child slot from the parent
                if top.remaining > 0 {
                    top.remaining -= 1;
                }
            }

            // push current node as new frame
            stack.push(Frame {
                id: nid,
                remaining: child_count(nid),
            });
        }

        Self {
            enodes,
            pos,
            class_members,
            cam: Cam::new(),
            dragging: None,
            need_fit: true,
            mode: Mode::All,
            dfs_edges,
        }
    }
}

/* ---------- App ---------- */

struct App {
    v: View,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // toggle IR-only
        if ctx.input(|i| i.key_pressed(egui::Key::D)) {
            self.v.mode = if self.v.mode == Mode::All {
                Mode::IROnly
            } else {
                Mode::All
            };
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Pan: drag • Zoom: ⌘/Ctrl+wheel • Drag nodes • Toggle IR-only: D");
                ui.label(match self.v.mode {
                    Mode::All => "Mode: All",
                    Mode::IROnly => "Mode: IR-only",
                });
                if ui.button("Fit").clicked() {
                    self.v.need_fit = true;
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let (resp, painter) = ui.allocate_painter(ui.available_size(), egui::Sense::drag());
            let origin = resp.rect.min;

            // zoom / pan / drag
            let (mods, raw_scroll, pinch, pointer, drag_delta) = ui.input(|i| {
                (
                    i.modifiers,
                    i.raw_scroll_delta,
                    i.zoom_delta_2d(),
                    i.pointer.clone(),
                    i.pointer.delta(),
                )
            });
            let hovered = resp.hovered();
            if hovered
                && ((mods.command || mods.ctrl) && raw_scroll.y != 0.0
                    || (pinch.y - 1.0).abs() > 1e-3)
            {
                if let Some(cur) = pointer.hover_pos() {
                    let factor = if (pinch.y - 1.0).abs() > 1e-3 {
                        pinch.y
                    } else {
                        (-0.001 * raw_scroll.y).exp()
                    };
                    self.v.cam.zoom_at(origin, cur, factor);
                }
            } else if hovered && raw_scroll != Vec2::ZERO {
                self.v.cam.pan += raw_scroll;
            }

            if let Some(pos) = pointer.interact_pos() {
                if hovered && pointer.primary_pressed() && self.v.dragging.is_none() {
                    let pick2 = 16.0 * 16.0;
                    self.v.dragging = self
                        .v
                        .pos
                        .iter()
                        .enumerate()
                        .map(|(j, &w)| (j, self.v.cam.w2s(origin, w).distance_sq(pos)))
                        .filter(|&(_, d2)| d2 <= pick2)
                        .min_by(|a, b| a.1.total_cmp(&b.1))
                        .map(|(j, _)| j);
                }
                if pointer.primary_down() {
                    if let Some(j) = self.v.dragging {
                        self.v.pos[j] = self.v.cam.s2w(origin, pos);
                    } else if hovered {
                        self.v.cam.pan += drag_delta;
                    }
                } else if pointer.primary_released() {
                    self.v.dragging = None;
                }
            }

            // fit once
            if self.v.need_fit {
                let (min, max) = world_bounds(&self.v);
                if min.x.is_finite() {
                    self.v.cam.fit(origin, resp.rect.size(), min, max, 40.0);
                }
                self.v.need_fit = false;
            }

            // draw class boxes first (screen-space), then edges, then nodes
            let rects = draw_class_boxes(&self.v, &painter, origin);
            draw_edges(&self.v, &painter, origin, &rects, self.v.dragging);
            draw_dfs_edges(&self.v, &painter, origin, &rects); // NEW
            draw_nodes(&self.v, &painter, origin);
        });
    }
}

/* ---------- Draw ---------- */

fn draw_dfs_edges(
    v: &View,
    p: &egui::Painter,
    origin: Pos2,
    rects: &IndexMap<ClassId, egui::Rect>,
) {
    let z = v.cam.zoom;
    let blue = egui::Stroke::new((3.0 * z).clamp(2.0, 6.0), Color32::from_rgb(66, 133, 244));

    for &(src_idx, ref target_cls) in &v.dfs_edges {
        // honor IR-only filter
        if !visible_class(v, &v.enodes[src_idx].eclass) || !visible_class(v, target_cls) {
            continue;
        }
        let from = v.cam.w2s(origin, v.pos[src_idx]);
        if let Some(rect) = rects.get(target_cls) {
            let to = rect_anchor(*rect, from);
            p.line_segment([from, to], blue);
        }
    }
}

fn visible_class(v: &View, cid: &ClassId) -> bool {
    v.mode == Mode::All || is_ir_class(cid)
}

fn draw_class_boxes(v: &View, p: &egui::Painter, origin: Pos2) -> IndexMap<ClassId, egui::Rect> {
    let z = v.cam.zoom;
    let r = (14.0 * z).clamp(6.0, 40.0);
    let label_dy = 22.0 * z;
    let pad = 16.0;
    let mut rects = IndexMap::new();

    for (cid, members) in &v.class_members {
        if !visible_class(v, cid) {
            continue;
        }
        let mut min = Pos2::new(f32::INFINITY, f32::INFINITY);
        let mut max = Pos2::new(f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &ei in members {
            // hide member node if its class filtered (keeps box empty -> skipped)
            if !visible_class(v, &v.enodes[ei].eclass) {
                continue;
            }
            let s = v.cam.w2s(origin, v.pos[ei]);
            min.x = min.x.min(s.x - r);
            min.y = min.y.min(s.y - r - label_dy);
            max.x = max.x.max(s.x + r);
            max.y = max.y.max(s.y + r);
        }
        if !min.x.is_finite() {
            continue;
        } // no visible members
        let rect = egui::Rect::from_min_max(min - egui::vec2(pad, pad), max + egui::vec2(pad, pad));
        let (fill, stroke) = class_style(cid);
        p.rect_filled(rect, 8.0, fill);
        dashed_rect(p, rect, stroke, 10.0, 6.0);
        p.text(
            rect.left_top() + egui::vec2(8.0, 6.0),
            egui::Align2::LEFT_TOP,
            cid.to_string(),
            egui::FontId::monospace(12.0),
            Color32::from_rgba_unmultiplied(
                stroke.color.r(),
                stroke.color.g(),
                stroke.color.b(),
                240,
            ),
        );
        rects.insert(cid.clone(), rect);
    }
    rects
}

fn draw_edges(
    v: &View,
    p: &egui::Painter,
    origin: Pos2,
    rects: &IndexMap<ClassId, egui::Rect>,
    active: Option<usize>,
) {
    let z = v.cam.zoom;
    let normal = egui::Stroke::new((2.0 * z).clamp(1.0, 4.0), Color32::from_gray(180));
    let highlight = egui::Stroke::new(
        (3.0 * z).clamp(2.0, 5.0),
        Color32::from_hex("#7FEE64").unwrap(),
    );

    // active node's class (so we can also highlight incoming edges to this class)
    let active_cls = active.map(|i| &v.enodes[i].eclass);

    for (ei, en) in v.enodes.iter().enumerate() {
        if !visible_class(v, &en.eclass) {
            continue;
        }

        let from = v.cam.w2s(origin, v.pos[ei]);

        for ch in &en.child_classes {
            if !visible_class(v, ch) {
                continue;
            }

            // highlight if (a) edge is from active node (downstream)
            // OR       if (b) edge goes into the active node's class (upstream)
            let stroke = if active == Some(ei) || active_cls.is_some_and(|ac| ac == ch) {
                highlight
            } else {
                normal
            };

            if let Some(rect) = rects.get(ch) {
                let to = rect_anchor(*rect, from);
                p.line_segment([from, to], stroke);
            }
        }
    }
}

fn draw_nodes(v: &View, p: &egui::Painter, origin: Pos2) {
    let z = v.cam.zoom;
    let r = (14.0 * z).clamp(6.0, 40.0);
    let label_dy = 22.0 * z;
    let font = egui::FontId::monospace((14.0 * z).clamp(9.0, 48.0));

    for (i, en) in v.enodes.iter().enumerate() {
        if !visible_class(v, &en.eclass) {
            continue;
        }
        let c = v.cam.w2s(origin, v.pos[i]);
        let color = Color32::from_hex("#55a042").unwrap();
        p.circle_filled(
            c,
            r,
            if v.dragging == Some(i) {
                Color32::from_hex("#7FEE64").unwrap()
            } else {
                color
            },
        );
        p.text(
            c + Vec2::new(0.0, -label_dy),
            egui::Align2::CENTER_CENTER,
            &en.label,
            font.clone(),
            Color32::from_hex("#7FEE64").unwrap(),
        );
    }
}

/* ---------- Helpers ---------- */

fn world_bounds(v: &View) -> (Pos2, Pos2) {
    const R: f32 = 14.0;
    const DY: f32 = 22.0;
    let mut min = Pos2::new(f32::INFINITY, f32::INFINITY);
    let mut max = Pos2::new(f32::NEG_INFINITY, f32::NEG_INFINITY);
    for i in 0..v.enodes.len() {
        let p = v.pos[i];
        let lmin = Pos2::new(p.x - 60.0, p.y - DY - 10.0); // rough label pad (cheap & fast)
        let lmax = Pos2::new(p.x + 60.0, p.y + 10.0);
        let nmin = Pos2::new(p.x - R, p.y - R);
        let nmax = Pos2::new(p.x + R, p.y + R);
        min.x = min.x.min(lmin.x.min(nmin.x));
        min.y = min.y.min(lmin.y.min(nmin.y));
        max.x = max.x.max(lmax.x.max(nmax.x));
        max.y = max.y.max(lmax.y.max(nmax.y));
    }
    if !min.x.is_finite() {
        (Pos2::ZERO, Pos2::new(400.0, 300.0))
    } else {
        (min, max)
    }
}

fn dashed_rect(p: &egui::Painter, r: egui::Rect, stroke: egui::Stroke, dash: f32, gap: f32) {
    let draw = |a: Pos2, b: Pos2| dashed(p, a, b, stroke, dash, gap);
    draw(r.left_top(), r.right_top());
    draw(r.right_top(), r.right_bottom());
    draw(r.right_bottom(), r.left_bottom());
    draw(r.left_bottom(), r.left_top());
}
fn dashed(p: &egui::Painter, a: Pos2, b: Pos2, s: egui::Stroke, dash: f32, gap: f32) {
    let v = b - a;
    let len = v.length().max(1.0);
    let dir = v / len;
    let mut t = 0.0;
    let mut x = a;
    while t < len {
        let seg = (t + dash).min(len);
        let y = a + dir * seg;
        p.line_segment([x, y], s);
        t = seg + gap;
        x = a + dir * t;
    }
}

fn rect_anchor(rect: egui::Rect, from: Pos2) -> Pos2 {
    let c = rect.center();
    let dx = from.x - c.x;
    let dy = from.y - c.y;
    if dx == 0.0 && dy == 0.0 {
        return c;
    }
    let hx = rect.width() * 0.5;
    let hy = rect.height() * 0.5;
    let sx = if dx != 0.0 {
        hx / dx.abs()
    } else {
        f32::INFINITY
    };
    let sy = if dy != 0.0 {
        hy / dy.abs()
    } else {
        f32::INFINITY
    };
    Pos2::new(c.x + dx * sx.min(sy), c.y + dy * sx.min(sy))
}

fn is_ir_class(cid: &ClassId) -> bool {
    let s = cid.to_string().to_ascii_uppercase();
    s.starts_with("IR-") || s.starts_with("IR_") || s.contains("IR-")
}

fn class_style(cid: &ClassId) -> (Color32, egui::Stroke) {
    let base = if is_ir_class(cid) {
        Color32::from_rgb(40, 178, 233)
    } else {
        Color32::from_rgb(245, 158, 11)
    };
    let fill = Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), 22);
    let stroke = egui::Stroke::new(
        2.0,
        Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), 200),
    );
    (fill, stroke)
}

/* ---------- Camera ---------- */

#[derive(Clone, Copy)]
struct Cam {
    zoom: f32,
    pan: Vec2,
}
impl Cam {
    fn new() -> Self {
        Self {
            zoom: 1.0,
            pan: Vec2::ZERO,
        }
    }
    #[inline]
    fn w2s(&self, o: Pos2, w: Pos2) -> Pos2 {
        o + (self.pan + w.to_vec2() * self.zoom)
    }
    #[inline]
    fn s2w(&self, o: Pos2, s: Pos2) -> Pos2 {
        (((s - o) - self.pan) / self.zoom).to_pos2()
    }
    fn zoom_at(&mut self, o: Pos2, cur: Pos2, f: f32) {
        let wb = self.s2w(o, cur);
        let nz = (self.zoom * f).clamp(0.1, 10.0);
        self.pan = (cur - o) - wb.to_vec2() * nz;
        self.zoom = nz;
    }
    fn fit(&mut self, _o: Pos2, vp: Vec2, min: Pos2, max: Pos2, m: f32) {
        let sz = (max - min).max(Vec2::splat(1.0));
        let usable = (vp - Vec2::splat(2.0 * m)).max(Vec2::splat(1.0));
        self.zoom = (usable.x / sz.x).min(usable.y / sz.y).clamp(0.1, 10.0);
        let mapped_min = Vec2::splat(m);
        let mapped_sz = sz * self.zoom;
        let extra = (usable - mapped_sz).max(Vec2::ZERO) * 0.5;
        self.pan = mapped_min + extra - min.to_vec2() * self.zoom;
    }
}
