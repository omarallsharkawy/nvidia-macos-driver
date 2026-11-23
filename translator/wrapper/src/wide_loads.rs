use rspirv::dr::{Instruction, Module, Operand};
use rspirv::spirv::{MemoryAccess, Op, StorageClass};
use std::collections::{BTreeMap, HashMap};

fn id(o: &Operand) -> Option<u32> { if let Operand::IdRef(v) = o { Some(*v) } else { None } }

pub fn hints(m: &Module, defs: &HashMap<u32, Instruction>, sb_types: &HashMap<u32, u32>, value_types: &HashMap<u32, u32>) -> HashMap<u32, u32> {
    let is_u32 = |t: u32| defs.get(&t).is_some_and(|d| d.class.opcode == Op::TypeInt && d.operands.first() == Some(&Operand::LiteralBit32(32)));
    m.functions.iter().flat_map(|f| f.blocks.iter()).flat_map(|b| b.instructions.iter()).filter_map(|i| {
        if i.class.opcode != Op::Load { return None; }
        let pointee = i.operands.first().and_then(id).and_then(|p| value_types.get(&p)).and_then(|t| sb_types.get(t))?;
        match i.operands.as_slice() {
            [_, Operand::MemoryAccess(f), Operand::LiteralBit32(a)] if *f == MemoryAccess::ALIGNED && *a >= 8 && a.is_power_of_two()
                && is_u32(*pointee) && i.result_type == Some(*pointee) => Some((i.result_id?, *a)),
            _ => None,
        }
    }).collect()
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Leaf { Value(u32), Convert(u32, u32) }

#[derive(Default, Debug)]
struct Affine { k: u32, terms: BTreeMap<Leaf, u32> }

struct Defs<'a> { local: &'a HashMap<u32, Instruction>, consts: &'a HashMap<u32, (u32, u32)>, uint: u32 }

impl Defs<'_> {
    fn constant(&self, v: u32) -> Option<u32> { self.consts.get(&v).map(|c| c.1) }
    fn affine(&self, v: u32, scale: u32, out: &mut Affine, budget: &mut u32) -> bool {
        if *budget == 0 { return false; }
        *budget -= 1;
        if let Some(&(ty, k)) = self.consts.get(&v) {
            if ty != self.uint { return false; }
            out.k = out.k.wrapping_add(k.wrapping_mul(scale));
            return true;
        }
        let mut leaf = |l: Leaf| { let c = out.terms.entry(l).or_insert(0); *c = c.wrapping_add(scale); true };
        let Some(d) = self.local.get(&v) else { return leaf(Leaf::Value(v)) };
        if d.result_type != Some(self.uint) { return false; }
        let ops: Vec<u32> = d.operands.iter().filter_map(id).collect();
        match (d.class.opcode, ops.as_slice()) {
            (Op::IAdd, [a, b]) => self.affine(*a, scale, out, budget) && self.affine(*b, scale, out, budget),
            (Op::ISub, [a, b]) => self.affine(*a, scale, out, budget) && self.affine(*b, scale.wrapping_neg(), out, budget),
            (Op::IMul, [a, b]) => match (self.constant(*a), self.constant(*b)) {
                (Some(c), _) => self.affine(*b, scale.wrapping_mul(c), out, budget),
                (_, Some(c)) => self.affine(*a, scale.wrapping_mul(c), out, budget),
                _ => leaf(Leaf::Value(v)),
            },
            (Op::ShiftLeftLogical, [a, b]) => match self.constant(*b) {
                Some(c) if c < 32 => self.affine(*a, scale.wrapping_mul(1u32 << c), out, budget),
                _ => leaf(Leaf::Value(v)),
            },
            (Op::UConvert | Op::SConvert | Op::Bitcast, [x]) => leaf(Leaf::Convert(d.class.opcode as u32, *x)),
            _ => leaf(Leaf::Value(v)),
        }
    }
    fn differ_by(&self, lo: u32, hi: u32, delta: u32) -> bool {
        let (mut a, mut b, mut budget) = (Affine::default(), Affine::default(), 256);
        if !self.affine(lo, 1, &mut a, &mut budget) || !self.affine(hi, 1, &mut b, &mut budget) { return false; }
        for (l, c) in a.terms { let e = b.terms.entry(l).or_insert(0); *e = e.wrapping_sub(c); }
        b.terms.values().all(|c| *c == 0) && b.k.wrapping_sub(a.k) == delta
    }
    fn word_ptr(&self, p: u32) -> Option<(u32, u32, u32)> {
        let d = self.local.get(&p)?;
        if !matches!(d.class.opcode, Op::AccessChain | Op::InBoundsAccessChain) { return None; }
        match d.operands.iter().map(id).collect::<Option<Vec<_>>>()?.as_slice() {
            [root, member, w] if self.constant(*member) == Some(0) => Some((*root, *member, *w)),
            _ => None,
        }
    }
}

fn is_pure_non_memory(w: &Instruction) -> bool {
    match w.class.opcode {
        Op::IAdd | Op::ISub | Op::IMul | Op::UDiv | Op::SDiv | Op::UMod | Op::SMod | Op::SRem | Op::SNegate | Op::Not
        | Op::ShiftLeftLogical | Op::ShiftRightLogical | Op::ShiftRightArithmetic | Op::BitwiseAnd | Op::BitwiseOr | Op::BitwiseXor
        | Op::UConvert | Op::SConvert | Op::FConvert | Op::ConvertUToF | Op::ConvertSToF | Op::ConvertFToU | Op::ConvertFToS | Op::Bitcast
        | Op::ConvertPtrToU | Op::ConvertUToPtr | Op::CompositeExtract | Op::CompositeConstruct | Op::CompositeInsert | Op::VectorShuffle
        | Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain | Op::Select | Op::CopyObject
        | Op::IEqual | Op::INotEqual | Op::ULessThan | Op::ULessThanEqual | Op::UGreaterThan | Op::UGreaterThanEqual
        | Op::SLessThan | Op::SLessThanEqual | Op::SGreaterThan | Op::SGreaterThanEqual | Op::LogicalAnd | Op::LogicalOr | Op::LogicalNot
        | Op::FAdd | Op::FSub | Op::FMul | Op::FDiv | Op::FNegate
        | Op::FOrdEqual | Op::FUnordEqual | Op::FOrdNotEqual | Op::FUnordNotEqual
        | Op::FOrdLessThan | Op::FUnordLessThan | Op::FOrdGreaterThan | Op::FUnordGreaterThan
        | Op::FOrdLessThanEqual | Op::FUnordLessThanEqual | Op::FOrdGreaterThanEqual | Op::FUnordGreaterThanEqual
        | Op::IsNan | Op::IsInf | Op::BitFieldInsert | Op::BitFieldSExtract | Op::BitFieldUExtract
        | Op::BitReverse | Op::BitCount | Op::Dot | Op::VectorTimesScalar => true,
        Op::ExtInst => !matches!(w.operands.get(1), Some(Operand::LiteralExtInstInteger(35 | 51))),
        _ => false,
    }
}

fn is_pure(w: &Instruction) -> bool {
    w.class.opcode == Op::Load || is_pure_non_memory(w)
}

enum Route { Byte(u32), Word }

struct PlanEntry {
    first: usize,
    others: [usize; 3],
    lanes: usize,
    route: Route,
    root: u32,
    member: u32,
    p0: u32,
}

fn find_or_push(types: &mut Vec<Instruction>, op: Op, operands: Vec<Operand>, fresh: &mut impl FnMut() -> u32) -> u32 {
    if let Some(n) = types.iter().find(|t| t.class.opcode == op && t.operands == operands).and_then(|t| t.result_id) { return n; }
    let n = fresh();
    types.push(Instruction::new(op, None, Some(n), operands));
    n
}

fn load_word_ptr(w: &Instruction, d: &Defs<'_>, uint: u32) -> Option<(u32, u32, u32)> {
    if w.class.opcode != Op::Load || w.result_type != Some(uint) { return None; }
    match w.operands.as_slice() {
        [_] => {}
        [_, Operand::MemoryAccess(f), Operand::LiteralBit32(_)] if *f == MemoryAccess::ALIGNED => {}
        _ => return None,
    }
    w.operands.first().and_then(id).and_then(|p| d.word_ptr(p))
}

fn div4_of(d: &Defs<'_>, x: u32) -> Option<u32> {
    let u = d.local.get(&x)?;
    if u.result_type != Some(d.uint) { return None; }
    match (u.class.opcode, u.operands.as_slice()) {
        (Op::UDiv, [Operand::IdRef(b), Operand::IdRef(c)]) if d.constant(*c) == Some(4) => Some(*b),
        (Op::ShiftRightLogical, [Operand::IdRef(b), Operand::IdRef(c)]) if d.constant(*c) == Some(2) => Some(*b),
        _ => None,
    }
}

fn plan_for_head(
    body: &[Instruction],
    i: usize,
    root: u32,
    member: u32,
    p0: u32,
    x0: u32,
    align: u32,
    quad_enabled: bool,
    taken: &std::collections::HashSet<usize>,
    d: &Defs<'_>,
) -> Option<PlanEntry> {
    let b0_opt = div4_of(d, x0);
    let max_k = if quad_enabled && align >= 16 { 3usize } else { 1usize };
    let mut slots: [Option<usize>; 3] = [None; 3];
    let mut chosen: Option<Route> = None;

    for (j, w) in body.iter().enumerate().skip(i + 1).take(96) {
        if !is_pure(w) { break; }
        if taken.contains(&j) { continue; }
        let Some((root1, _, xk)) = load_word_ptr(w, d, d.uint) else { continue };
        if root1 != root { continue; }
        let bk_opt = div4_of(d, xk);
        for k in 1..=max_k {
            if slots[k - 1].is_some() { continue; }
            let matched = match (b0_opt, bk_opt, &chosen) {
                (Some(b0), Some(bk), None | Some(Route::Byte(_)))
                    if d.differ_by(b0, bk, (k * 4) as u32) =>
                {
                    chosen.get_or_insert(Route::Byte(b0));
                    true
                }
                (_, _, None | Some(Route::Word)) if d.differ_by(x0, xk, k as u32) => {
                    chosen.get_or_insert(Route::Word);
                    true
                }
                _ => false,
            };
            if matched {
                slots[k - 1] = Some(j);
                break;
            }
        }
        if (0..max_k).all(|idx| slots[idx].is_some()) { break; }
    }

    let route = chosen?;
    match (quad_enabled && align >= 16, slots[0], slots[1], slots[2]) {
        (true, Some(j1), Some(j2), Some(j3)) => Some(PlanEntry {
            first: i, others: [j1, j2, j3], lanes: 4, route, root, member, p0,
        }),
        (_, Some(j1), _, _) => Some(PlanEntry {
            first: i, others: [j1, 0, 0], lanes: 2, route, root, member, p0,
        }),
        _ => None,
    }
}

pub fn fuse(m: &mut Module, next: &mut u32, uint: u32, ulong: u32, hints: &HashMap<u32, u32>) -> usize {
    if hints.is_empty() { return 0; }
    let quad_enabled = std::env::var_os("NVMTL_NO_QUAD_WORD_LOAD").is_none();
    let mut fresh = || { let n = *next; *next += 1; n };
    let consts: HashMap<u32, (u32, u32)> = m.types_global_values.iter().filter(|i| i.class.opcode == Op::Constant)
        .filter_map(|i| match i.operands.as_slice() { [Operand::LiteralBit32(k)] => Some((i.result_id?, (i.result_type?, *k))), _ => None }).collect();
    let mut vector2: Option<(u32, u32)> = None;
    let mut vector4: Option<(u32, u32)> = None;
    let mut fused = 0;
    for fi in 0..m.functions.len() {
        let local: HashMap<u32, Instruction> = m.functions[fi].blocks.iter().flat_map(|b| b.instructions.iter())
            .filter_map(|i| Some((i.result_id?, i.clone()))).collect();
        let d = Defs { local: &local, consts: &consts, uint };
        for bi in 0..m.functions[fi].blocks.len() {
            let body = &m.functions[fi].blocks[bi].instructions;
            let mut plan: Vec<PlanEntry> = Vec::new();
            let mut taken = std::collections::HashSet::new();
            for (i, w0) in body.iter().enumerate() {
                let Some(align) = w0.result_id.and_then(|r| hints.get(&r).copied()) else { continue };
                if taken.contains(&i) || w0.result_type != Some(uint) { continue; }
                let Some(p0) = w0.operands.first().and_then(id) else { continue };
                let Some((root, member, x0)) = d.word_ptr(p0) else { continue };
                if let Some(entry) = plan_for_head(body, i, root, member, p0, x0, align, quad_enabled, &taken, &d) {
                    taken.insert(entry.first);
                    for &o in &entry.others[..entry.lanes - 1] { taken.insert(o); }
                    plan.push(entry);
                }
            }
            if plan.is_empty() { continue; }
            let need2 = plan.iter().any(|p| p.lanes == 2);
            let need4 = plan.iter().any(|p| p.lanes == 4);
            if need2 && vector2.is_none() {
                let v2 = find_or_push(&mut m.types_global_values, Op::TypeVector, vec![Operand::IdRef(uint), Operand::LiteralBit32(2)], &mut fresh);
                let pv2 = find_or_push(&mut m.types_global_values, Op::TypePointer,
                    vec![Operand::StorageClass(StorageClass::PhysicalStorageBuffer), Operand::IdRef(v2)], &mut fresh);
                vector2 = Some((v2, pv2));
            }
            if need4 && vector4.is_none() {
                let v4 = find_or_push(&mut m.types_global_values, Op::TypeVector, vec![Operand::IdRef(uint), Operand::LiteralBit32(4)], &mut fresh);
                let pv4 = find_or_push(&mut m.types_global_values, Op::TypePointer,
                    vec![Operand::StorageClass(StorageClass::PhysicalStorageBuffer), Operand::IdRef(v4)], &mut fresh);
                vector4 = Some((v4, pv4));
            }
            let body = std::mem::take(&mut m.functions[fi].blocks[bi].instructions);
            let seconds: std::collections::HashSet<usize> = plan.iter()
                .flat_map(|p| p.others[..p.lanes - 1].iter().copied()).collect();
            let mut out = Vec::with_capacity(body.len() + 8 * plan.len());
            for (i, inst) in body.iter().enumerate() {
                if seconds.contains(&i) { continue; }
                let Some(entry) = plan.iter().find(|p| p.first == i) else { out.push(inst.clone()); continue };
                let address = match entry.route {
                    Route::Byte(b0) => {
                        let (p, u, o, a) = (fresh(), fresh(), fresh(), fresh());
                        out.push(Instruction::new(Op::InBoundsAccessChain, local[&entry.p0].result_type, Some(p), vec![Operand::IdRef(entry.root), Operand::IdRef(entry.member), Operand::IdRef(entry.member)]));
                        out.push(Instruction::new(Op::ConvertPtrToU, Some(ulong), Some(u), vec![Operand::IdRef(p)]));
                        out.push(Instruction::new(Op::UConvert, Some(ulong), Some(o), vec![Operand::IdRef(b0)]));
                        out.push(Instruction::new(Op::IAdd, Some(ulong), Some(a), vec![Operand::IdRef(u), Operand::IdRef(o)]));
                        a
                    }
                    Route::Word => {
                        let u = fresh();
                        out.push(Instruction::new(Op::ConvertPtrToU, Some(ulong), Some(u), vec![Operand::IdRef(entry.p0)]));
                        u
                    }
                };
                let (v_ty, pv_ty) = if entry.lanes == 4 { vector4.unwrap() } else { vector2.unwrap() };
                let (q, v) = (fresh(), fresh());
                let align = hints[&inst.result_id.unwrap()];
                out.push(Instruction::new(Op::ConvertUToPtr, Some(pv_ty), Some(q), vec![Operand::IdRef(address)]));
                out.push(Instruction::new(Op::Load, Some(v_ty), Some(v), vec![Operand::IdRef(q), Operand::MemoryAccess(MemoryAccess::ALIGNED), Operand::LiteralBit32(align)]));
                out.push(Instruction::new(Op::CompositeExtract, Some(uint), inst.result_id, vec![Operand::IdRef(v), Operand::LiteralBit32(0)]));
                for (lane, &other_idx) in entry.others[..entry.lanes - 1].iter().enumerate() {
                    out.push(Instruction::new(Op::CompositeExtract, Some(uint), body[other_idx].result_id, vec![Operand::IdRef(v), Operand::LiteralBit32((lane + 1) as u32)]));
                }
                fused += 1;
            }
            m.functions[fi].blocks[bi].instructions = out;
        }
    }
    fused
}

struct StorePlanEntry {
    indices: [usize; 4],
    vals: [u32; 4],
    last: usize,
    lanes: usize,
    align: u32,
    route: Route,
    root: u32,
    member: u32,
    p0: u32,
}

fn store_word_info(
    w: &Instruction,
    d: &Defs<'_>,
    psb_uint_ptrs: &std::collections::HashSet<u32>,
    value_types: &HashMap<u32, u32>,
) -> Option<(u32, u32, u32, u32, u32, u32)> {
    if w.class.opcode != Op::Store { return None; }
    let (p, val, align) = match w.operands.as_slice() {
        [Operand::IdRef(p), Operand::IdRef(val), Operand::MemoryAccess(f), Operand::LiteralBit32(a)]
            if *f == MemoryAccess::ALIGNED => (*p, *val, *a),
        _ => return None,
    };
    if value_types.get(&val) != Some(&d.uint) { return None; }
    if !d.local.get(&p).and_then(|inst| inst.result_type).is_some_and(|t| psb_uint_ptrs.contains(&t)) {
        return None;
    }
    let (root, member, x) = d.word_ptr(p)?;
    Some((p, val, align, root, member, x))
}

#[allow(clippy::too_many_arguments)]
fn plan_store_for_head(
    body: &[Instruction],
    i: usize,
    root: u32,
    member: u32,
    p0: u32,
    v0: u32,
    x0: u32,
    align: u32,
    quad_enabled: bool,
    taken: &std::collections::HashSet<usize>,
    d: &Defs<'_>,
    psb_uint_ptrs: &std::collections::HashSet<u32>,
    value_types: &HashMap<u32, u32>,
) -> Option<StorePlanEntry> {
    let b0_opt = div4_of(d, x0);
    let max_k = if quad_enabled && align >= 16 { 3usize } else { 1usize };
    let mut slots: [Option<(usize, u32)>; 3] = [None; 3];
    let mut chosen: Option<Route> = None;

    for (j, w) in body.iter().enumerate().skip(i + 1).take(96) {
        if w.class.opcode == Op::Store {
            if taken.contains(&j) { break; }
            let Some((_, vk, _, root1, _, xk)) = store_word_info(w, d, psb_uint_ptrs, value_types) else { break };
            if root1 != root { break; }
            let bk_opt = div4_of(d, xk);
            let mut matched_any = false;
            for k in 1..=max_k {
                if slots[k - 1].is_some() { continue; }
                let matched = match (b0_opt, bk_opt, &chosen) {
                    (Some(b0), Some(bk), None | Some(Route::Byte(_)))
                        if d.differ_by(b0, bk, (k * 4) as u32) =>
                    {
                        chosen.get_or_insert(Route::Byte(b0));
                        true
                    }
                    (_, _, None | Some(Route::Word)) if d.differ_by(x0, xk, k as u32) => {
                        chosen.get_or_insert(Route::Word);
                        true
                    }
                    _ => false,
                };
                if matched {
                    slots[k - 1] = Some((j, vk));
                    matched_any = true;
                    break;
                }
            }
            if !matched_any { break; }
            if (0..max_k).all(|idx| slots[idx].is_some()) { break; }
        } else if !is_pure_non_memory(w) {
            break;
        }
    }

    let route = chosen?;
    match (quad_enabled && align >= 16, slots[0], slots[1], slots[2]) {
        (true, Some((j1, v1)), Some((j2, v2)), Some((j3, v3))) => Some(StorePlanEntry {
            indices: [i, j1, j2, j3],
            vals: [v0, v1, v2, v3],
            last: i.max(j1).max(j2).max(j3),
            lanes: 4,
            align: 16,
            route,
            root,
            member,
            p0,
        }),
        (_, Some((j1, v1)), _, _) => Some(StorePlanEntry {
            indices: [i, j1, 0, 0],
            vals: [v0, v1, 0, 0],
            last: i.max(j1),
            lanes: 2,
            align: align.min(8),
            route,
            root,
            member,
            p0,
        }),
        _ => None,
    }
}

pub fn fuse_stores(m: &mut Module, next: &mut u32, uint: u32, ulong: u32) -> usize {
    let quad_enabled = std::env::var_os("NVMTL_NO_QUAD_WORD_LOAD").is_none();
    let mut fresh = || { let n = *next; *next += 1; n };
    let consts: HashMap<u32, (u32, u32)> = m.types_global_values.iter().filter(|i| i.class.opcode == Op::Constant)
        .filter_map(|i| match i.operands.as_slice() { [Operand::LiteralBit32(k)] => Some((i.result_id?, (i.result_type?, *k))), _ => None }).collect();
    let psb_uint_ptrs: std::collections::HashSet<u32> = m.types_global_values.iter()
        .filter(|t| t.class.opcode == Op::TypePointer
            && t.operands.as_slice() == [Operand::StorageClass(StorageClass::PhysicalStorageBuffer), Operand::IdRef(uint)])
        .filter_map(|t| t.result_id)
        .collect();
    if psb_uint_ptrs.is_empty() { return 0; }
    let mut global_value_types: HashMap<u32, u32> = HashMap::new();
    for i in &m.types_global_values {
        if let (Some(id), Some(ty)) = (i.result_id, i.result_type) {
            global_value_types.insert(id, ty);
        }
    }
    let mut vector2: Option<(u32, u32)> = None;
    let mut vector4: Option<(u32, u32)> = None;
    let mut fused = 0;
    for fi in 0..m.functions.len() {
        let local: HashMap<u32, Instruction> = m.functions[fi].blocks.iter().flat_map(|b| b.instructions.iter())
            .filter_map(|i| Some((i.result_id?, i.clone()))).collect();
        let mut value_types = global_value_types.clone();
        for p in &m.functions[fi].parameters {
            if let (Some(id), Some(ty)) = (p.result_id, p.result_type) {
                value_types.insert(id, ty);
            }
        }
        for i in local.values() {
            if let (Some(id), Some(ty)) = (i.result_id, i.result_type) {
                value_types.insert(id, ty);
            }
        }
        let d = Defs { local: &local, consts: &consts, uint };
        for bi in 0..m.functions[fi].blocks.len() {
            let body = &m.functions[fi].blocks[bi].instructions;
            let mut plan: Vec<StorePlanEntry> = Vec::new();
            let mut taken = std::collections::HashSet::new();
            for (i, w0) in body.iter().enumerate() {
                if taken.contains(&i) { continue; }
                let Some((p0, v0, align, root, member, x0)) = store_word_info(w0, &d, &psb_uint_ptrs, &value_types) else { continue };
                if align < 8 || !align.is_power_of_two() { continue; }
                if let Some(entry) = plan_store_for_head(body, i, root, member, p0, v0, x0, align, quad_enabled, &taken, &d, &psb_uint_ptrs, &value_types) {
                    for &idx in &entry.indices[..entry.lanes] { taken.insert(idx); }
                    plan.push(entry);
                }
            }
            if plan.is_empty() { continue; }
            let need2 = plan.iter().any(|p| p.lanes == 2);
            let need4 = plan.iter().any(|p| p.lanes == 4);
            if need2 && vector2.is_none() {
                let v2 = find_or_push(&mut m.types_global_values, Op::TypeVector, vec![Operand::IdRef(uint), Operand::LiteralBit32(2)], &mut fresh);
                let pv2 = find_or_push(&mut m.types_global_values, Op::TypePointer,
                    vec![Operand::StorageClass(StorageClass::PhysicalStorageBuffer), Operand::IdRef(v2)], &mut fresh);
                vector2 = Some((v2, pv2));
            }
            if need4 && vector4.is_none() {
                let v4 = find_or_push(&mut m.types_global_values, Op::TypeVector, vec![Operand::IdRef(uint), Operand::LiteralBit32(4)], &mut fresh);
                let pv4 = find_or_push(&mut m.types_global_values, Op::TypePointer,
                    vec![Operand::StorageClass(StorageClass::PhysicalStorageBuffer), Operand::IdRef(v4)], &mut fresh);
                vector4 = Some((v4, pv4));
            }
            let body = std::mem::take(&mut m.functions[fi].blocks[bi].instructions);
            let dropped: std::collections::HashSet<usize> = plan.iter()
                .flat_map(|p| {
                    let last = p.last;
                    p.indices[..p.lanes].iter().copied().filter(move |&idx| idx != last)
                })
                .collect();
            let mut out = Vec::with_capacity(body.len() + 8 * plan.len());
            for (i, inst) in body.iter().enumerate() {
                if dropped.contains(&i) { continue; }
                let Some(entry) = plan.iter().find(|p| p.last == i) else { out.push(inst.clone()); continue };
                let address = match entry.route {
                    Route::Byte(b0) => {
                        let (p, u, o, a) = (fresh(), fresh(), fresh(), fresh());
                        out.push(Instruction::new(Op::InBoundsAccessChain, local[&entry.p0].result_type, Some(p), vec![Operand::IdRef(entry.root), Operand::IdRef(entry.member), Operand::IdRef(entry.member)]));
                        out.push(Instruction::new(Op::ConvertPtrToU, Some(ulong), Some(u), vec![Operand::IdRef(p)]));
                        out.push(Instruction::new(Op::UConvert, Some(ulong), Some(o), vec![Operand::IdRef(b0)]));
                        out.push(Instruction::new(Op::IAdd, Some(ulong), Some(a), vec![Operand::IdRef(u), Operand::IdRef(o)]));
                        a
                    }
                    Route::Word => {
                        let u = fresh();
                        out.push(Instruction::new(Op::ConvertPtrToU, Some(ulong), Some(u), vec![Operand::IdRef(entry.p0)]));
                        u
                    }
                };
                let (v_ty, pv_ty) = if entry.lanes == 4 { vector4.unwrap() } else { vector2.unwrap() };
                let (q, v) = (fresh(), fresh());
                let vals: Vec<Operand> = entry.vals[..entry.lanes].iter().map(|&val| Operand::IdRef(val)).collect();
                out.push(Instruction::new(Op::CompositeConstruct, Some(v_ty), Some(v), vals));
                out.push(Instruction::new(Op::ConvertUToPtr, Some(pv_ty), Some(q), vec![Operand::IdRef(address)]));
                out.push(Instruction::new(Op::Store, None, None, vec![
                    Operand::IdRef(q),
                    Operand::IdRef(v),
                    Operand::MemoryAccess(MemoryAccess::ALIGNED),
                    Operand::LiteralBit32(entry.align),
                ]));
                fused += 1;
            }
            m.functions[fi].blocks[bi].instructions = out;
        }
    }
    fused
}

#[cfg(test)]
mod tests {
    use crate::buffer_addresses::lower;
    use rspirv::dr::{Module, Operand};
    use rspirv::spirv::{MemoryAccess, Op};
    use spirv_tools::{assembler::Assembler, val::Validator};
    const PAIR: &str = r#"
OpCapability Shader
OpCapability Int64
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %buf %out %gid
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %gid BuiltIn GlobalInvocationId
OpDecorate %arr ArrayStride 4
OpDecorate %block Block
OpMemberDecorate %block 0 Offset 0
OpDecorate %buf DescriptorSet 0
OpDecorate %buf Binding 0
OpDecorate %out DescriptorSet 0
OpDecorate %out Binding 1
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%ulong = OpTypeInt 64 0
%v3 = OpTypeVector %uint 3
%pin = OpTypePointer Input %v3
%gid = OpVariable %pin Input
%zero = OpConstant %uint 0
%one = OpConstant %uint 1
%two = OpConstant %uint 2
%four = OpConstant %uint 4
%eight = OpConstant %uint 8
%arr = OpTypeRuntimeArray %uint
%block = OpTypeStruct %arr
%pb = OpTypePointer StorageBuffer %block
%pu = OpTypePointer StorageBuffer %uint
%buf = OpVariable %pb StorageBuffer
%out = OpVariable %pb StorageBuffer
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%label = OpLabel
%g3 = OpLoad %v3 %gid
%g = OpCompositeExtract %uint %g3 0
%h = OpCompositeExtract %uint %g3 1
%gl = OpUConvert %ulong %g
%c0 = OpUConvert %uint %gl
%m0 = OpIMul %uint %c0 %eight
%b0 = OpIAdd %uint %zero %m0
%w0 = OpUDiv %uint %b0 %four
%p0 = OpInBoundsAccessChain %pu %buf %zero %w0
%x0 = OpLoad %uint %p0 Aligned 8
%c1 = OpUConvert %uint %gl
%m1 = OpIMul %uint %c1 %eight
%b1 = OpIAdd %uint %four %m1
%w1 = OpUDiv %uint %b1 %four
%p1 = OpInBoundsAccessChain %pu %buf %zero %w1
%x1 = OpLoad %uint %p1
%s = OpIAdd %uint %x0 %x1
%po = OpAccessChain %pu %out %zero %g
OpStore %po %s
OpReturn
OpFunctionEnd
"#;
    fn source(text: &str) -> Vec<u8> {
        spirv_tools::assembler::create(Some(spirv_tools::TargetEnv::Vulkan_1_2)).assemble(text,Default::default()).unwrap().as_bytes().to_vec()
    }
    fn validate(b: &[u8]) {
        let w:Vec<u32>=b.chunks_exact(4).map(|w|u32::from_le_bytes(w.try_into().unwrap())).collect();
        spirv_tools::val::create(Some(spirv_tools::TargetEnv::Vulkan_1_2)).validate(&w,None).unwrap();
    }
    fn census(text: &str) -> (Vec<u32>, usize) {
        let input = source(text); validate(&input);
        let output = lower(&input).unwrap(); validate(&output);
        let m: Module = rspirv::dr::load_bytes(&output).unwrap();
        let types: std::collections::HashMap<u32, rspirv::dr::Instruction> =
            m.types_global_values.iter().filter_map(|i| Some((i.result_id?, i.clone()))).collect();
        let vector = |t: Option<u32>| t.and_then(|t| types.get(&t)).is_some_and(|d| d.class.opcode == Op::TypeVector);
        let loads: Vec<_> = m.functions.iter().flat_map(|f| f.blocks.iter()).flat_map(|b| b.instructions.iter())
            .filter(|i| i.class.opcode == Op::Load).collect();
        let wide = loads.iter().filter(|i| vector(i.result_type) && i.operands.get(1) == Some(&Operand::MemoryAccess(MemoryAccess::ALIGNED)))
            .filter_map(|i| match i.operands.get(2) { Some(Operand::LiteralBit32(a)) => Some(*a), _ => None }).filter(|a| *a >= 8).collect();
        // A scalar word may keep the first load's stated alignment of 8. Count scalar uint loads by type, not by
        // alignment 4; otherwise every correctly unmerged hinted pair is misreported as a single load.
        let words = loads.iter().filter(|i| i.result_type.and_then(|t| types.get(&t)).is_some_and(|t|
            t.class.opcode == Op::TypeInt && t.operands == [Operand::LiteralBit32(32), Operand::LiteralBit32(0)])).count();
        (wide, words)
    }
    #[test] fn a_hinted_pair_whose_offsets_differ_by_four_is_one_aligned_8_load() {
        assert_eq!(census(PAIR), (vec![8], 0));
    }
    #[test] fn the_byte_route_leaves_no_divide_feeding_the_wide_load() {
        let m: Module = rspirv::dr::load_bytes(&lower(&source(PAIR)).unwrap()).unwrap();
        let body: Vec<_> = m.functions[0].blocks.iter().flat_map(|b| b.instructions.iter()).collect();
        let at = body.iter().position(|i| i.class.opcode == Op::ConvertUToPtr && body.iter().any(|l| l.class.opcode == Op::Load
            && l.operands.first() == Some(&Operand::IdRef(i.result_id.unwrap())) && l.operands.get(2) == Some(&Operand::LiteralBit32(8)))).unwrap();
        let add = body.iter().find(|i| Some(i.result_id) == body[at].operands.first().map(|o| if let Operand::IdRef(v) = o { Some(*v) } else { None })).unwrap();
        assert_eq!(add.class.opcode, Op::IAdd, "base + u64(byte), not a word pointer");
    }
    #[test] fn a_word_index_pair_one_apart_is_fused_through_the_first_words_pointer() {
        let t = PAIR.replace("%w0 = OpUDiv %uint %b0 %four", "%w0 = OpIAdd %uint %m0 %two")
            .replace("%w1 = OpUDiv %uint %b1 %four", "%w1a = OpIAdd %uint %m1 %two\n%w1 = OpIAdd %uint %w1a %one");
        assert_eq!(census(&t), (vec![8], 0));
    }
    #[test] fn an_unhinted_pair_is_left_as_two_aligned_4_words() {
        assert_eq!(census(&PAIR.replace("%x0 = OpLoad %uint %p0 Aligned 8", "%x0 = OpLoad %uint %p0")), (vec![], 2));
    }
    #[test] fn a_hint_of_4_is_not_a_hint() {
        assert_eq!(census(&PAIR.replace("%p0 Aligned 8", "%p0 Aligned 4")), (vec![], 2));
    }
    #[test] fn offsets_eight_apart_are_not_adjacent_words() {
        assert_eq!(census(&PAIR.replace("%b1 = OpIAdd %uint %four %m1", "%b1 = OpIAdd %uint %eight %m1")), (vec![], 2));
    }
    #[test] fn offsets_over_different_sources_are_not_proved_adjacent() {
        assert_eq!(census(&PAIR.replace("%c1 = OpUConvert %uint %gl", "%c1 = OpCopyObject %uint %h")), (vec![], 2));
    }
    #[test] fn a_store_between_the_words_blocks_the_merge() {
        assert_eq!(census(&PAIR.replace("%c1 = OpUConvert %uint %gl", "OpStore %p0 %zero\n%c1 = OpUConvert %uint %gl")), (vec![], 2));
    }
    #[test] fn words_from_two_buffers_are_not_one_access() {
        assert_eq!(census(&PAIR.replace("%p1 = OpInBoundsAccessChain %pu %buf", "%p1 = OpInBoundsAccessChain %pu %out")), (vec![], 2));
    }
    #[test] fn a_volatile_first_word_is_never_merged() {
        assert_eq!(census(&PAIR.replace("%p0 Aligned 8", "%p0 Volatile|Aligned 8")), (vec![], 2));
    }
    #[test] fn the_affine_prover_is_exact_mod_2_32_for_wrapping_offsets() {
        let t = PAIR.replace("%b0 = OpIAdd %uint %zero %m0", "%b0 = OpISub %uint %m0 %four")
            .replace("%b1 = OpIAdd %uint %four %m1", "%b1 = OpIAdd %uint %zero %m1");
        assert_eq!(census(&t), (vec![8], 0));
    }
    const QUAD: &str = r#"
OpCapability Shader
OpCapability Int64
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %buf %out %gid
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %gid BuiltIn GlobalInvocationId
OpDecorate %arr ArrayStride 4
OpDecorate %block Block
OpMemberDecorate %block 0 Offset 0
OpDecorate %buf DescriptorSet 0
OpDecorate %buf Binding 0
OpDecorate %out DescriptorSet 0
OpDecorate %out Binding 1
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%ulong = OpTypeInt 64 0
%v3 = OpTypeVector %uint 3
%pin = OpTypePointer Input %v3
%gid = OpVariable %pin Input
%zero = OpConstant %uint 0
%one = OpConstant %uint 1
%two = OpConstant %uint 2
%three = OpConstant %uint 3
%four = OpConstant %uint 4
%eight = OpConstant %uint 8
%twelve = OpConstant %uint 12
%sixteen = OpConstant %uint 16
%arr = OpTypeRuntimeArray %uint
%block = OpTypeStruct %arr
%pb = OpTypePointer StorageBuffer %block
%pu = OpTypePointer StorageBuffer %uint
%buf = OpVariable %pb StorageBuffer
%out = OpVariable %pb StorageBuffer
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%label = OpLabel
%g3 = OpLoad %v3 %gid
%g = OpCompositeExtract %uint %g3 0
%base = OpIMul %uint %g %sixteen
%b0 = OpIAdd %uint %zero %base
%w0 = OpUDiv %uint %b0 %four
%p0 = OpInBoundsAccessChain %pu %buf %zero %w0
%x0 = OpLoad %uint %p0 Aligned 16
%b1 = OpIAdd %uint %four %base
%w1 = OpUDiv %uint %b1 %four
%p1 = OpInBoundsAccessChain %pu %buf %zero %w1
%x1 = OpLoad %uint %p1
%b2 = OpIAdd %uint %eight %base
%w2 = OpUDiv %uint %b2 %four
%p2 = OpInBoundsAccessChain %pu %buf %zero %w2
%x2 = OpLoad %uint %p2
%b3 = OpIAdd %uint %twelve %base
%w3 = OpUDiv %uint %b3 %four
%p3 = OpInBoundsAccessChain %pu %buf %zero %w3
%x3 = OpLoad %uint %p3
%s01 = OpIAdd %uint %x0 %x1
%s23 = OpIAdd %uint %x2 %x3
%s = OpIAdd %uint %s01 %s23
%po = OpAccessChain %pu %out %zero %g
OpStore %po %s
OpReturn
OpFunctionEnd
"#;
    fn vector_lanes(text: &str) -> Vec<u32> {
        let input = source(text); validate(&input);
        let output = lower(&input).unwrap(); validate(&output);
        let m: Module = rspirv::dr::load_bytes(&output).unwrap();
        let types: std::collections::HashMap<u32, rspirv::dr::Instruction> =
            m.types_global_values.iter().filter_map(|i| Some((i.result_id?, i.clone()))).collect();
        m.functions.iter().flat_map(|f| f.blocks.iter()).flat_map(|b| b.instructions.iter())
            .filter(|i| i.class.opcode == Op::Load)
            .filter(|i| i.operands.get(1) == Some(&Operand::MemoryAccess(MemoryAccess::ALIGNED)))
            .filter_map(|i| i.result_type.and_then(|t| types.get(&t)))
            .filter(|t| t.class.opcode == Op::TypeVector)
            .filter_map(|t| match t.operands.get(1) { Some(Operand::LiteralBit32(n)) => Some(*n), _ => None })
            .collect()
    }
    #[test] fn quad_aligned16_fuses_to_v4() {
        assert_eq!(census(QUAD), (vec![16], 0));
        assert_eq!(vector_lanes(QUAD), vec![4]);
    }
    #[test] fn quad_aligned8_keeps_v2() {
        let t = QUAD.replace("%p0 Aligned 16", "%p0 Aligned 8")
            .replace("%x2 = OpLoad %uint %p2", "%x2 = OpLoad %uint %p2 Aligned 8");
        assert_eq!(census(&t), (vec![8, 8], 0));
        assert_eq!(vector_lanes(&t), vec![2, 2]);
    }
    #[test] fn quad_word_route_aligned16_fuses_to_v4() {
        let t = QUAD
            .replace("%base = OpIMul %uint %g %sixteen\n%b0 = OpIAdd %uint %zero %base\n%w0 = OpUDiv %uint %b0 %four", "%w0 = OpIMul %uint %g %four")
            .replace("%b1 = OpIAdd %uint %four %base\n%w1 = OpUDiv %uint %b1 %four", "%w1 = OpIAdd %uint %w0 %one")
            .replace("%b2 = OpIAdd %uint %eight %base\n%w2 = OpUDiv %uint %b2 %four", "%w2 = OpIAdd %uint %w0 %two")
            .replace("%b3 = OpIAdd %uint %twelve %base\n%w3 = OpUDiv %uint %b3 %four", "%w3 = OpIAdd %uint %w0 %three");
        assert_eq!(census(&t), (vec![16], 0));
        assert_eq!(vector_lanes(&t), vec![4]);
    }
    #[test] fn shift_right_two_byte_route_fuses_to_v4() {
        let t = QUAD
            .replace("OpUDiv %uint %b0 %four", "OpShiftRightLogical %uint %b0 %two")
            .replace("OpUDiv %uint %b1 %four", "OpShiftRightLogical %uint %b1 %two")
            .replace("OpUDiv %uint %b2 %four", "OpShiftRightLogical %uint %b2 %two")
            .replace("OpUDiv %uint %b3 %four", "OpShiftRightLogical %uint %b3 %two");
        assert_eq!(census(&t), (vec![16], 0));
        assert_eq!(vector_lanes(&t), vec![4]);
    }
    fn store_vector_lanes(text: &str) -> Vec<(u32, u32)> {
        let input = source(text); validate(&input);
        let output = lower(&input).unwrap(); validate(&output);
        let m: Module = rspirv::dr::load_bytes(&output).unwrap();
        let types: std::collections::HashMap<u32, rspirv::dr::Instruction> =
            m.types_global_values.iter().filter_map(|i| Some((i.result_id?, i.clone()))).collect();
        let mut value_types: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for i in m.all_inst_iter() {
            if let (Some(id), Some(ty)) = (i.result_id, i.result_type) { value_types.insert(id, ty); }
        }
        m.functions.iter().flat_map(|f| f.blocks.iter()).flat_map(|b| b.instructions.iter())
            .filter(|i| i.class.opcode == Op::Store)
            .filter_map(|i| match i.operands.as_slice() {
                [_, Operand::IdRef(val), Operand::MemoryAccess(f), Operand::LiteralBit32(a)] if *f == MemoryAccess::ALIGNED => {
                    let ty = types.get(value_types.get(val)?)?;
                    if ty.class.opcode != Op::TypeVector { return None; }
                    match ty.operands.get(1) {
                        Some(Operand::LiteralBit32(n)) => Some((*n, *a)),
                        _ => None,
                    }
                }
                _ => None,
            })
            .collect()
    }
    #[test] fn quad_store_aligned16_fuses_to_v4() {
        let t = QUAD.replace(
            "%s01 = OpIAdd %uint %x0 %x1\n%s23 = OpIAdd %uint %x2 %x3\n%s = OpIAdd %uint %s01 %s23\n%po = OpAccessChain %pu %out %zero %g\nOpStore %po %s",
            "%ow0 = OpIMul %uint %g %four\n%ow1 = OpIAdd %uint %ow0 %one\n%ow2 = OpIAdd %uint %ow0 %two\n%ow3 = OpIAdd %uint %ow0 %three\n%op0 = OpInBoundsAccessChain %pu %out %zero %ow0\nOpStore %op0 %x0 Aligned 16\n%op1 = OpInBoundsAccessChain %pu %out %zero %ow1\nOpStore %op1 %x1\n%op2 = OpInBoundsAccessChain %pu %out %zero %ow2\nOpStore %op2 %x2\n%op3 = OpInBoundsAccessChain %pu %out %zero %ow3\nOpStore %op3 %x3",
        );
        assert_eq!(store_vector_lanes(&t), vec![(4, 16)]);
    }
}
