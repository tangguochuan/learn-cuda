# PTX mma.m16n8k16 指令说明（.f16 / .bf16 类型）

## 1. 指令语法

```
// f16 类型（A/B 为 .f16）
mma.sync.aligned.m16n8k16.row.col.dtype.f16.f16.ctype  d, a, b, c;
  .ctype = {.f16, .f32}
  .dtype = {.f16, .f32}  // dtype 必须与 ctype 相同

// bf16 类型（A/B 为 .bf16）
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32    d, a, b, c;
  // bf16 只支持 .f32 accumulator
```

**语义**：D = A × B + C，其中 A 是 16×16，B 是 16×8，C/D 是 16×8。

**硬件要求**：sm_80 或更高（Ampere+）。

---

## 2. 操作数寄存器组成

### 矩阵 A（16×16，.f16 / .bf16）

| 操作数 | 寄存器类型 | 元素数 | 说明 |
|--------|-----------|--------|------|
| `a` | 4 个 `.f16x2` 寄存器 | a0–a7（共 8 个元素） | 每个 `.f16x2` 含 2 个 f16/bf16 元素 |

PTX 示例：
```
.reg .f16x2 %Ra<4>;   // Ra0, Ra1, Ra2, Ra3
// a = {Ra0, Ra1, Ra2, Ra3}
```

### 矩阵 B（16×8，.f16 / .bf16）

| 操作数 | 寄存器类型 | 元素数 | 说明 |
|--------|-----------|--------|------|
| `b` | 2 个 `.f16x2` 寄存器 | b0–b3（共 4 个元素） | 每个 `.f16x2` 含 2 个 f16/bf16 元素 |

PTX 示例：
```
.reg .f16x2 %Rb<2>;   // Rb0, Rb1
// b = {Rb0, Rb1}
```

### 矩阵 C / D（16×8，accumulator）

| .ctype/.dtype | 寄存器类型 | 元素数 |
|---------------|-----------|--------|
| `.f16` | 2 个 `.f16x2` 寄存器 | c0–c3（共 4 个元素） |
| `.f32` | 4 个 `.f32` 寄存器 | c0–c3（共 4 个元素） |

PTX 示例（f16 accumulator）：
```
.reg .f16x2 %Rc<2>, %Rd<2>;
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
    {%Rd0, %Rd1},
    {%Ra0, %Ra1, %Ra2, %Ra3},
    {%Rb0, %Rb1},
    {%Rc0, %Rc1};
```

PTX 示例（f32 accumulator）：
```
.reg .f16x2 %Ra<4>, %Rb<2>;
.reg .f32   %Rc<4>, %Rd<4>;
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {%Rd0, %Rd1, %Rd2, %Rd3},
    {%Ra0, %Ra1, %Ra2, %Ra3},
    {%Rb0, %Rb1},
    {%Rc0, %Rc1, %Rc2, %Rc3};
```

---

## 3. 每个线程持有的矩阵元素位置

辅助变量定义：
```
groupID           = %laneid >> 2      // 取值范围 [0, 7]
threadID_in_group = %laneid % 4       // 取值范围 [0, 3]
```

### 矩阵 A（16×16）的 fragment layout

每个线程持有 a0–a7（8 个元素，存于 4 个 f16x2 寄存器）：

| 元素索引 i | 所在寄存器 | row（矩阵行） | col（矩阵列） |
|-----------|-----------|--------------|--------------|
| 0, 1 | Ra0（低/高 half） | groupID | threadID_in_group * 2 + (i & 1) |
| 2, 3 | Ra1（低/高 half） | groupID + 8 | threadID_in_group * 2 + (i & 1) |
| 4, 5 | Ra2（低/高 half） | groupID | threadID_in_group * 2 + (i & 1) + 8 |
| 6, 7 | Ra3（低/高 half） | groupID + 8 | threadID_in_group * 2 + (i & 1) + 8 |

规律总结：
- **row**：`i` 在 {0,1,4,5}（寄存器偶数位置）时 = `groupID`；在 {2,3,6,7}（寄存器奇数位置）时 = `groupID + 8`
- **col**：`i < 4` 时 col 在 [0,7]；`i >= 4` 时 col 在 [8,15]

### 矩阵 B（16×8）的 fragment layout

每个线程持有 b0–b3（4 个元素，存于 2 个 f16x2 寄存器）：

| 元素索引 i | 所在寄存器 | row（矩阵行） | col（矩阵列） |
|-----------|-----------|--------------|--------------|
| 0, 1 | Rb0（低/高 half） | threadID_in_group * 2 + (i & 1) | groupID |
| 2, 3 | Rb1（低/高 half） | threadID_in_group * 2 + (i & 1) + 8 | groupID |

规律总结：
- **row**：`b0,b1` 的 row 在 [0,7]；`b2,b3` 的 row 在 [8,15]
- **col**：始终等于 `groupID`（即由 laneid>>2 决定，范围 [0,7]，对应 B 矩阵 8 列）

### 矩阵 C/D（16×8）的 fragment layout

每个线程持有 c0–c3（4 个元素）：

| 元素索引 i | row（矩阵行） | col（矩阵列） |
|-----------|--------------|--------------|
| 0, 1 | groupID | threadID_in_group * 2 + (i & 1) |
| 2, 3 | groupID + 8 | threadID_in_group * 2 + (i & 1) |

规律总结：
- **row**：`c0,c1` → `groupID`；`c2,c3` → `groupID + 8`
- **col**：`(threadID_in_group * 2) + (i & 0x1)`，范围 [0,7]

---

## 4. 直观图示（以 lane 0 为例）

`lane 0`：groupID=0, threadID_in_group=0

- A: a0@(0,0), a1@(0,1), a2@(8,0), a3@(8,1), a4@(0,8), a5@(0,9), a6@(8,8), a7@(8,9)
- B: b0@(0,0), b1@(1,0), b2@(8,0), b3@(9,0)
- C/D: c0@(0,0), c1@(0,1), c2@(8,0), c3@(8,1)

`lane 4`：groupID=1, threadID_in_group=0

- A: a0@(1,0), a1@(1,1), a2@(9,0), a3@(9,1), a4@(1,8), a5@(1,9), a6@(9,8), a7@(9,9)
- B: b0@(0,1), b1@(1,1), b2@(8,1), b3@(9,1)
- C/D: c0@(1,0), c1@(1,1), c2@(9,0), c3@(9,1)
