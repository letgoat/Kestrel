# -*- coding: utf-8 -*-
"""
inspect_gguf.py  -- 完整解析 GGUF 文件，手动 struct.unpack 实现。

GGUF 文件布局（小端序）：
  ┌─────────────────────────────────────────┐
  │  Header  (24 bytes)                     │
  │    magic     char[4]  = "GGUF"          │
  │    version   uint32                     │
  │    n_tensors uint64                     │
  │    n_kv      uint64                     │
  ├─────────────────────────────────────────┤
  │  Metadata KV pairs  × n_kv             │
  │    key        gguf_string               │
  │    value_type uint32  (GGUFValueType)   │
  │    value      <depends on type>         │
  ├─────────────────────────────────────────┤
  │  Tensor Infos  × n_tensors             │
  │    name   gguf_string                   │
  │    n_dims uint32                        │
  │    dims   uint64 × n_dims               │
  │    dtype  uint32  (GGMLType)            │
  │    offset uint64  (相对于数据区起始)    │
  ├─────────────────────────────────────────┤
  │  <padding to ALIGNMENT boundary>        │
  ├─────────────────────────────────────────┤
  │  Tensor Data                            │
  └─────────────────────────────────────────┘
"""

import struct
import sys
from pathlib import Path

# ─── 常量 ────────────────────────────────────────────────────────────────────

MAGIC_EXPECTED  = b"GGUF"
HEADER_FMT      = "<4sIQQ"           # magic(4) version(4) n_tensors(8) n_kv(8)
HEADER_SIZE     = struct.calcsize(HEADER_FMT)   # 24 bytes
GGUF_ALIGNMENT  = 32                 # 数据区默认对齐字节数

# ─── GGUFValueType 枚举 ───────────────────────────────────────────────────────

VALUE_TYPE_NAME = {
    0:  "UINT8",
    1:  "INT8",
    2:  "UINT16",
    3:  "INT16",
    4:  "UINT32",
    5:  "INT32",
    6:  "FLOAT32",
    7:  "BOOL",
    8:  "STRING",
    9:  "ARRAY",
    10: "UINT64",
    11: "INT64",
    12: "FLOAT64",
}

# struct format 对应标量类型（不含 STRING / ARRAY / BOOL）
VALUE_TYPE_FMT = {
    0:  ("<B",  1),   # UINT8
    1:  ("<b",  1),   # INT8
    2:  ("<H",  2),   # UINT16
    3:  ("<h",  2),   # INT16
    4:  ("<I",  4),   # UINT32
    5:  ("<i",  4),   # INT32
    6:  ("<f",  4),   # FLOAT32
    10: ("<Q",  8),   # UINT64
    11: ("<q",  8),   # INT64
    12: ("<d",  8),   # FLOAT64
}

# ─── GGMLType 枚举 ────────────────────────────────────────────────────────────

GGML_TYPE_NAME = {
    0:  "F32",
    1:  "F16",
    2:  "Q4_0",
    3:  "Q4_1",
    6:  "Q5_0",
    7:  "Q5_1",
    8:  "Q8_0",
    9:  "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
}

# ─── BinaryReader：带偏移量追踪的流式读取器 ───────────────────────────────────

class BinaryReader:
    def __init__(self, data: bytes):
        self._data = data
        self._pos  = 0

    @property
    def pos(self) -> int:
        return self._pos

    def read(self, n: int) -> bytes:
        chunk = self._data[self._pos : self._pos + n]
        if len(chunk) < n:
            raise EOFError(f"需要读 {n} 字节，但只剩 {len(chunk)} 字节（偏移 {self._pos}）")
        self._pos += n
        return chunk

    def unpack(self, fmt: str):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.read(size))

    def read_u8(self)  -> int: return self.unpack("<B")[0]
    def read_u16(self) -> int: return self.unpack("<H")[0]
    def read_u32(self) -> int: return self.unpack("<I")[0]
    def read_u64(self) -> int: return self.unpack("<Q")[0]
    def read_i8(self)  -> int: return self.unpack("<b")[0]
    def read_i16(self) -> int: return self.unpack("<h")[0]
    def read_i32(self) -> int: return self.unpack("<i")[0]
    def read_i64(self) -> int: return self.unpack("<q")[0]
    def read_f32(self) -> float: return self.unpack("<f")[0]
    def read_f64(self) -> float: return self.unpack("<d")[0]

    def read_gguf_string(self) -> str:
        """gguf_string = uint64(length) + bytes(length)，UTF-8 编码"""
        length = self.read_u64()
        return self.read(length).decode("utf-8", errors="replace")

    def align(self, alignment: int) -> None:
        """跳过填充字节，使当前偏移对齐到 alignment 的整数倍"""
        remainder = self._pos % alignment
        if remainder != 0:
            self._pos += alignment - remainder

# ─── 值解析 ───────────────────────────────────────────────────────────────────

def read_value(r: BinaryReader, vtype: int):
    """根据 GGUFValueType 读取对应的值，返回 Python 原生对象"""
    if vtype in VALUE_TYPE_FMT:
        fmt, _ = VALUE_TYPE_FMT[vtype]
        return r.unpack(fmt)[0]
    if vtype == 7:    # BOOL
        return bool(r.read_u8())
    if vtype == 8:    # STRING
        return r.read_gguf_string()
    if vtype == 9:    # ARRAY
        elem_type  = r.read_u32()
        elem_count = r.read_u64()
        return [read_value(r, elem_type) for _ in range(elem_count)]
    raise ValueError(f"未知的 GGUFValueType: {vtype}")

def format_value(val) -> str:
    """把值格式化成可读字符串（超长数组截断显示）"""
    if isinstance(val, list):
        if len(val) > 8:
            preview = ", ".join(repr(v) for v in val[:8])
            return f"[{preview}, ... ({len(val)} 项)]"
        return repr(val)
    if isinstance(val, float):
        return f"{val:.6g}"
    return repr(val)

# ─── 主解析函数 ───────────────────────────────────────────────────────────────

def inspect(path: Path, show_tensors: bool = True) -> None:
    file_size = path.stat().st_size
    print(f"文件路径 : {path}")
    print(f"文件大小 : {file_size:,} bytes  ({file_size / 1024 / 1024:.2f} MB)")
    print("=" * 60)

    data = path.read_bytes()
    r = BinaryReader(data)

    # ── 1. Header ─────────────────────────────────────────────────────────────
    magic, version, n_tensors, n_kv = r.unpack(HEADER_FMT)

    if magic != MAGIC_EXPECTED:
        print(f"[错误] Magic 不匹配：期望 {MAGIC_EXPECTED}，实际 {magic}")
        sys.exit(1)

    print(f"[Header]")
    print(f"  Magic     : {magic.decode()}  ✓")
    print(f"  Version   : {version}")
    print(f"  n_tensors : {n_tensors}")
    print(f"  n_kv      : {n_kv}")
    print()

    # ── 2. Metadata KV ────────────────────────────────────────────────────────
    print(f"[Metadata KV]  共 {n_kv} 条")
    print("-" * 60)
    metadata: dict = {}
    for i in range(n_kv):
        key        = r.read_gguf_string()
        vtype      = r.read_u32()
        val        = read_value(r, vtype)
        tname      = VALUE_TYPE_NAME.get(vtype, f"type_{vtype}")
        metadata[key] = val
        print(f"  [{i:3d}] {key}")
        print(f"        type  = {tname}")
        print(f"        value = {format_value(val)}")
    print()

    # ── 3. Tensor Infos ───────────────────────────────────────────────────────
    print(f"[Tensor Infos]  共 {n_tensors} 条")
    print("-" * 60)
    tensors = []
    for i in range(n_tensors):
        name   = r.read_gguf_string()
        n_dims = r.read_u32()
        dims   = [r.read_u64() for _ in range(n_dims)]
        dtype  = r.read_u32()
        offset = r.read_u64()
        dtype_name = GGML_TYPE_NAME.get(dtype, f"type_{dtype}")
        shape_str  = "×".join(str(d) for d in dims)
        tensors.append((name, dims, dtype_name, offset))
        if show_tensors:
            print(f"  [{i:3d}] {name}")
            print(f"        shape  = [{shape_str}]")
            print(f"        dtype  = {dtype_name}")
            print(f"        offset = {offset:,}")
    print()

    # ── 4. 数据区起始偏移 ─────────────────────────────────────────────────────
    r.align(GGUF_ALIGNMENT)
    data_section_start = r.pos

    # 计算数据区大小（文件末尾 - 数据区起始）
    data_section_size = file_size - data_section_start

    print(f"[Summary]")
    print(f"  Header 结束偏移       : {HEADER_SIZE} bytes")
    print(f"  Metadata KV 结束偏移  : {r.pos:,} bytes（对齐前）")
    print(f"  数据区起始偏移        : {data_section_start:,} bytes（{GGUF_ALIGNMENT}字节对齐后）")
    print(f"  数据区大小            : {data_section_size:,} bytes  ({data_section_size / 1024 / 1024:.2f} MB)")
    print(f"  张量总数              : {n_tensors}")
    print(f"  元数据条数            : {n_kv}")

    # 验证最后一个 tensor 的实际数据终点
    if tensors:
        last_name, last_dims, last_dtype, last_offset = tensors[-1]
        print(f"  最后一个 Tensor      : {last_name}  offset={last_offset:,}")

    print("=" * 60)
    print("解析完成。")


# ─── 入口 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="解析并打印 GGUF 文件的完整内容")
    parser.add_argument("file", nargs="?", help="GGUF 文件路径")
    parser.add_argument("--no-tensors", action="store_true", help="不逐条打印 tensor 信息（只打印 summary）")
    args = parser.parse_args()

    if args.file:
        target = Path(args.file)
    else:
        default_dir = Path(__file__).parent.parent / "models"
        candidates  = sorted(default_dir.glob("*.gguf"))
        if not candidates:
            print("用法: python inspect_gguf.py <path/to/model.gguf>")
            sys.exit(1)
        target = candidates[0]
        print(f"未指定文件，自动使用: {target.name}\n")

    if not target.exists():
        print(f"[错误] 文件不存在: {target}")
        sys.exit(1)

    inspect(target, show_tensors=not args.no_tensors)
