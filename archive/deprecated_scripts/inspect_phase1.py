import struct
import zlib
from pathlib import Path
from typing import Dict, Tuple

try:
    import numpy as np
except ImportError:
    np = None

MAT_V5_COMPRESSED = 15
MAT_V5_MATRIX = 14

DTYPE_FORMAT = {
    9: 'd',   # miDOUBLE
    8: 'f',   # miSINGLE
    5: 'i',   # miINT32
    6: 'I',   # miUINT32
    2: 'B',   # miUINT8
}


def padded_length(n: int) -> int:
    return ((n + 7) // 8) * 8


def read_tag(buffer: bytes, pos: int) -> Tuple[int, int]:
    return struct.unpack('<II', buffer[pos:pos + 8])


def decode_data(data_type: int, data_buffer: bytes):
    if data_type not in DTYPE_FORMAT:
        return tuple(data_buffer)
    fmt = DTYPE_FORMAT[data_type]
    item_size = struct.calcsize(fmt)
    count = len(data_buffer) // item_size
    return struct.unpack(f'<{count}{fmt}', data_buffer)


def load_mat_file(mat_path: Path) -> Dict[str, Tuple[Tuple[int, ...], tuple]]:
    with open(mat_path, 'rb') as f:
        f.seek(128)
        raw = f.read()

    variables = {}
    pos = 0
    while pos + 8 <= len(raw):
        dt, nbytes = read_tag(raw, pos)
        pos += 8
        if dt != MAT_V5_COMPRESSED:
            pos += padded_length(nbytes)
            continue

        compressed_block = raw[pos:pos + nbytes]
        pos += nbytes
        payload = zlib.decompress(compressed_block)

        ppos = 0
        while ppos + 8 <= len(payload):
            dt2, nbytes2 = read_tag(payload, ppos)
            ppos += 8
            if dt2 != MAT_V5_MATRIX:
                ppos += padded_length(nbytes2)
                continue

            _, flag_bytes = read_tag(payload, ppos)
            ppos += 8 + padded_length(flag_bytes)

            _, dim_bytes = read_tag(payload, ppos)
            ppos += 8
            dims = struct.unpack('<' + 'i' * (dim_bytes // 4), payload[ppos:ppos + dim_bytes])
            ppos += padded_length(dim_bytes)

            _, name_bytes = read_tag(payload, ppos)
            ppos += 8
            name = payload[ppos:ppos + name_bytes].decode('ascii', errors='replace')
            ppos += padded_length(name_bytes)

            data_type, data_bytes = read_tag(payload, ppos)
            ppos += 8
            data_buffer = payload[ppos:ppos + data_bytes]
            ppos += padded_length(data_bytes)

            data_values = decode_data(data_type, data_buffer)
            variables[name] = (dims, data_values)

            if ppos + 8 <= len(payload):
                next_type, next_bytes = read_tag(payload, ppos)
                if next_type in DTYPE_FORMAT:
                    ppos += 8 + padded_length(next_bytes)

    return variables


def summarize_case(case_path: Path) -> None:
    data = load_mat_file(case_path)
    x_vals = data['X_coords'][1]
    z_vals = data['Z_coords'][1]
    coord_pairs = set(zip(x_vals, z_vals))

    print(f'\n{case_path.name}')
    print('  P_matrix', data['P_matrix'][0], 'shape')
    print('  dt_days unique', sorted(set(data['dt_days'][1])))
    print('  inj_status unique', sorted(set(data['inj_status'][1])))
    print('  prod_status unique', sorted(set(data['prod_status'][1])))
    print('  X_coords unique', len(set(x_vals)))
    print('  Z_coords unique', len(set(z_vals)))
    print('  coordinate pairs', len(coord_pairs))
    print('  X range', min(x_vals), max(x_vals))
    print('  Z range', min(z_vals), max(z_vals))


def main() -> None:
    case_files = sorted(Path('.').glob('Case_*.mat'))
    print('cases', len(case_files))
    for case_path in case_files:
        summarize_case(case_path)

    if np is not None:
        print('\nLoaded dataset file: pressure_surrogate_dataset.npz')
        npz = np.load('pressure_surrogate_dataset.npz')
        print('keys', npz.files)
        for name in ['X_train', 'Y_train', 'X_val', 'Y_val', 'X_test', 'Y_test']:
            print(' ', name, npz[name].shape, npz[name].dtype)
        print('X_train min/max', npz['X_train'].min(), npz['X_train'].max())
        print('Y_train min/max', npz['Y_train'].min(), npz['Y_train'].max())
    else:
        print('\nNumPy unavailable: skipped NPZ dataset inspection.')


if __name__ == '__main__':
    main()
