"""Lossless RGB transport; legacy uncompressed payloads remain readable."""
import base64
import zlib

MAX_IMAGE_BYTES = 32 * 1024 * 1024


def decode_image_bytes(image, expected):
    if not 0 < expected <= MAX_IMAGE_BYTES:
        raise ValueError('image exceeds transport size limit')
    try:
        raw = base64.b64decode(image['data_b64'], validate=True)
        codec = image.get('compression', 'none')
        if codec == 'zlib':
            decoder = zlib.decompressobj()
            raw = decoder.decompress(raw, expected + 1)
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise ValueError('invalid or oversized compressed image')
        elif codec != 'none':
            raise ValueError('unsupported image compression')
    except (KeyError, TypeError, zlib.error) as exc:
        raise ValueError('invalid encoded image') from exc
    if len(raw) != expected:
        raise ValueError(f'image byte length mismatch: expected {expected}, got {len(raw)}')
    return raw


def encode_image_bytes(pixels):
    return base64.b64encode(zlib.compress(pixels.tobytes(order='C'), level=1)).decode('ascii')
