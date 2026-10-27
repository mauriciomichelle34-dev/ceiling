"""Shared limits for image upload, camera receipt and decoding."""
MAX_IMAGE_BYTES = 100 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
IMAGE_FORMATS = frozenset({'JPEG', 'PNG', 'BMP'})
IMAGE_CONTENT_TYPES = frozenset({'application/octet-stream', 'image/jpeg', 'image/png', 'image/bmp', 'image/x-ms-bmp'})
