def what(file, h=None):
    if h is None:
        if isinstance(file, (str, bytes)):
            try:
                with open(file, 'rb') as f:
                    h = f.read(32)
            except:
                return None
        else:
            location = file.tell()
            h = file.read(32)
            file.seek(location)

    if h.startswith(b'\xff\xd8\xff'):
        return 'jpeg'
    if h.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'png'
    if h.startswith(b'GIF87a') or h.startswith(b'GIF89a'):
        return 'gif'
    if h.startswith(b'RIFF') and h[8:12] == b'WEBP':
        return 'webp'
    return None
