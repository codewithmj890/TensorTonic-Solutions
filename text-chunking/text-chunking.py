def text_chunking(tokens, chunk_size, overlap):
    step = chunk_size - overlap
    n = len(tokens)

    chunks = []
    start = 0

    while start < n:
        chunk = tokens[start:start + chunk_size]
        chunks.append(chunk)

        if start + chunk_size >= n:
            break

        start += step
    return chunks