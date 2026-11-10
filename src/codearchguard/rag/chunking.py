
from codearchguard.rag.document import Document, Chunk


def _split_long_text(text: str, max_chars: int):
    """
    Split long text into smaller pieces.
    Prefer splitting at word boundaries.
    Return each piece with its character offsets.
    """

    pieces = []
    start = 0

    while start < len(text):

        end = min(start + max_chars, len(text))

        # Prefer the last whitespace before the limit
        if end < len(text):
            split_position = text.rfind(" ", start, end)

            if split_position > start:
                end = split_position

        piece = text[start:end]

        if piece.strip():
            pieces.append((piece, start, end))

        start = end

        # Skip whitespace between pieces
        while start < len(text) and text[start].isspace():
            start += 1

    return pieces


def _split_into_pieces(text: str, max_chars: int):
    """
    Split text by paragraphs first.
    Split oversized paragraphs when necessary.
    """

    pieces = []
    paragraphs = text.split("\n\n")

    current_offset = 0

    for paragraph in paragraphs:

        paragraph_start = text.find(paragraph, current_offset)
        paragraph_end = paragraph_start + len(paragraph)

        if len(paragraph) <= max_chars:
            if paragraph.strip():
                pieces.append(
                    (paragraph, paragraph_start, paragraph_end)
                )
        else:
            long_pieces = _split_long_text(
                paragraph,
                max_chars
            )

            for piece, start, end in long_pieces:
                pieces.append(
                    (
                        piece,
                        paragraph_start + start,
                        paragraph_start + end
                    )
                )

        current_offset = paragraph_end + 2

    return pieces


def chunk_document(
    document: Document,
    max_chars: int = 1000
) -> list[Chunk]:

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")

    text = document.text.strip()

    if not text:
        return []

    pieces = _split_into_pieces(text, max_chars)

    chunks = []

    for index, (piece, start, end) in enumerate(pieces):

        chunks.append(
            Chunk(
                id=f"{document.id}_chunk_{index}",
                parent_document_id=document.id,
                owner_symbol_id=document.owner_symbol_id,
                source_type=document.source_type,
                source_file=document.source_file,
                start_line=document.start_line,
                end_line=document.end_line,
                chunk_index=index,
                text=piece.strip(),
                start_char=start,
                end_char=end,
                context=document.owner_symbol_id
            )
        )

    return chunks



def chunk_documents(
    documents: list[Document],
    max_chars: int = 1000
) -> list[Chunk]:

    chunks = []

    for document in documents:

        # Code documents stay as complete units
        if document.source_type == "code":

            text = document.text.strip()

            if not text:
                continue

            chunks.append(
                Chunk(
                    id=f"{document.id}_chunk_0",
                    parent_document_id=document.id,
                    owner_symbol_id=document.owner_symbol_id,
                    source_type=document.source_type,
                    source_file=document.source_file,
                    start_line=document.start_line,
                    end_line=document.end_line,
                    chunk_index=0,
                    text=text,
                    start_char=0,
                    end_char=len(document.text),
                    context=document.owner_symbol_id
                )
            )

        # Documentation uses paragraph-based chunking
        else:
            chunks.extend(
                chunk_document(document, max_chars)
            )

    return chunks
