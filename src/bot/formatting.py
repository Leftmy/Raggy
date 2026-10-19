from aiogram.enums import MessageEntityType
from aiogram.types import MessageEntity


def code_entities_for_values(text: str, values: list[str]) -> list[MessageEntity]:
    matches = []
    for value in sorted({value for value in values if value}, key=len, reverse=True):
        start = 0
        while (index := text.find(value, start)) != -1:
            matches.append((index, index + len(value)))
            start = index + len(value)

    entities = []
    occupied_until = -1
    for start, end in sorted(matches):
        if start < occupied_until:
            continue
        entities.append(
            MessageEntity(
                type=MessageEntityType.CODE,
                offset=len(text[:start].encode("utf-16-le")) // 2,
                length=len(text[start:end].encode("utf-16-le")) // 2,
            )
        )
        occupied_until = end
    return entities
