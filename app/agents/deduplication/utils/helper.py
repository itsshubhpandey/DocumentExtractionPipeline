from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.agents.deduplication.graph.agent_state import (
    DeduplicationState,
)


# ================================================================
# FILE PATHS
# ================================================================

HELPER_FILE_PATH = Path(__file__).resolve()

# Current file:
# app/agents/deduplication/utils/helper.py
#
# parents[0] -> app/agents/deduplication/utils
# parents[1] -> app/agents/deduplication
DEDUPLICATION_AGENT_DIRECTORY = (
    HELPER_FILE_PATH.parents[1]
)

DEDUPLICATION_INPUT_FILES_DIRECTORY = (
    DEDUPLICATION_AGENT_DIRECTORY
    / "input_files"
)

DEDUPLICATION_CONSTITUTION_PATH = (
    DEDUPLICATION_INPUT_FILES_DIRECTORY
    / "constitution.md"
)

DEDUPLICATION_SPECIFICATION_PATH = (
    DEDUPLICATION_INPUT_FILES_DIRECTORY
    / "specification.md"
)


# ================================================================
# GENERIC INPUT HELPERS
# ================================================================

def convert_to_plain_value(value: Any) -> Any:
    """
    Convert Pydantic models, dataclasses, mappings and sequences
    into normal Python dictionaries and lists.
    """

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")

    if is_dataclass(value):
        return asdict(value)

    if isinstance(value, Mapping):
        return {
            str(key): convert_to_plain_value(item)
            for key, item in value.items()
        }

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            convert_to_plain_value(item)
            for item in value
        ]

    return value


def normalize_requirements(
    raw_requirements: Any,
) -> list[Any]:
    """
    Convert the supplied requirement input into one normal list.
    """

    raw_requirements = convert_to_plain_value(
        raw_requirements
    )

    if raw_requirements is None:
        return []

    if isinstance(raw_requirements, str):
        requirement_text = raw_requirements.strip()

        if not requirement_text:
            return []

        return [requirement_text]

    if isinstance(raw_requirements, Mapping):
        possible_keys = (
            "RawRequirements",
            "raw_requirements",
            "Requirements",
            "requirements",
            "Items",
            "items",
        )

        for key in possible_keys:
            value = raw_requirements.get(key)

            if value is not None:
                return normalize_requirements(value)

        return [dict(raw_requirements)]

    if isinstance(raw_requirements, Sequence) and not isinstance(
        raw_requirements,
        (str, bytes, bytearray),
    ):
        return [
            item
            for item in raw_requirements
            if item is not None and item != ""
        ]

    return [raw_requirements]


def _find_nested_requirement_value(
    requirement: Any,
    keys: Sequence[str],
    *,
    max_depth: int = 4,
) -> Any:
    """
    Find a property at the top level or inside common nested
    requirement wrappers.
    """

    if max_depth < 0:
        return None

    if not isinstance(requirement, Mapping):
        return None

    for key in keys:
        value = requirement.get(key)

        if value is None:
            continue

        if isinstance(value, str):
            if value.strip():
                return value
        else:
            return value

    nested_keys = (
        "Document",
        "document",
        "Requirement",
        "requirement",
        "SourceRequirement",
        "sourceRequirement",
        "Payload",
        "payload",
        "Data",
        "data",
        "Metadata",
        "metadata",
        "Item",
        "item",
    )

    for nested_key in nested_keys:
        nested_value = requirement.get(nested_key)

        if not isinstance(nested_value, Mapping):
            continue

        found_value = _find_nested_requirement_value(
            nested_value,
            keys,
            max_depth=max_depth - 1,
        )

        if found_value is not None:
            return found_value

    return None


def get_requirement_text(
    requirement: Any,
) -> str:
    """
    Extract readable RequirementText from a requirement.
    """

    if isinstance(requirement, str):
        return requirement.strip()

    value = _find_nested_requirement_value(
        requirement,
        (
            "RequirementText",
            "requirement_text",
            "requirementText",
            "Text",
            "text",
            "Description",
            "description",
            "Content",
            "content",
        ),
    )

    if value is not None:
        text = str(value).strip()

        if text:
            return text

    return json.dumps(
        requirement,
        ensure_ascii=False,
        default=str,
        sort_keys=True,
    )


def get_requirement_id(
    requirement: Any,
    index: int,
) -> str:
    """
    Extract RequirementId or generate a temporary ID.
    """

    value = _find_nested_requirement_value(
        requirement,
        (
            "RequirementId",
            "requirement_id",
            "requirementId",
            "Id",
            "id",
            "_id",
        ),
    )

    if value is not None:
        requirement_id = str(value).strip()

        if requirement_id:
            return requirement_id

    return f"REQ-{index:04d}"


def get_requirement_type(
    requirement: Any,
) -> str:
    """
    Extract RequirementType from a requirement.
    """

    value = _find_nested_requirement_value(
        requirement,
        (
            "RequirementType",
            "requirement_type",
            "requirementType",
            "Type",
            "type",
        ),
    )

    return str(value or "").strip()


def build_initial_state(
    raw_requirements: Any,
    context: Mapping[str, Any] | None = None,
    bearer_token: str | None = None,
    correlation_id: str | None = None,
) -> DeduplicationState:
    """
    Create the initial LangGraph state.
    """

    requirements = normalize_requirements(
        raw_requirements
    )

    if not requirements:
        raise ValueError(
            "raw_requirements cannot be empty"
        )

    return DeduplicationState(
        raw_requirements=requirements,
        context=dict(context or {}),
        bearer_token=bearer_token,
        correlation_id=correlation_id,
        prompt="",
        llm_response=None,
        result={},
        error=None,
    )


# ================================================================
# CONSTITUTION AND SPECIFICATION FILES
# ================================================================

@lru_cache(maxsize=4)
def read_required_deduplication_instruction_file(
    file_path: str,
) -> str:
    """
    Read and cache one required instruction file.
    """

    resolved_path = Path(file_path).resolve()

    if not resolved_path.is_file():
        raise FileNotFoundError(
            "Required Requirement Deduplication instruction "
            f"file was not found: {resolved_path}"
        )

    try:
        content = resolved_path.read_text(
            encoding="utf-8"
        ).strip()

    except UnicodeDecodeError as exc:
        raise ValueError(
            "Requirement Deduplication instruction file "
            "must use UTF-8 encoding: "
            f"{resolved_path}"
        ) from exc

    except OSError as exc:
        raise RuntimeError(
            "Unable to read Requirement Deduplication "
            f"instruction file: {resolved_path}. "
            f"Error: {exc}"
        ) from exc

    if not content:
        raise ValueError(
            "Required Requirement Deduplication instruction "
            f"file is empty: {resolved_path}"
        )

    return content


@lru_cache(maxsize=1)
def load_deduplication_instructions() -> tuple[str, str]:
    """
    Load and cache constitution.md and specification.md.
    """

    constitution = (
        read_required_deduplication_instruction_file(
            str(DEDUPLICATION_CONSTITUTION_PATH)
        )
    )

    specification = (
        read_required_deduplication_instruction_file(
            str(DEDUPLICATION_SPECIFICATION_PATH)
        )
    )

    print(
        "Requirement Deduplication instruction files loaded:",
        {
            "constitutionPath": str(
                DEDUPLICATION_CONSTITUTION_PATH
            ),
            "specificationPath": str(
                DEDUPLICATION_SPECIFICATION_PATH
            ),
            "constitutionCharacters": len(
                constitution
            ),
            "specificationCharacters": len(
                specification
            ),
        },
    )

    return constitution, specification


# ================================================================
# PROMPT BUILDING
# ================================================================

def build_deduplication_prompt(
    raw_requirements: Any,
    context: Mapping[str, Any] | None = None,
) -> str:
    """
    Build one Requirement Deduplication prompt.

    Rules and output instructions are loaded from constitution.md
    and specification.md.

    Only RequirementId, RequirementText and RequirementType are
    sent to the LLM.
    """

    del context

    constitution, specification = (
        load_deduplication_instructions()
    )

    requirements = normalize_requirements(
        raw_requirements
    )

    if not requirements:
        raise ValueError(
            "Cannot build Requirement Deduplication prompt "
            "because no requirements were supplied."
        )

    llm_requirements: list[dict[str, str]] = []

    for index, requirement in enumerate(
        requirements,
        start=1,
    ):
        requirement_id = get_requirement_id(
            requirement,
            index,
        )

        requirement_text = get_requirement_text(
            requirement
        )

        requirement_type = get_requirement_type(
            requirement
        )

        if not requirement_id:
            raise ValueError(
                "A requirement is missing RequirementId "
                f"at position {index}."
            )

        if not requirement_text:
            raise ValueError(
                "A requirement is missing RequirementText "
                f"at position {index}. "
                f"RequirementId: {requirement_id}."
            )

        llm_requirements.append(
            {
                "RequirementId": requirement_id,
                "RequirementText": requirement_text,
                "RequirementType": requirement_type,
            }
        )

    requirements_json = json.dumps(
        llm_requirements,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )

    prompt = "\n\n".join(
        [
            constitution.strip(),
            specification.strip(),
            requirements_json,
        ]
    ).strip()

    if not prompt:
        raise ValueError(
            "The final Requirement Deduplication prompt "
            "is empty."
        )

    print(
        "Deduplication prompt prepared:",
        {
            "requirementCount": len(
                llm_requirements
            ),
            "characterCount": len(
                prompt
            ),
            "estimatedTokenCount": round(
                len(prompt) / 4
            ),
            "constitutionCharacters": len(
                constitution
            ),
            "specificationCharacters": len(
                specification
            ),
            "runtimeInputCharacters": len(
                requirements_json
            ),
            "fieldsSentToLlm": [
                "RequirementId",
                "RequirementText",
                "RequirementType",
            ],
            "fullOriginalObjectsIncluded": False,
            "contextIncluded": False,
            "constitutionIncluded": True,
            "specificationIncluded": True,
            "promptRulesStoredInPython": False,
            "oneGoLlmInvocation": True,
        },
    )

    return prompt


# ================================================================
# LLM RESPONSE PARSING
# ================================================================

def extract_response_content(
    llm_response: Any,
) -> Any:
    """
    Extract content from LangChain and plain response objects.
    """

    if llm_response is None:
        raise ValueError(
            "LLM returned no response"
        )

    if isinstance(llm_response, (dict, list)):
        return llm_response

    content = getattr(
        llm_response,
        "content",
        None,
    )

    if content is not None:
        if isinstance(content, list):
            text_parts: list[str] = []

            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)

                elif isinstance(part, Mapping):
                    text = (
                        part.get("text")
                        or part.get("content")
                    )

                    if text is not None:
                        text_parts.append(
                            str(text)
                        )

            return "\n".join(text_parts)

        return content

    return str(llm_response)


def remove_markdown_code_fence(
    text: str,
) -> str:
    """
    Remove an accidental JSON markdown code fence.
    """

    text = text.strip()

    fenced_match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if fenced_match:
        return fenced_match.group(1).strip()

    return text


def decode_json_response(
    text: str,
) -> Any:
    """
    Decode JSON even when the LLM adds surrounding text.
    """

    text = remove_markdown_code_fence(
        text
    )

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()

    for position, character in enumerate(text):
        if character not in "[{":
            continue

        try:
            parsed_value, _ = decoder.raw_decode(
                text[position:]
            )

            return parsed_value

        except json.JSONDecodeError:
            continue

    raise ValueError(
        "LLM response did not contain valid JSON. "
        f"Response preview: {text[:500]}"
    )


def parse_llm_response(
    llm_response: Any,
) -> dict[str, Any]:
    """
    Convert the LLM response into a Python dictionary.
    """

    content = extract_response_content(
        llm_response
    )

    if isinstance(content, Mapping):
        parsed_result = dict(content)

    elif isinstance(content, list):
        parsed_result = {
            "DuplicateGroups": content
        }

    else:
        decoded_value = decode_json_response(
            str(content)
        )

        if not isinstance(
            decoded_value,
            Mapping,
        ):
            raise ValueError(
                "LLM JSON response must be an object"
            )

        parsed_result = dict(decoded_value)

    possible_wrapper_keys = (
        "result",
        "Result",
        "output",
        "Output",
        "data",
        "Data",
    )

    for wrapper_key in possible_wrapper_keys:
        wrapped_result = parsed_result.get(
            wrapper_key
        )

        if isinstance(wrapped_result, Mapping):
            parsed_result = dict(
                wrapped_result
            )
            break

    return parsed_result


def get_first_value(
    mapping: Mapping[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    """
    Return the first matching property value.
    """

    for key in keys:
        if key in mapping:
            return mapping[key]

    return default


def normalize_indexes(
    value: Any,
    total_requirements: int,
) -> list[int]:
    """
    Normalize one-based requirement indexes.
    """

    if value is None:
        return []

    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        value = [value]

    indexes: list[int] = []

    for item in value:
        try:
            index = int(item)

        except (TypeError, ValueError):
            continue

        if not 1 <= index <= total_requirements:
            continue

        if index in indexes:
            continue

        indexes.append(index)

    return indexes


def normalize_requirement_ids(
    value: Any,
    valid_requirement_ids: set[str],
) -> list[str]:
    """
    Normalize RequirementIds returned by the LLM.

    Unknown IDs and repeated IDs are removed.
    """

    if value is None:
        return []

    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        value = [value]

    normalized_ids: list[str] = []

    for item in value:
        requirement_id = str(
            item or ""
        ).strip()

        if not requirement_id:
            continue

        if requirement_id not in valid_requirement_ids:
            continue

        if requirement_id in normalized_ids:
            continue

        normalized_ids.append(
            requirement_id
        )

    return normalized_ids


# ================================================================
# INTENT RESTORATION
# ================================================================

def normalize_output_string_list(
    value: Any,
) -> list[str]:
    """
    Convert a value into a clean unique list of strings.

    Duplicate values are removed case-insensitively while preserving
    the original order and display text.
    """

    if value is None:
        return []

    if isinstance(value, str):
        values: Sequence[Any] = [value]

    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        values = value

    else:
        values = [value]

    normalized_values: list[str] = []
    seen_values: set[str] = set()

    for item in values:
        item_text = str(
            item or ""
        ).strip()

        if not item_text:
            continue

        normalized_key = item_text.casefold()

        if normalized_key in seen_values:
            continue

        seen_values.add(
            normalized_key
        )

        normalized_values.append(
            item_text
        )

    return normalized_values


def collect_requirement_intent_values(
    source_requirements: Sequence[Any],
) -> dict[str, list[str]]:
    """
    Restore and merge CapabilityIntent, EvidenceSections and
    SemanticAnchors from original requirement objects.

    Supports:
    - direct properties;
    - IntentResult;
    - common nested MongoDB wrappers.
    """

    capability_intents: list[str] = []
    evidence_sections: list[str] = []
    semantic_anchors: list[str] = []

    def collect_from_mapping(
        mapping: Mapping[str, Any],
        depth: int = 0,
    ) -> None:
        if depth > 4:
            return

        capability_intents.extend(
            normalize_output_string_list(
                get_first_value(
                    mapping,
                    "CapabilityIntent",
                    "capability_intent",
                    "capabilityIntent",
                    default=[],
                )
            )
        )

        evidence_sections.extend(
            normalize_output_string_list(
                get_first_value(
                    mapping,
                    "EvidenceSections",
                    "evidence_sections",
                    "evidenceSections",
                    default=[],
                )
            )
        )

        semantic_anchors.extend(
            normalize_output_string_list(
                get_first_value(
                    mapping,
                    "SemanticAnchors",
                    "semantic_anchors",
                    "semanticAnchors",
                    default=[],
                )
            )
        )

        intent_result = get_first_value(
            mapping,
            "IntentResult",
            "intentResult",
            "intent_result",
            default=None,
        )

        if isinstance(intent_result, Mapping):
            capability_intents.extend(
                normalize_output_string_list(
                    get_first_value(
                        intent_result,
                        "CapabilityIntent",
                        "capability_intent",
                        "capabilityIntent",
                        default=[],
                    )
                )
            )

            evidence_sections.extend(
                normalize_output_string_list(
                    get_first_value(
                        intent_result,
                        "EvidenceSections",
                        "evidence_sections",
                        "evidenceSections",
                        default=[],
                    )
                )
            )

            semantic_anchors.extend(
                normalize_output_string_list(
                    get_first_value(
                        intent_result,
                        "SemanticAnchors",
                        "semantic_anchors",
                        "semanticAnchors",
                        default=[],
                    )
                )
            )

        nested_mapping_keys = (
            "Document",
            "document",
            "Requirement",
            "requirement",
            "SourceRequirement",
            "sourceRequirement",
            "Payload",
            "payload",
            "Data",
            "data",
            "Metadata",
            "metadata",
            "Item",
            "item",
        )

        for nested_key in nested_mapping_keys:
            nested_value = mapping.get(
                nested_key
            )

            if isinstance(nested_value, Mapping):
                collect_from_mapping(
                    nested_value,
                    depth + 1,
                )

    for source_requirement in source_requirements:
        if isinstance(source_requirement, Mapping):
            collect_from_mapping(
                source_requirement
            )

    return {
        "CapabilityIntent": (
            normalize_output_string_list(
                capability_intents
            )
        ),
        "EvidenceSections": (
            normalize_output_string_list(
                evidence_sections
            )
        ),
        "SemanticAnchors": (
            normalize_output_string_list(
                semantic_anchors
            )
        ),
    }


# ================================================================
# DETERMINISTIC DUPLICATE VALIDATION
# ================================================================

_COMPARISON_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
}

_NEGATION_WORDS = {
    "no",
    "not",
    "never",
    "neither",
    "nor",
    "without",
    "cannot",
    "cant",
    "wont",
    "mustnt",
    "shallnt",
}

_STANDALONE_PREFIXES = (
    "the supplier ",
    "supplier ",
    "the bidder ",
    "bidder ",
    "the contractor ",
    "contractor ",
    "the provider ",
    "provider ",
    "the authority ",
    "the customer ",
    "cadent ",
    "you ",
    "all suppliers ",
    "each supplier ",
)


def normalize_requirement_for_comparison(
    value: Any,
) -> str:
    """
    Normalize requirement text for deterministic comparison.
    """

    text = str(value or "").casefold()

    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("won't", "will not")
        .replace("can't", "cannot")
        .replace("mustn't", "must not")
        .replace("shalln't", "shall not")
    )

    text = re.sub(
        r"[^a-z0-9%£$€\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def _comparison_tokens(
    value: Any,
) -> list[str]:
    """
    Return normalized comparison tokens.
    """

    return normalize_requirement_for_comparison(
        value
    ).split()


def _content_token_set(
    value: Any,
) -> set[str]:
    """
    Return meaningful tokens without common stop words.
    """

    return {
        token
        for token in _comparison_tokens(value)
        if token not in _COMPARISON_STOP_WORDS
    }


def _material_markers(
    value: Any,
) -> set[str]:
    """
    Extract numbers, percentages, money and clause references.
    """

    normalized = normalize_requirement_for_comparison(
        value
    )

    markers = set(
        re.findall(
            r"(?:£|\$|€)?\b\d+(?:\.\d+)?%?\b",
            normalized,
        )
    )

    clause_matches = re.findall(
        r"\b(?:clause|section|schedule|annex|appendix)\s+"
        r"[a-z0-9]+(?:\.[a-z0-9]+)*\b",
        normalized,
    )

    markers.update(
        clause_matches
    )

    return markers


def _negation_signature(
    value: Any,
) -> set[str]:
    """
    Return negation words contained in the requirement.
    """

    return {
        token
        for token in _comparison_tokens(value)
        if token in _NEGATION_WORDS
    }


def _token_jaccard(
    first: set[str],
    second: set[str],
) -> float:
    """
    Calculate Jaccard similarity.
    """

    union = first | second

    if not union:
        return 0.0

    return len(first & second) / len(union)


def _token_containment(
    first: set[str],
    second: set[str],
) -> float:
    """
    Calculate token containment against the smaller set.
    """

    smaller_size = min(
        len(first),
        len(second),
    )

    if not smaller_size:
        return 0.0

    return len(first & second) / smaller_size


def _has_material_tail_conflict(
    first_text: str,
    second_text: str,
) -> bool:
    """
    Detect requirements that share a long opening but have
    materially different ending conditions or triggers.
    """

    first_tokens = _comparison_tokens(
        first_text
    )

    second_tokens = _comparison_tokens(
        second_text
    )

    if not first_tokens or not second_tokens:
        return False

    prefix_length = 0

    for first_token, second_token in zip(
        first_tokens,
        second_tokens,
    ):
        if first_token != second_token:
            break

        prefix_length += 1

    smaller_length = min(
        len(first_tokens),
        len(second_tokens),
    )

    required_prefix_length = max(
        8,
        int(smaller_length * 0.45),
    )

    if prefix_length < required_prefix_length:
        return False

    first_tail = set(
        first_tokens[prefix_length:]
    )

    second_tail = set(
        second_tokens[prefix_length:]
    )

    if len(first_tail) < 3:
        return False

    if len(second_tail) < 3:
        return False

    return (
        _token_jaccard(
            first_tail,
            second_tail,
        )
        < 0.35
    )


def are_high_confidence_duplicates(
    first_text: Any,
    second_text: Any,
) -> bool:
    """
    Return True only when two requirements are high-confidence
    duplicates.

    The comparison intentionally favours keeping requirements
    separate instead of making unsafe false merges.
    """

    first = normalize_requirement_for_comparison(
        first_text
    )

    second = normalize_requirement_for_comparison(
        second_text
    )

    if not first or not second:
        return False

    if first == second:
        return True

    if _material_markers(first) != _material_markers(
        second
    ):
        return False

    if _negation_signature(first) != _negation_signature(
        second
    ):
        return False

    if _has_material_tail_conflict(
        first,
        second,
    ):
        return False

    shorter, longer = sorted(
        (first, second),
        key=len,
    )

    first_tokens = _content_token_set(
        first
    )

    second_tokens = _content_token_set(
        second
    )

    if not first_tokens or not second_tokens:
        return False

    jaccard = _token_jaccard(
        first_tokens,
        second_tokens,
    )

    containment = _token_containment(
        first_tokens,
        second_tokens,
    )

    sequence_similarity = SequenceMatcher(
        None,
        first,
        second,
    ).ratio()

    # Exact shortened version of the same detailed requirement.
    if (
        len(shorter) >= 55
        and shorter in longer
        and containment >= 0.92
    ):
        return True

    # Near-identical wording or formatting variation.
    if (
        sequence_similarity >= 0.95
        and jaccard >= 0.84
    ):
        return True

    # Strong token containment and close sentence structure.
    length_ratio = (
        min(len(first), len(second))
        / max(len(first), len(second))
    )

    if (
        sequence_similarity >= 0.88
        and containment >= 0.94
        and length_ratio >= 0.55
    ):
        return True

    return False


def split_unsafe_duplicate_group(
    requirement_indexes: Sequence[int],
    raw_requirements: Sequence[Any],
) -> list[list[int]]:
    """
    Split an LLM-proposed duplicate group using complete-link
    deterministic validation.
    """

    safe_groups: list[list[int]] = []

    for requirement_index in requirement_indexes:
        zero_based_index = requirement_index - 1

        if not 0 <= zero_based_index < len(
            raw_requirements
        ):
            continue

        requirement_text = get_requirement_text(
            raw_requirements[zero_based_index]
        )

        placed = False

        for safe_group in safe_groups:
            is_compatible = all(
                are_high_confidence_duplicates(
                    requirement_text,
                    get_requirement_text(
                        raw_requirements[
                            existing_index - 1
                        ]
                    ),
                )
                for existing_index in safe_group
            )

            if is_compatible:
                safe_group.append(
                    requirement_index
                )

                placed = True
                break

        if not placed:
            safe_groups.append(
                [requirement_index]
            )

    return safe_groups


def _clusters_are_compatible(
    first_cluster: Sequence[int],
    second_cluster: Sequence[int],
    requirements: Sequence[Any],
) -> bool:
    """
    Require every cross-cluster pair to be a safe duplicate.
    """

    return all(
        are_high_confidence_duplicates(
            get_requirement_text(
                requirements[first_index - 1]
            ),
            get_requirement_text(
                requirements[second_index - 1]
            ),
        )
        for first_index in first_cluster
        for second_index in second_cluster
    )


def merge_safe_final_clusters(
    clusters: Sequence[Sequence[int]],
    requirements: Sequence[Any],
) -> list[list[int]]:
    """
    Merge high-confidence duplicates that the LLM left separate.
    """

    merged_clusters: list[list[int]] = []

    normalized_clusters = sorted(
        (
            list(cluster)
            for cluster in clusters
            if cluster
        ),
        key=lambda cluster: min(cluster),
    )

    for raw_cluster in normalized_clusters:
        cluster = sorted(
            set(raw_cluster)
        )

        matching_cluster: list[int] | None = None

        for existing_cluster in merged_clusters:
            if _clusters_are_compatible(
                existing_cluster,
                cluster,
                requirements,
            ):
                matching_cluster = existing_cluster
                break

        if matching_cluster is None:
            merged_clusters.append(
                cluster
            )

        else:
            matching_cluster.extend(
                cluster
            )

            matching_cluster[:] = sorted(
                set(matching_cluster)
            )

    return sorted(
        merged_clusters,
        key=lambda cluster: min(cluster),
    )


def _is_standalone_requirement(
    text: str,
) -> bool:
    """
    Check whether requirement text appears standalone.
    """

    normalized = normalize_requirement_for_comparison(
        text
    )

    return normalized.startswith(
        _STANDALONE_PREFIXES
    )


def select_canonical_requirement(
    requirement_indexes: Sequence[int],
    requirements: Sequence[Any],
) -> tuple[str, int]:
    """
    Select the safest canonical text from original requirements.

    No new obligation is generated by Python.
    """

    candidates: list[tuple[int, str]] = [
        (
            requirement_index,
            get_requirement_text(
                requirements[
                    requirement_index - 1
                ]
            ).strip(),
        )
        for requirement_index in requirement_indexes
    ]

    if not candidates:
        return "", 0

    def score(
        candidate: tuple[int, str],
    ) -> tuple[int, int, int]:
        _, text = candidate

        standalone_bonus = (
            1
            if _is_standalone_requirement(text)
            else 0
        )

        punctuation_bonus = (
            1
            if text.endswith((".", ";", ":"))
            else 0
        )

        return (
            standalone_bonus,
            punctuation_bonus,
            len(text),
        )

    selected_index, selected_text = max(
        candidates,
        key=score,
    )

    return selected_text, selected_index


# ================================================================
# FINAL RESULT VALIDATION
# ================================================================

def validate_deduplication_result(
    result: Any,
    raw_requirements: Any,
) -> dict[str, Any]:
    """
    Validate the LLM response and build the final safe result.

    Behaviour:
    - each input requirement appears exactly once;
    - unsafe LLM duplicate groups are split;
    - high-confidence missed duplicates are merged;
    - canonical IDs are generated as CR-0001, CR-0002, etc.;
    - summary values are calculated in Python.
    """

    requirements = normalize_requirements(
        raw_requirements
    )

    if not requirements:
        raise ValueError(
            "No raw requirements are available "
            "for validation"
        )

    if not isinstance(result, Mapping):
        raise ValueError(
            "Deduplication result must be a dictionary"
        )

    total_requirements = len(
        requirements
    )

    requirement_id_to_index: dict[str, int] = {}
    requirement_index_to_id: dict[int, str] = {}

    for index, requirement in enumerate(
        requirements,
        start=1,
    ):
        requirement_id = get_requirement_id(
            requirement,
            index,
        )

        if requirement_id in requirement_id_to_index:
            raise ValueError(
                "Duplicate RequirementId found in input: "
                f"{requirement_id}"
            )

        requirement_id_to_index[
            requirement_id
        ] = index

        requirement_index_to_id[
            index
        ] = requirement_id

    valid_requirement_ids = set(
        requirement_id_to_index
    )

    raw_duplicate_groups = get_first_value(
        result,
        "DuplicateGroups",
        "duplicate_groups",
        "Groups",
        "groups",
        default=[],
    )

    if not isinstance(
        raw_duplicate_groups,
        Sequence,
    ) or isinstance(
        raw_duplicate_groups,
        (str, bytes, bytearray),
    ):
        raw_duplicate_groups = []

    accepted_clusters: list[list[int]] = []
    assigned_indexes: set[int] = set()

    rejected_false_merge_members = 0
    proposed_group_count = 0

    for raw_group in raw_duplicate_groups:
        if not isinstance(raw_group, Mapping):
            continue

        requirement_ids = (
            normalize_requirement_ids(
                get_first_value(
                    raw_group,
                    "RequirementIds",
                    "requirement_ids",
                    "SourceRequirementIds",
                    "source_requirement_ids",
                    default=[],
                ),
                valid_requirement_ids,
            )
        )

        requirement_indexes = [
            requirement_id_to_index[
                requirement_id
            ]
            for requirement_id in requirement_ids
        ]

        if not requirement_indexes:
            requirement_indexes = normalize_indexes(
                get_first_value(
                    raw_group,
                    "RequirementIndexes",
                    "requirement_indexes",
                    "SourceRequirementIndexes",
                    "source_requirement_indexes",
                    "Indexes",
                    "indexes",
                    default=[],
                ),
                total_requirements,
            )

        requirement_indexes = [
            index
            for index in requirement_indexes
            if index not in assigned_indexes
        ]

        if len(requirement_indexes) < 2:
            continue

        proposed_group_count += 1

        safe_groups = split_unsafe_duplicate_group(
            requirement_indexes,
            requirements,
        )

        for safe_group in safe_groups:
            if len(safe_group) >= 2:
                accepted_clusters.append(
                    safe_group
                )

                assigned_indexes.update(
                    safe_group
                )

            else:
                rejected_false_merge_members += 1

    requested_unique_ids = (
        normalize_requirement_ids(
            get_first_value(
                result,
                "UniqueRequirementIds",
                "unique_requirement_ids",
                "UniqueIds",
                "unique_ids",
                default=[],
            ),
            valid_requirement_ids,
        )
    )

    requested_unique_indexes = [
        requirement_id_to_index[
            requirement_id
        ]
        for requirement_id in requested_unique_ids
    ]

    if not requested_unique_indexes:
        requested_unique_indexes = (
            normalize_indexes(
                get_first_value(
                    result,
                    "UniqueRequirementIndexes",
                    "unique_requirement_indexes",
                    "UniqueIndexes",
                    "unique_indexes",
                    default=[],
                ),
                total_requirements,
            )
        )

    singleton_indexes: list[int] = []

    for index in requested_unique_indexes:
        if index in assigned_indexes:
            continue

        if index in singleton_indexes:
            continue

        singleton_indexes.append(
            index
        )

    # Automatically include every requirement not assigned by the LLM.
    for index in range(
        1,
        total_requirements + 1,
    ):
        if index in assigned_indexes:
            continue

        if index in singleton_indexes:
            continue

        singleton_indexes.append(
            index
        )

    provisional_clusters: list[list[int]] = [
        sorted(set(cluster))
        for cluster in accepted_clusters
    ]

    provisional_clusters.extend(
        [index]
        for index in singleton_indexes
    )

    final_clusters = merge_safe_final_clusters(
        provisional_clusters,
        requirements,
    )

    represented_indexes = [
        index
        for cluster in final_clusters
        for index in cluster
    ]

    expected_indexes = list(
        range(
            1,
            total_requirements + 1,
        )
    )

    if sorted(represented_indexes) != expected_indexes:
        raise ValueError(
            "Internal deduplication validation failed: "
            "every input requirement must appear exactly once."
        )

    deduplicated_requirements: list[
        dict[str, Any]
    ] = []

    for canonical_number, cluster in enumerate(
        final_clusters,
        start=1,
    ):
        source_requirements = [
            requirements[index - 1]
            for index in cluster
        ]

        (
            canonical_requirement,
            canonical_source_index,
        ) = select_canonical_requirement(
            cluster,
            requirements,
        )

        intent_values = (
            collect_requirement_intent_values(
                source_requirements
            )
        )

        if canonical_source_index:
            canonical_source_requirement = (
                requirements[
                    canonical_source_index - 1
                ]
            )
        else:
            canonical_source_requirement = (
                source_requirements[0]
            )

        deduplicated_requirements.append(
            {
                "CanonicalRequirementId": (
                    f"CR-{canonical_number:04d}"
                ),
                "CanonicalRequirement": (
                    canonical_requirement
                ),
                "RequirementIds": [
                    requirement_index_to_id[index]
                    for index in cluster
                ],
                "RequirementType": (
                    get_requirement_type(
                        canonical_source_requirement
                    )
                ),
                "IntentResult": {
                    "CapabilityIntent": (
                        intent_values[
                            "CapabilityIntent"
                        ]
                    ),
                    "EvidenceSections": (
                        intent_values[
                            "EvidenceSections"
                        ]
                    ),
                    "SemanticAnchors": (
                        intent_values[
                            "SemanticAnchors"
                        ]
                    ),
                },
            }
        )

    total_deduplicated_requirements = len(
        deduplicated_requirements
    )

    duplicates_removed = max(
        0,
        total_requirements
        - total_deduplicated_requirements,
    )

    print(
        "Deterministic deduplication validation completed:",
        {
            "inputRequirementCount": (
                total_requirements
            ),
            "llmProposedDuplicateGroupCount": (
                proposed_group_count
            ),
            "unsafeGroupMembersReturnedToUniquePool": (
                rejected_false_merge_members
            ),
            "acceptedOutputCount": (
                total_deduplicated_requirements
            ),
            "duplicatesRemoved": duplicates_removed,
            "canonicalIdStart": (
                "CR-0001"
                if deduplicated_requirements
                else None
            ),
            "canonicalIdEnd": (
                deduplicated_requirements[-1][
                    "CanonicalRequirementId"
                ]
                if deduplicated_requirements
                else None
            ),
        },
    )

    return {
        "Summary": {
            "TotalInputRequirements": (
                total_requirements
            ),
            "TotalDeduplicatedRequirements": (
                total_deduplicated_requirements
            ),
            "DuplicatesRemoved": (
                duplicates_removed
            ),
        },
        "DeduplicatedRequirements": (
            deduplicated_requirements
        ),
    }


# ================================================================
# FINAL LANGGRAPH RESULT
# ================================================================

def extract_deduplication_result(
    final_state: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """
    Extract the final validated result from LangGraph state.
    """

    if not isinstance(final_state, Mapping):
        raise ValueError(
            "LangGraph returned an invalid final state"
        )

    error = final_state.get(
        "error"
    )

    if error:
        raise RuntimeError(
            str(error)
        )

    result = final_state.get(
        "result"
    )

    if not isinstance(result, Mapping):
        raise ValueError(
            "LangGraph completed without a valid "
            "deduplication result"
        )

    return dict(result)