from typing import Dict, List, Set, Optional, TypedDict, Literal
import json
from pathlib import Path
from functools import lru_cache

# Type definitions
MainRole = Literal["Protagonist", "Antagonist", "Innocent"]
FineGrainedRole = Literal["Guardian", "Virtuous", "Martyr", "Rebel", "Peacemaker", "Underdog", "Instigator", "Conspirator", "Tyrant", "Foreign Adversary",
                          "Traitor", "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot", "Forgotten", "Exploited", "Victim", "Scapegoat"]
RoleTaxonomy = Dict[MainRole, List[FineGrainedRole]]


class TaxonomyData(TypedDict):
    ROLE_TAXONOMY: List[RoleTaxonomy]


# Constants
TAXONOMY_PATH = Path(__file__).parent / "taxonomy.json"

# Main role constants
PROTAGONIST: MainRole = "Protagonist"
ANTAGONIST: MainRole = "Antagonist"
INNOCENT: MainRole = "Innocent"

MAIN_ROLES: Set[MainRole] = {PROTAGONIST, ANTAGONIST, INNOCENT}

# Fine-grained role constants
PROTAGONIST_ROLES = {"Guardian", "Virtuous",
                     "Martyr", "Rebel", "Peacemaker", "Underdog"}
ANTAGONIST_ROLES = {
    "Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor",
    "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot"
}
INNOCENT_ROLES = {"Forgotten", "Exploited", "Victim", "Scapegoat"}


class TaxonomyError(Exception):
    """Base exception for taxonomy-related errors."""
    pass


class TaxonomyNotFoundError(TaxonomyError):
    """Raised when taxonomy file cannot be found."""
    pass


class TaxonomyFormatError(TaxonomyError):
    """Raised when taxonomy format is invalid."""
    pass


def validate_taxonomy_structure(data: Dict) -> None:
    """
    Validate the structure of loaded taxonomy data.

    Args:
        data: Dictionary containing taxonomy data

    Raises:
        TaxonomyFormatError: If taxonomy structure is invalid
    """
    if not isinstance(data, dict):
        raise TaxonomyFormatError(
            "Invalid taxonomy format: root must be a dictionary")

    if "ROLE_TAXONOMY" not in data:
        raise TaxonomyFormatError("Missing ROLE_TAXONOMY key in taxonomy")

    if not isinstance(data["ROLE_TAXONOMY"], list):
        raise TaxonomyFormatError("ROLE_TAXONOMY must be a list")

    if not data["ROLE_TAXONOMY"]:
        raise TaxonomyFormatError("ROLE_TAXONOMY cannot be empty")

    # Validate main roles and their fine-grained roles
    taxonomy = data["ROLE_TAXONOMY"][0]
    if not all(role in taxonomy for role in MAIN_ROLES):
        raise TaxonomyFormatError(
            f"Taxonomy must contain all main roles: {MAIN_ROLES}")

    # Validate fine-grained roles match constants
    if set(taxonomy[PROTAGONIST]) != PROTAGONIST_ROLES:
        raise TaxonomyFormatError(
            f"Invalid Protagonist roles. Expected: {PROTAGONIST_ROLES}")
    if set(taxonomy[ANTAGONIST]) != ANTAGONIST_ROLES:
        raise TaxonomyFormatError(
            f"Invalid Antagonist roles. Expected: {ANTAGONIST_ROLES}")
    if set(taxonomy[INNOCENT]) != INNOCENT_ROLES:
        raise TaxonomyFormatError(
            f"Invalid Innocent roles. Expected: {INNOCENT_ROLES}")


@lru_cache(maxsize=1)
def load_taxonomy(filepath: str = str(TAXONOMY_PATH)) -> RoleTaxonomy:
    """
    Load and validate taxonomy JSON. Results are cached for performance.

    Args:
        filepath: Path to taxonomy JSON file

    Returns:
        Dictionary containing role taxonomy

    Raises:
        TaxonomyNotFoundError: If taxonomy file is not found
        TaxonomyFormatError: If taxonomy format is invalid
    """
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
            validate_taxonomy_structure(data)
            return data["ROLE_TAXONOMY"][0]
    except FileNotFoundError:
        raise TaxonomyNotFoundError(f"Taxonomy file not found: {filepath}")
    except json.JSONDecodeError:
        raise TaxonomyFormatError("Invalid JSON format in taxonomy file")
    except Exception as e:
        raise TaxonomyError(f"Error loading taxonomy: {str(e)}")


def get_main_roles(taxonomy: RoleTaxonomy) -> Set[MainRole]:
    """
    Get all available main roles (Protagonist, Antagonist, Innocent).

    Args:
        taxonomy: Loaded role taxonomy

    Returns:
        Set of main role names
    """
    return set(taxonomy.keys())


def get_subroles(taxonomy: RoleTaxonomy, role: MainRole) -> List[FineGrainedRole]:
    """
    Get subroles for given main role.

    Args:
        taxonomy: Loaded role taxonomy
        role: Main role name

    Returns:
        List of subrole names

    Raises:
        ValueError: If role is invalid
    """
    if role not in taxonomy:
        raise ValueError(
            f"Invalid role: {role}. Must be one of: {list(taxonomy.keys())}")
    return taxonomy[role]


def validate_role(taxonomy: RoleTaxonomy, role: MainRole, subroles: List[FineGrainedRole]) -> bool:
    """
    Validate if the role and subroles combination is valid according to taxonomy.

    Args:
        taxonomy: Loaded role taxonomy
        role: Main role name
        subroles: List of subrole names to validate

    Returns:
        True if valid

    Raises:
        ValueError: If role or subroles are invalid
        ValueError: If number of subroles is invalid (must be at least 1)
    """
    if not subroles:
        raise ValueError("At least one fine-grained role must be assigned")

    if role not in taxonomy:
        raise ValueError(
            f"Invalid main role: {role}. Must be one of: {list(taxonomy.keys())}")

    valid_subroles = set(taxonomy[role])
    invalid_subroles = [sr for sr in subroles if sr not in valid_subroles]

    if invalid_subroles:
        raise ValueError(
            f"Invalid subroles for {role}: {invalid_subroles}\n"
            f"Valid subroles are: {list(valid_subroles)}"
        )
    return True


def is_valid_role(taxonomy: RoleTaxonomy, role: str) -> bool:
    """
    Check if role exists in taxonomy without raising exception.

    Args:
        taxonomy: Loaded role taxonomy
        role: Role name to check

    Returns:
        True if role is valid, False otherwise
    """
    return role in taxonomy


def is_valid_subrole(taxonomy: RoleTaxonomy, role: MainRole, subrole: str) -> bool:
    """
    Check if subrole exists for given role without raising exception.

    Args:
        taxonomy: Loaded role taxonomy
        role: Main role name
        subrole: Subrole name to check

    Returns:
        True if subrole is valid for the role, False otherwise
    """
    return role in taxonomy and subrole in taxonomy[role]


def get_all_subroles(taxonomy: RoleTaxonomy) -> Set[str]:
    """
    Get all available subroles across all main roles.

    Args:
        taxonomy: Loaded role taxonomy

    Returns:
        Set of all subrole names
    """
    return {subrole for subroles in taxonomy.values() for subrole in subroles}


def get_role_for_subrole(taxonomy: RoleTaxonomy, subrole: str) -> Optional[MainRole]:
    """
    Find the main role that contains the given subrole.

    Args:
        taxonomy: Loaded role taxonomy
        subrole: Subrole name to look up

    Returns:
        Main role name if found, None otherwise
    """
    for role, subroles in taxonomy.items():
        if subrole in subroles:
            return role
    return None


def get_valid_fine_roles(taxonomy: RoleTaxonomy, main_role: MainRole) -> List[FineGrainedRole]:
    """
    Get valid fine-grained roles for a given main role.
    """
    return taxonomy[main_role]
