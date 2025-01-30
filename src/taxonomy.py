from typing import Dict, List, Set, Optional, TypedDict, Literal, FrozenSet, Mapping
import json
from pathlib import Path
from functools import lru_cache

# Type definitions
MainRole = Literal["Protagonist", "Antagonist", "Innocent"]
FineGrainedRole = Literal[
    "Guardian", "Virtuous", "Martyr", "Rebel", "Peacemaker", "Underdog",
    "Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor",
    "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot",
    "Forgotten", "Exploited", "Victim", "Scapegoat"
]

class RoleTaxonomy(TypedDict):
    main_roles: FrozenSet[MainRole]
    fine_roles: Dict[MainRole, FrozenSet[FineGrainedRole]]
    role_indices: Dict[str, int]  # Combined indices for both main and fine roles
    fine_role_indices: Dict[FineGrainedRole, int]  # Indices for fine roles only
    main_role_indices: Dict[MainRole, int]  # Indices for main roles only

# Constants
TAXONOMY_PATH = Path(__file__).parent / "taxonomy.json"

# Main role constants
PROTAGONIST: MainRole = "Protagonist"
ANTAGONIST: MainRole = "Antagonist"
INNOCENT: MainRole = "Innocent"

# Fine-grained role sets
PROTAGONIST_ROLES: FrozenSet[FineGrainedRole] = frozenset({
    "Guardian", "Virtuous", "Martyr", "Rebel", "Peacemaker", "Underdog"
})
ANTAGONIST_ROLES: FrozenSet[FineGrainedRole] = frozenset({
    "Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor",
    "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot"
})
INNOCENT_ROLES: FrozenSet[FineGrainedRole] = frozenset({
    "Forgotten", "Exploited", "Victim", "Scapegoat"
})

# Base taxonomy structure
BASE_TAXONOMY: Dict[MainRole, FrozenSet[FineGrainedRole]] = {
    PROTAGONIST: PROTAGONIST_ROLES,
    ANTAGONIST: ANTAGONIST_ROLES,
    INNOCENT: INNOCENT_ROLES
}

class TaxonomyError(Exception):
    """Base exception for taxonomy-related errors."""
    pass

class TaxonomyNotFoundError(TaxonomyError):
    """Raised when taxonomy file cannot be found."""
    pass

class TaxonomyFormatError(TaxonomyError):
    """Raised when taxonomy format is invalid."""
    pass

@lru_cache(maxsize=1)
def get_taxonomy() -> RoleTaxonomy:
    """
    Get the complete taxonomy with all role mappings and indices.
    Results are cached for performance.
    """
    main_roles = frozenset(BASE_TAXONOMY.keys())
    
    # Create indices for all roles (both main and fine)
    role_indices: Dict[str, int] = {}
    idx = 0
    
    # First add main roles
    main_role_indices: Dict[MainRole, int] = {}
    for role in sorted(main_roles):
        role_indices[role] = idx
        main_role_indices[role] = idx
        idx += 1
    
    # Then add fine roles
    fine_role_indices: Dict[FineGrainedRole, int] = {}
    for main_role in sorted(main_roles):
        for fine_role in sorted(BASE_TAXONOMY[main_role]):
            if fine_role not in fine_role_indices:
                fine_role_indices[fine_role] = idx
                role_indices[fine_role] = idx
                idx += 1
    
    return RoleTaxonomy(
        main_roles=main_roles,
        fine_roles=BASE_TAXONOMY,
        role_indices=role_indices,
        fine_role_indices=fine_role_indices,
        main_role_indices=main_role_indices
    )

def get_main_roles() -> FrozenSet[MainRole]:
    """Get all available main roles."""
    return get_taxonomy()["main_roles"]

def get_fine_roles(main_role: MainRole) -> FrozenSet[FineGrainedRole]:
    """Get fine-grained roles for a given main role."""
    taxonomy = get_taxonomy()
    if main_role not in taxonomy["fine_roles"]:
        raise ValueError(f"Invalid main role: {main_role}")
    return taxonomy["fine_roles"][main_role]

def get_all_fine_roles() -> FrozenSet[FineGrainedRole]:
    """Get all fine-grained roles across all main roles."""
    taxonomy = get_taxonomy()
    return frozenset().union(*taxonomy["fine_roles"].values())

def get_role_indices() -> Mapping[str, int]:
    """Get indices for all roles (both main and fine-grained)."""
    return get_taxonomy()["role_indices"]

def get_fine_role_indices() -> Mapping[FineGrainedRole, int]:
    """Get indices for fine-grained roles only."""
    return get_taxonomy()["fine_role_indices"]

def get_main_role_indices() -> Mapping[MainRole, int]:
    """Get indices for main roles only."""
    return get_taxonomy()["main_role_indices"]

def get_main_roles_count() -> int:
    """Get the number of main roles."""
    return len(get_main_roles())

def get_fine_roles_count() -> int:
    """Get the total number of fine-grained roles."""
    return len(get_all_fine_roles())

def get_main_role_for_fine_role(fine_role: FineGrainedRole) -> Optional[MainRole]:
    """Find the main role that contains the given fine-grained role."""
    taxonomy = get_taxonomy()
    for main_role, fine_roles in taxonomy["fine_roles"].items():
        if fine_role in fine_roles:
            return main_role
    return None

def validate_roles(main_role: MainRole, fine_roles: List[FineGrainedRole]) -> bool:
    """
    Validate if the main role and fine-grained roles combination is valid.
    
    Args:
        main_role: Main role name
        fine_roles: List of fine-grained role names
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If roles are invalid or if no fine-grained roles provided
    """
    if not fine_roles:
        raise ValueError("At least one fine-grained role must be assigned")

    taxonomy = get_taxonomy()
    if main_role not in taxonomy["fine_roles"]:
        raise ValueError(f"Invalid main role: {main_role}")

    valid_fine_roles = taxonomy["fine_roles"][main_role]
    invalid_roles = [role for role in fine_roles if role not in valid_fine_roles]

    if invalid_roles:
        raise ValueError(
            f"Invalid fine-grained roles for {main_role}: {invalid_roles}\n"
            f"Valid roles are: {sorted(valid_fine_roles)}"
        )
    return True

def is_valid_main_role(role: str) -> bool:
    """Check if a string is a valid main role."""
    return role in get_main_roles()

def is_valid_fine_role(main_role: MainRole, fine_role: str) -> bool:
    """Check if a string is a valid fine-grained role for the given main role."""
    try:
        return fine_role in get_fine_roles(main_role)
    except ValueError:
        return False
