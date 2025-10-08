#!/usr/bin/env python3
"""
Create simple sample configurations for testing
"""

import numpy as np
import pickle
import os

def create_identity_config_4x4():
    """Create 4x4x4x4 identity configuration in correct format"""
    lattice_dims = [4, 4, 4, 4]
    total_sites = np.prod(lattice_dims)

    # Create identity links: shape (total_sites * 4, 4) where each link is [1,0,0,0]
    links = np.zeros((total_sites * 4, 4))
    links[:, 0] = 1.0  # All links are identity [1,0,0,0]

    # Format: [plaquette, links_array]
    config_data = [1.0, links]

    return config_data

def create_random_config_4x4():
    """Create 4x4x4x4 random configuration"""
    lattice_dims = [4, 4, 4, 4]
    total_sites = np.prod(lattice_dims)

    # Create random SU(2) links
    links = []
    for _ in range(total_sites * 4):
        # Random 4-component vector, normalized to unit length
        vec = np.random.randn(4)
        vec = vec / np.linalg.norm(vec)
        links.append(vec)

    links = np.array(links)

    # Estimate plaquette (rough approximation)
    plaquette = 0.5  # Typical for random config

    config_data = [plaquette, links]
    return config_data

def main():
    """Create sample configurations"""
    print("Creating simple sample configurations...")

    os.makedirs("sample_inputs", exist_ok=True)

    # Identity configuration
    print("1. Identity configuration (free field theory)")
    identity_config = create_identity_config_4x4()
    with open("sample_inputs/identity_4x4x4x4.pkl", "wb") as f:
        pickle.dump(identity_config, f)
    print("   ✓ Saved: sample_inputs/identity_4x4x4x4.pkl")

    # Random configuration
    print("2. Random configuration (strong coupling)")
    random_config = create_random_config_4x4()
    with open("sample_inputs/random_4x4x4x4.pkl", "wb") as f:
        pickle.dump(random_config, f)
    print("   ✓ Saved: sample_inputs/random_4x4x4x4.pkl")

    # Create simple metadata
    metadata = {
        "description": "Simple 4x4x4x4 sample configurations",
        "configurations": {
            "identity": {
                "file": "identity_4x4x4x4.pkl",
                "description": "Free field theory (all links = identity)",
                "plaquette": 1.0
            },
            "random": {
                "file": "random_4x4x4x4.pkl",
                "description": "Random SU(2) links (strong coupling)",
                "plaquette": 0.5
            }
        }
    }

    import json
    with open("sample_inputs/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("   ✓ Saved: sample_inputs/metadata.json")

    print("\nSample configurations ready for testing!")

if __name__ == "__main__":
    main()