#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
from typing import List, Set, Optional
from pathlib import Path
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def modify_usda_file(input_file, output_file, target_links):
    """
    Modify a USDA file:
    - Set instanceable to false for all visuals.
    - Set instanceable to false for collisions under the specified link.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    modified_lines = []
    current_link = None
    inside_visuals_block = False
    inside_collisions_block = False
    find_target_link = False
    
    # Compile the regular expression
    visuals_start_re = re.compile(r'^\s*def\s+Xform\s+"visuals"\s*\(')
    collisions_start_re = re.compile(r'\s*def\s+Xform\s+"collisions"\s*\(')
    instanceable_re = re.compile(r'^\s*instanceable\s*=\s*true\s*$')
    collisions_instanceable_re = re.compile(r'^\s*instanceable\s*=\s*true\s*$')
    
    for line in lines:
        # Check whether this is an Xform definition line
        if 'def Xform "' in line:
            if visuals_start_re.match(line):
                inside_visuals_block = True
                modified_lines.append(line)
                continue
            elif collisions_start_re.match(line):
                inside_collisions_block = True
                modified_lines.append(line)
                continue
            else:
                # Extract the link name
                current_link = line.split('"')[1]
                
                if current_link in target_links:
                    find_target_link = True
                    inside_collisions_block = False
                    inside_visuals_block = False
                modified_lines.append(line)
                continue
        # Process the visuals section
        if inside_visuals_block:
            if instanceable_re.match(line):
                modified_lines.append('            instanceable = false\n')
            else:
                modified_lines.append(line)
            if line.strip().endswith(')'):
                inside_visuals_block = False
            continue
        # Process the collisions section
        if inside_collisions_block and find_target_link:
            if  instanceable_re.match(line):
                modified_lines.append('            instanceable = false\n')
            else:
                modified_lines.append(line)
            if line.strip().endswith(')'):
                inside_collisions_block = False
                find_target_link = False
            continue
        # Append other lines directly
        else:
            modified_lines.append(line)
            continue

    # Save the modified content
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(modified_lines)

def main():
    """Main entry point."""
    # Usage
    target_links = [
        "right_hand_index_0_link", 
        "right_hand_index_1_link",
        "right_hand_middle_0_link",
        "right_hand_middle_1_link",
        "right_hand_thumb_0_link",
        "right_hand_thumb_1_link",
        "right_hand_thumb_2_link",
        "left_hand_index_0_link", 
        "left_hand_index_1_link",
        "left_hand_middle_0_link",
        "left_hand_middle_1_link",
        "left_hand_thumb_0_link",
        "left_hand_thumb_1_link",
        "left_hand_thumb_2_link",

        
    ]
    
    input_file = "/home/unitree/newDisk/URDF/urdf-to-usd/g1withdex3/g1_29dof_with_hand_rev_1_0.usda"
    output_file = "/home/unitree/newDisk/URDF/urdf-to-usd/g1withdex3/g1_29dof_with_hand_rev_1_0_edit.usda"
    
    try:
        modify_usda_file(input_file, output_file, target_links)
        logger.info(f"File processed and saved to: {output_file}")
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
