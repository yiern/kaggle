"""
Utility functions for handling newly opened stores.

Newly opened stores are those with no sales during training period
but have sales in validation/test periods.
"""

import pandas as pd
from typing import Tuple
import numpy as np


def identify_new_stores(train_df: pd.DataFrame) -> list[int]:
    """
    Identify stores with no sales during training period (newly opened stores).

    Args:
        train_df: Training dataframe with 'store_nbr' and 'sales' columns

    Returns:
        list: Store IDs that have no training sales
    """
    new_stores = []

    for store_id in range(1, 55):
        store_data = train_df[train_df["store_nbr"] == store_id]
        total_sales = store_data["sales"].sum()

        if total_sales == 0:
            new_stores.append(store_id)

    return new_stores


def get_similar_store_mapping(
    new_stores: list[int], train_df: pd.DataFrame
) -> dict[int, int]:
    """
    Find similar established stores for each new store.

    Strategy: Use store cluster information to find the most similar store.
    If no cluster info, use stores with closest store ID.

    Args:
        new_stores: List of new store IDs
        train_df: Training dataframe with store metadata

    Returns:
        dict: {new_store_id: similar_established_store_id}
    """
    mapping = {}

    # Try to use cluster information if available
    if "cluster" in train_df.columns:
        for new_store in new_stores:
            # Find cluster of new store from validation data
            # For now, use simple heuristic: closest store ID
            similar = find_closest_store(new_store, new_stores)
            mapping[new_store] = similar
    else:
        # Fallback: use closest store number
        for new_store in new_stores:
            similar = find_closest_store(new_store, new_stores)
            mapping[new_store] = similar

    return mapping


def find_closest_store(target_store: int, exclude_stores: list[int]) -> int:
    """
    Find the closest store by store ID (simple heuristic).

    Args:
        target_store: Target store ID
        exclude_stores: Stores to exclude from search

    Returns:
        int: Closest store ID
    """
    min_distance = float("inf")
    closest = None

    for store_id in range(1, 55):
        if store_id in exclude_stores:
            continue

        distance = abs(store_id - target_store)
        if distance < min_distance:
            min_distance = distance
            closest = store_id

    return closest


def filter_new_stores_from_df(df: pd.DataFrame, new_stores: list[int]) -> pd.DataFrame:
    """
    Filter out new stores from dataframe.

    Args:
        df: Input dataframe
        new_stores: List of new store IDs to filter out

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    return df[~df["store_nbr"].isin(new_stores)].copy()
