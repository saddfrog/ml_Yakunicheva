from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2

def test_new_quality_flags():
    """Тест новых эвристик качества данных"""
    
    # Тест 1: Константные колонки
    df_constant = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "constant_col": [5, 5, 5, 5],  # константная колонка
        "normal_col": [1, 2, 3, 4],
        "mixed_col": [1, 2, 1, 2],
    })
    
    summary = summarize_dataset(df_constant)
    missing_df = missing_table(df_constant)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем новую эвристику
    assert "has_constant_columns" in flags
    assert flags["has_constant_columns"] == True
    
    # Тест 2: Высокая кардинальность категориальных признаков
    # Должны быть СТРОКОВЫЕ значения, а не числовые!
    df_high_card = pd.DataFrame({
        "id": list(range(100)),
        "high_card_col": [f"value_{i}" for i in range(100)],  # СТРОКОВЫЕ уникальные значения
        "low_card_col": ["A", "B"] * 50,  # только 2 уникальных значения
        "numeric_col": list(range(100)),  # числовая колонка (не должна учитываться)
    })
    
    summary2 = summarize_dataset(df_high_card)
    missing_df2 = missing_table(df_high_card)
    flags2 = compute_quality_flags(summary2, missing_df2)
    
    # Проверяем вторую новую эвристику
    assert "has_high_cardinality_categoricals" in flags2
    # high_card_col имеет 100 уникальных строковых значений при 100 строках = 100%
    # numeric_col не учитывается, так как числовая
    assert flags2["has_high_cardinality_categoricals"] == True
    
    # Тест 3: Проверка влияния на quality_score
    assert "quality_score" in flags
    assert 0.0 <= flags["quality_score"] <= 1.0
    # При наличии константной колонки score должен быть уменьшен
    assert flags["quality_score"] < 1.0
    
    # Тест 4: Проверка что числовые колонки не учитываются в has_high_cardinality_categoricals
    df_only_numeric = pd.DataFrame({
        "numeric1": list(range(100)),
        "numeric2": list(range(100, 200)),
    })
    
    summary3 = summarize_dataset(df_only_numeric)
    missing_df3 = missing_table(df_only_numeric)
    flags3 = compute_quality_flags(summary3, missing_df3)
    
    # В датасете только числовые колонки - флаг должен быть False
    assert flags3["has_high_cardinality_categoricals"] == False