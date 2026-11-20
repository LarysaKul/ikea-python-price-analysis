import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 100)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 7)
plt.rcParams["font.size"] = 11

URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-11-03/ikea.csv"
FILE_NAME = "ikea.csv"
OUTPUT_DIRECTORY = Path("ikea_final_project_results")
FIGURES_DIRECTORY = OUTPUT_DIRECTORY / "figures"
RANDOM_STATE = 42
ALPHA = 0.05


def compact_number(value, position=None):
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def currency_number(value, position=None):
    return f"{value:,.0f}"


def print_markdown(title, frame, index=False, floatfmt=".2f"):
    print(f"\n## {title}")
    print(frame.to_markdown(index=index, floatfmt=floatfmt))


def save_plot(file_name):
    plt.tight_layout()
    plt.savefig(FIGURES_DIRECTORY / file_name, dpi=300, bbox_inches="tight")
    plt.close()


def create_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def clean_designer(value):
    if pd.isna(value):
        return "Unknown"

    value = str(value).strip()
    if not value or value[0].isdigit():
        return "Unknown"

    designers = [designer.strip() for designer in value.split("/") if designer.strip()]
    if not designers:
        return "Unknown"

    cleaned_value = "/".join(sorted(set(designers)))
    if len(cleaned_value) > 100:
        return "Unknown"
    return cleaned_value


def permutation_median_test(first_group, second_group, iterations=5000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    first_group = np.asarray(first_group, dtype=float)
    second_group = np.asarray(second_group, dtype=float)
    observed_difference = np.median(first_group) - np.median(second_group)
    combined = np.concatenate([first_group, second_group])
    extreme_results = 0

    for _ in range(iterations):
        permuted = rng.permutation(combined)
        first_permuted = permuted[: len(first_group)]
        second_permuted = permuted[len(first_group) :]
        difference = np.median(first_permuted) - np.median(second_permuted)
        if abs(difference) >= abs(observed_difference):
            extreme_results += 1

    p_value = (extreme_results + 1) / (iterations + 1)
    return observed_difference, p_value


def bootstrap_median_difference(first_group, second_group, iterations=5000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    first_group = np.asarray(first_group, dtype=float)
    second_group = np.asarray(second_group, dtype=float)
    differences = np.empty(iterations)
    first_medians = np.empty(iterations)
    second_medians = np.empty(iterations)

    for iteration in range(iterations):
        first_sample = rng.choice(first_group, size=len(first_group), replace=True)
        second_sample = rng.choice(second_group, size=len(second_group), replace=True)
        first_medians[iteration] = np.median(first_sample)
        second_medians[iteration] = np.median(second_sample)
        differences[iteration] = first_medians[iteration] - second_medians[iteration]

    confidence_interval = np.percentile(differences, [2.5, 97.5])
    return confidence_interval, differences, first_medians, second_medians


def download_data():
    if os.path.exists(FILE_NAME):
        print(f"Файл {FILE_NAME} уже існує. Завантаження не потрібне.")
        return

    response = requests.get(URL, timeout=60)
    print(f"HTTP status_code: {response.status_code}")

    if response.status_code != 200:
        raise RuntimeError(f"Не вдалося завантажити дані. Код відповіді: {response.status_code}")

    with open(FILE_NAME, "wb") as file:
        file.write(response.content)

    print(f"Файл {FILE_NAME} успішно завантажено.")


def clean_data(raw_data):
    data = raw_data.copy()
    unnamed_columns = [column for column in data.columns if column.startswith("Unnamed")]
    data = data.drop(columns=unnamed_columns).copy()

    full_duplicates = int(data.duplicated().sum())
    duplicated_item_rows = int(data.duplicated(subset=["item_id"]).sum())
    duplicated_item_ids = int(
        data.loc[data.duplicated(subset=["item_id"], keep=False), "item_id"].nunique()
    )
    duplicated_across_categories = int(
        data.loc[data.duplicated(subset=["item_id"], keep=False)]
        .groupby("item_id")["category"]
        .nunique()
        .gt(1)
        .sum()
    )

    rows_before_item_cleaning = len(data)
    data = data.drop_duplicates(subset=["item_id"], keep="first").copy()
    rows_after_item_cleaning = len(data)

    extracted_old_price = (
        data["old_price"]
        .astype("string")
        .str.extract(r"(\d[\d,]*(?:\.\d+)?)", expand=False)
        .str.replace(",", "", regex=False)
    )

    data.loc[:, "old_price_numeric"] = pd.to_numeric(
        extracted_old_price, errors="coerce"
    ).astype(float)
    data.loc[:, "has_old_price"] = data["old_price_numeric"].notna()
    data.loc[:, "discount_amount"] = data["old_price_numeric"] - data["price"]
    data.loc[:, "discount_percent"] = np.where(
        data["old_price_numeric"].gt(0),
        data["discount_amount"] / data["old_price_numeric"] * 100,
        np.nan,
    )

    original_designer = data["designer"].copy()
    data.loc[:, "designer_clean"] = data["designer"].apply(clean_designer)
    invalid_designer_rows = int(
        (data["designer_clean"].eq("Unknown") & original_designer.notna()).sum()
    )

    data.loc[:, "category"] = data["category"].astype("string").fillna("Unknown")
    data.loc[:, "name"] = data["name"].astype("string").fillna("Unknown")
    data.loc[:, "other_colors_binary"] = data["other_colors"].map({"Yes": 1, "No": 0})
    data.loc[:, "description_word_count"] = (
        data["short_description"]
        .astype("string")
        .str.split()
        .str.len()
        .fillna(0)
        .astype(int)
    )
    data.loc[:, "dimension_count"] = data[["depth", "height", "width"]].notna().sum(axis=1)
    data.loc[:, "volume_cm3"] = data[["depth", "height", "width"]].prod(axis=1, min_count=3)
    data.loc[:, "log_price"] = np.log1p(data["price"])
    data.loc[:, "log_volume"] = np.log1p(data["volume_cm3"])

    q1 = data["price"].quantile(0.25)
    q3 = data["price"].quantile(0.75)
    iqr = q3 - q1
    upper_outlier_limit = q3 + 1.5 * iqr
    price_outliers = int(data["price"].gt(upper_outlier_limit).sum())

    cleaning_summary = pd.DataFrame(
        {
            "Показник": [
                "Рядків у початковому наборі",
                "Повних дублікатів без службового стовпця",
                "Повторних рядків за item_id",
                "Унікальних item_id, які повторювалися",
                "item_id, що зустрічалися у різних категоріях",
                "Рядків після видалення дублікатів item_id",
                "Видалених рядків за item_id",
                "Некоректних значень designer",
                "Цінових викидів за правилом 1.5 IQR",
            ],
            "Значення": [
                len(raw_data),
                full_duplicates,
                duplicated_item_rows,
                duplicated_item_ids,
                duplicated_across_categories,
                rows_after_item_cleaning,
                rows_before_item_cleaning - rows_after_item_cleaning,
                invalid_designer_rows,
                price_outliers,
            ],
        }
    )

    return data, cleaning_summary


def create_eda(data):
    missing_summary = (
        data.isna()
        .sum()
        .rename("missing_values")
        .to_frame()
        .assign(missing_percent=lambda frame: frame["missing_values"] / len(data) * 100)
        .query("missing_values > 0")
        .sort_values("missing_values", ascending=False)
        .reset_index(names="column")
    )

    numeric_columns = ["price", "old_price_numeric", "depth", "height", "width", "volume_cm3"]
    numeric_summary = pd.DataFrame(
        {
            "count": data[numeric_columns].count(),
            "median": data[numeric_columns].median(),
            "q1": data[numeric_columns].quantile(0.25),
            "q3": data[numeric_columns].quantile(0.75),
            "min": data[numeric_columns].min(),
            "max": data[numeric_columns].max(),
        }
    ).reset_index(names="column")

    category_summary = (
        data.groupby("category", as_index=False)
        .agg(
            products=("item_id", "count"),
            median_price=("price", "median"),
            q1_price=("price", lambda values: values.quantile(0.25)),
            q3_price=("price", lambda values: values.quantile(0.75)),
        )
        .sort_values("products", ascending=False)
        .reset_index(drop=True)
    )

    designer_count_summary = (
        data.groupby("designer_clean", as_index=False)
        .agg(products=("item_id", "count"), median_price=("price", "median"))
        .sort_values("products", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    designer_price_summary = (
        data.groupby("designer_clean", as_index=False)
        .agg(products=("item_id", "count"), median_price=("price", "median"))
        .query("products >= 5 and designer_clean != 'Unknown'")
        .nlargest(15, "median_price")
        .sort_values("median_price", ascending=False)
        .reset_index(drop=True)
    )

    color_summary = (
        data.groupby("other_colors", as_index=False)
        .agg(products=("item_id", "count"), median_price=("price", "median"))
        .sort_values("products", ascending=False)
        .reset_index(drop=True)
    )

    online_summary = (
        data.assign(
            sellable_label=data["sellable_online"].map(
                {True: "Продається онлайн", False: "Не продається онлайн"}
            )
        )
        .groupby("sellable_label", as_index=False)
        .agg(products=("item_id", "count"), median_price=("price", "median"))
        .sort_values("products", ascending=False)
        .reset_index(drop=True)
    )

    old_price_summary = (
        data.assign(
            old_price_label=data["has_old_price"].map(
                {True: "Є стара ціна", False: "Немає старої ціни"}
            )
        )
        .groupby("old_price_label", as_index=False)
        .agg(products=("item_id", "count"), median_price=("price", "median"))
        .sort_values("products", ascending=False)
        .reset_index(drop=True)
    )

    print_markdown("Пропущені значення", missing_summary)
    print_markdown("Описова статистика з акцентом на медіану", numeric_summary)
    print_markdown("Кількість товарів і медіанна ціна за категоріями", category_summary)
    print_markdown("Топ-15 дизайнерів за кількістю товарів", designer_count_summary)
    print_markdown(
        "Топ-15 дизайнерів за медіанною ціною, мінімум 5 товарів",
        designer_price_summary,
    )
    print_markdown("Наявність інших кольорів", color_summary)
    print_markdown("Продаж онлайн", online_summary)
    print_markdown("Стара і поточна ціна", old_price_summary)

    if not missing_summary.empty:
        missing_plot = missing_summary.sort_values("missing_percent")
        plt.figure(figsize=(11, 6))
        bars = plt.barh(missing_plot["column"], missing_plot["missing_percent"])
        plt.title("Частка пропущених значень за стовпцями")
        plt.xlabel("Пропущені значення, %")
        plt.ylabel("Стовпець")
        for bar, value in zip(bars, missing_plot["missing_percent"]):
            plt.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f" {value:.1f}%",
                va="center",
            )
        save_plot("01_missing_values.png")

    category_count_plot = category_summary.sort_values("products", ascending=False)
    plt.figure(figsize=(14, 7))
    sns.barplot(data=category_count_plot, x="category", y="products", color="steelblue")
    plt.title("Кількість товарів за категоріями")
    plt.xlabel("Категорія")
    plt.ylabel("Кількість товарів")
    plt.xticks(rotation=60, ha="right")
    save_plot("02_products_by_category.png")

    category_price_plot = category_summary.sort_values("median_price", ascending=False)
    plt.figure(figsize=(14, 7))
    sns.barplot(data=category_price_plot, x="category", y="median_price", color="steelblue")
    plt.title("Медіанна ціна товарів за категоріями")
    plt.xlabel("Категорія")
    plt.ylabel("Медіанна ціна")
    plt.xticks(rotation=60, ha="right")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_number))
    save_plot("03_median_price_by_category.png")

    plt.figure(figsize=(12, 7))
    sns.histplot(data=data, x="price", bins=45, kde=True, color="steelblue")
    plt.title("Розподіл поточної ціни")
    plt.xlabel("Ціна")
    plt.ylabel("Кількість товарів")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(compact_number))
    save_plot("04_price_distribution.png")

    plt.figure(figsize=(12, 7))
    sns.histplot(data=data, x="log_price", bins=45, kde=True, color="steelblue")
    plt.title("Розподіл логарифмованої ціни")
    plt.xlabel("log(1 + price)")
    plt.ylabel("Кількість товарів")
    save_plot("05_log_price_distribution.png")

    correlation_columns = [
        "price",
        "depth",
        "height",
        "width",
        "volume_cm3",
        "description_word_count",
    ]
    correlation_matrix = data[correlation_columns].corr(method="spearman")
    correlation_mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        correlation_matrix,
        mask=correlation_mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
    )
    plt.title("Кореляції Спірмена: ціна, розміри та опис")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    save_plot("06_price_dimensions_correlation.png")

    designer_count_plot = designer_count_summary.sort_values("products")
    plt.figure(figsize=(12, 8))
    sns.barplot(data=designer_count_plot, x="products", y="designer_clean", color="steelblue")
    plt.title("Топ-15 дизайнерів за кількістю товарів")
    plt.xlabel("Кількість товарів")
    plt.ylabel("Дизайнер")
    save_plot("07_top_designers_by_product_count.png")

    designer_price_plot = designer_price_summary.sort_values("median_price")
    plt.figure(figsize=(12, 8))
    sns.barplot(data=designer_price_plot, x="median_price", y="designer_clean", color="steelblue")
    plt.title("Топ-15 дизайнерів за медіанною ціною")
    plt.xlabel("Медіанна ціна")
    plt.ylabel("Дизайнер")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(currency_number))
    save_plot("08_top_designers_by_median_price.png")

    plt.figure(figsize=(8, 6))
    sns.barplot(data=color_summary, x="other_colors", y="products", color="steelblue")
    plt.title("Кількість товарів з іншими кольорами та без них")
    plt.xlabel("Доступні інші кольори")
    plt.ylabel("Кількість товарів")
    save_plot("09_other_colors_product_count.png")

    plt.figure(figsize=(8, 6))
    sns.barplot(data=color_summary, x="other_colors", y="median_price", color="steelblue")
    plt.title("Медіанна ціна товарів залежно від наявності інших кольорів")
    plt.xlabel("Доступні інші кольори")
    plt.ylabel("Медіанна ціна")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_number))
    save_plot("10_other_colors_median_price.png")

    plt.figure(figsize=(9, 6))
    sns.barplot(data=online_summary, x="sellable_label", y="products", color="steelblue")
    plt.title("Кількість товарів за доступністю онлайн")
    plt.xlabel("Статус")
    plt.ylabel("Кількість товарів")
    plt.xticks(rotation=45, ha="right")
    save_plot("11_online_product_count.png")

    plt.figure(figsize=(9, 6))
    sns.barplot(data=online_summary, x="sellable_label", y="median_price", color="steelblue")
    plt.title("Медіанна ціна товарів за доступністю онлайн")
    plt.xlabel("Статус")
    plt.ylabel("Медіанна ціна")
    plt.xticks(rotation=45, ha="right")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_number))
    save_plot("12_online_median_price.png")

    old_price_data = data.dropna(subset=["old_price_numeric", "price"]).copy()
    plt.figure(figsize=(10, 7))
    sns.regplot(
        data=old_price_data,
        color="steelblue",
        x="old_price_numeric",
        y="price",
        scatter_kws={"alpha": 0.35, "s": 25},
        line_kws={"linewidth": 2},
    )
    maximum_value = max(
        old_price_data["old_price_numeric"].max(), old_price_data["price"].max()
    )
    plt.plot(
        [0, maximum_value],
        [0, maximum_value],
        linestyle="--",
        linewidth=1.5,
        label="Стара ціна = нова ціна",
    )
    plt.title("Співвідношення старої та поточної ціни")
    plt.xlabel("Стара ціна")
    plt.ylabel("Поточна ціна")
    plt.legend()
    plt.gca().xaxis.set_major_formatter(FuncFormatter(compact_number))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(compact_number))
    save_plot("13_old_price_vs_price_regression.png")

    plt.figure(figsize=(11, 7))
    sns.histplot(data=old_price_data, x="discount_percent", bins=30, kde=True, color="steelblue")
    plt.axvline(
        old_price_data["discount_percent"].median(),
        linestyle="--",
        linewidth=2,
        label=f"Медіана = {old_price_data['discount_percent'].median():.1f}%",
    )
    plt.title("Розподіл знижки для товарів зі старою ціною")
    plt.xlabel("Знижка, %")
    plt.ylabel("Кількість товарів")
    plt.legend()
    save_plot("14_discount_distribution.png")

    return {
        "missing_summary": missing_summary,
        "numeric_summary": numeric_summary,
        "category_summary": category_summary,
        "designer_count_summary": designer_count_summary,
        "designer_price_summary": designer_price_summary,
        "color_summary": color_summary,
        "online_summary": online_summary,
        "old_price_summary": old_price_summary,
    }


def check_hypotheses(data):
    color_yes = data.loc[data["other_colors"].eq("Yes"), "price"].dropna().to_numpy()
    color_no = data.loc[data["other_colors"].eq("No"), "price"].dropna().to_numpy()

    mann_whitney = stats.mannwhitneyu(color_yes, color_no, alternative="two-sided")
    median_test_statistic, median_test_p_value, _, _ = stats.median_test(color_yes, color_no)
    observed_median_difference, permutation_p_value = permutation_median_test(color_yes, color_no)
    bootstrap_interval, bootstrap_differences, color_yes_medians, color_no_medians = (
        bootstrap_median_difference(color_yes, color_no)
    )

    hypothesis_one_results = pd.DataFrame(
        {
            "Тест": ["Mann-Whitney U", "Mood median test", "Permutation test медіан"],
            "Що перевіряє": [
                "Відмінність положення розподілів двох незалежних груп",
                "Рівність медіан двох груп",
                "Рівність медіан через випадкове переставляння груп",
            ],
            "Статистика": [
                mann_whitney.statistic,
                median_test_statistic,
                observed_median_difference,
            ],
            "p-value": [
                mann_whitney.pvalue,
                median_test_p_value,
                permutation_p_value,
            ],
            "Рішення при alpha=0.05": [
                "Відхиляємо H0" if mann_whitney.pvalue < ALPHA else "Не відхиляємо H0",
                "Відхиляємо H0" if median_test_p_value < ALPHA else "Не відхиляємо H0",
                "Відхиляємо H0" if permutation_p_value < ALPHA else "Не відхиляємо H0",
            ],
        }
    )

    hypothesis_one_effect = pd.DataFrame(
        {
            "Група": ["Є інші кольори", "Немає інших кольорів"],
            "Кількість": [len(color_yes), len(color_no)],
            "Медіанна ціна": [np.median(color_yes), np.median(color_no)],
        }
    )

    print("\n## Гіпотеза 1")
    print("H0: цінові розподіли та типова ціна товарів не відрізняються залежно від наявності інших кольорів.")
    print("H1: між товарами з іншими кольорами та без них є статистично значуща відмінність у цінах.")
    print_markdown("Групи гіпотези 1", hypothesis_one_effect)
    print_markdown("Статистичні тести гіпотези 1", hypothesis_one_results, floatfmt=".6g")
    print(f"Фактична різниця медіан: {observed_median_difference:.2f}")
    print(
        f"95% bootstrap CI для різниці медіан: "
        f"[{bootstrap_interval[0]:.2f}; {bootstrap_interval[1]:.2f}]"
    )

    plt.figure(figsize=(10, 7))
    sns.boxplot(data=data, x="other_colors", y="price", showfliers=False, color="lightsteelblue")
    sns.stripplot(
        data=data.sample(min(800, len(data)), random_state=RANDOM_STATE),
        x="other_colors",
        y="price",
        alpha=0.18,
        size=3,
        color="black",
    )
    plt.title("Ціна товарів залежно від наявності інших кольорів")
    plt.xlabel("Доступні інші кольори")
    plt.ylabel("Ціна")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(compact_number))
    save_plot("15_hypothesis_1_other_colors_price.png")

    plt.figure(figsize=(11, 7))
    plt.hist(color_yes_medians, bins=35, density=True, alpha=0.45, label="Є інші кольори")
    plt.hist(color_no_medians, bins=35, density=True, alpha=0.45, label="Немає інших кольорів")
    plt.title("Bootstrap-розподіли медіанної ціни двох груп")
    plt.xlabel("Bootstrap-медіана ціни")
    plt.ylabel("Щільність")
    plt.legend()
    plt.gca().xaxis.set_major_formatter(FuncFormatter(compact_number))
    save_plot("16_hypothesis_1_bootstrap_group_medians.png")

    plt.figure(figsize=(11, 7))
    plt.hist(bootstrap_differences, bins=35)
    plt.axvline(0, linestyle="--", linewidth=2)
    plt.axvline(bootstrap_interval[0], linestyle=":", linewidth=2)
    plt.axvline(bootstrap_interval[1], linestyle=":", linewidth=2)
    plt.title("Bootstrap-розподіл різниці медіан")
    plt.xlabel("Медіана(Yes) − Медіана(No)")
    plt.ylabel("Кількість")
    save_plot("17_hypothesis_1_bootstrap_difference.png")

    volume_data = (
        data.dropna(subset=["volume_cm3", "price"])
        .loc[lambda frame: frame["volume_cm3"].gt(0)]
        .copy()
    )

    spearman_result = stats.spearmanr(volume_data["volume_cm3"], volume_data["price"])
    kendall_result = stats.kendalltau(volume_data["volume_cm3"], volume_data["price"])
    pearson_log_result = stats.pearsonr(volume_data["log_volume"], volume_data["log_price"])

    hypothesis_two_results = pd.DataFrame(
        {
            "Тест": ["Spearman", "Kendall tau", "Pearson для log(volume) і log(price)"],
            "Що перевіряє": [
                "Монотонний зв'язок між об'ємом і ціною",
                "Рангову узгодженість між об'ємом і ціною",
                "Лінійний зв'язок після логарифмування",
            ],
            "Коефіцієнт": [
                spearman_result.statistic,
                kendall_result.statistic,
                pearson_log_result.statistic,
            ],
            "p-value": [
                spearman_result.pvalue,
                kendall_result.pvalue,
                pearson_log_result.pvalue,
            ],
            "Рішення при alpha=0.05": [
                "Відхиляємо H0" if spearman_result.pvalue < ALPHA else "Не відхиляємо H0",
                "Відхиляємо H0" if kendall_result.pvalue < ALPHA else "Не відхиляємо H0",
                "Відхиляємо H0" if pearson_log_result.pvalue < ALPHA else "Не відхиляємо H0",
            ],
        }
    )

    print("\n## Гіпотеза 2")
    print("H0: між геометричним об'ємом товару та його ціною немає статистично значущого зв'язку.")
    print("H1: між геометричним об'ємом товару та його ціною є статистично значущий зв'язок.")
    print(f"Кількість товарів з усіма трьома розмірами: {len(volume_data)}")
    print_markdown("Статистичні тести гіпотези 2", hypothesis_two_results, floatfmt=".6g")

    plt.figure(figsize=(11, 7))
    sns.regplot(
        data=volume_data,
        color="steelblue",
        x="log_volume",
        y="log_price",
        scatter_kws={"alpha": 0.22, "s": 22},
        line_kws={"linewidth": 2},
    )
    plt.title("Зв'язок логарифмованого об'єму з логарифмованою ціною")
    plt.xlabel("log(1 + depth × height × width)")
    plt.ylabel("log(1 + price)")
    save_plot("18_hypothesis_2_volume_price.png")

    hypothesis_one_export = hypothesis_one_results.assign(hypothesis="Other colors and price")
    hypothesis_two_export = hypothesis_two_results.rename(columns={"Коефіцієнт": "Статистика"}).assign(
        hypothesis="Volume and price"
    )

    return pd.concat([hypothesis_one_export, hypothesis_two_export], ignore_index=True)


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("onehot", create_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def evaluate_models(data):
    numeric_features = [
        "depth",
        "height",
        "width",
        "description_word_count",
        "other_colors_binary",
    ]
    categorical_features = ["category", "designer_clean"]
    feature_columns = numeric_features + categorical_features

    categorical_missing = data[categorical_features].isna().sum()
    if categorical_missing.sum() > 0:
        raise ValueError(
            "Категоріальні ознаки містять пропущені значення:\n"
            f"{categorical_missing[categorical_missing.gt(0)]}"
        )

    excluded_columns = pd.DataFrame(
        {
            "Стовпець": [
                "price",
                "Unnamed: 0",
                "item_id",
                "old_price",
                "old_price_numeric",
                "has_old_price",
                "discount_amount",
                "discount_percent",
                "sellable_online",
                "link",
                "short_description",
                "name",
                "designer",
                "volume_cm3",
                "log_price",
                "log_volume",
                "dimension_count",
            ],
            "Причина виключення": [
                "Цільова змінна, яку прогнозує модель",
                "Службовий індекс із CSV",
                "Ідентифікатор товару, не змістова характеристика",
                "Майже пряме джерело інформації про поточну ціну",
                "Числова версія old_price та ризик витоку цільової інформації",
                "Похідна ознака від old_price",
                "Похідна ознака від old_price і price",
                "Похідна ознака від old_price і price",
                "Майже стала ознака з дуже низькою варіативністю",
                "Унікальне посилання з високою кардинальністю",
                "Необроблений текст; використано кількість слів",
                "Висококардинальна текстова ознака, що різко збільшує кількість one-hot колонок",
                "Неочищений стовпець; використано designer_clean",
                "Похідна ознака з великою кількістю пропусків через відсутні габарити",
                "Похідна від цільової змінної price",
                "Похідна від volume_cm3",
                "Технічна похідна ознака, не включена до фінальної моделі",
            ],
        }
    )

    print_markdown("Стовпці, виключені з машинного навчання", excluded_columns)
    print(f"\nОзнаки моделі: {feature_columns}")

    X = data[feature_columns].copy()
    y = data["price"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    models = {
        "Ridge": Ridge(alpha=10.0),
        "DecisionTree": DecisionTreeRegressor(
            max_depth=12,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
        ),
    }

    cross_validation = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    baseline_results = []
    baseline_pipelines = {}

    for model_name, model in models.items():
        model_pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", model),
            ]
        )

        cv_r2 = cross_val_score(
            model_pipeline,
            X_train,
            y_train,
            cv=cross_validation,
            scoring="r2",
            n_jobs=-1,
        )
        cv_rmse = -cross_val_score(
            model_pipeline,
            X_train,
            y_train,
            cv=cross_validation,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )

        baseline_results.append(
            {
                "Model": model_name,
                "CV_R2_mean": cv_r2.mean(),
                "CV_R2_std": cv_r2.std(),
                "CV_RMSE_mean": cv_rmse.mean(),
                "CV_RMSE_std": cv_rmse.std(),
            }
        )
        baseline_pipelines[model_name] = model_pipeline

    baseline_results = (
        pd.DataFrame(baseline_results)
        .sort_values("CV_R2_mean", ascending=False)
        .reset_index(drop=True)
    )

    print_markdown("Порівняння базових моделей на train cross-validation", baseline_results, floatfmt=".5f")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=baseline_results, x="Model", y="CV_R2_mean", color="steelblue")
    plt.title("Порівняння базових моделей за середнім CV R²")
    plt.xlabel("Модель")
    plt.ylabel("Середній R² на train cross-validation")
    plt.xticks(rotation=45, ha="right")
    save_plot("19_baseline_model_comparison.png")

    best_baseline_name = baseline_results.iloc[0]["Model"]
    best_baseline_pipeline = baseline_pipelines[best_baseline_name]

    parameter_grids = {
        "Ridge": {
            "model__alpha": [0.1, 1.0, 10.0, 50.0, 100.0],
        },
        "DecisionTree": {
            "model__max_depth": [8, 12, 16],
            "model__min_samples_leaf": [1, 3, 5],
            "model__min_samples_split": [2, 5, 10],
        },
        "RandomForest": {
            "model__n_estimators": [150, 250],
            "model__max_depth": [14, 18],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": [0.6, 0.8],
        },
        "GradientBoosting": {
            "model__n_estimators": [200, 300],
            "model__learning_rate": [0.03, 0.05],
            "model__max_depth": [2, 3],
            "model__min_samples_leaf": [2, 5],
        },
    }

    grid_cross_validation = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=clone(best_baseline_pipeline),
        param_grid=parameter_grids[best_baseline_name],
        scoring="r2",
        cv=grid_cross_validation,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    tuned_pipeline = grid_search.best_estimator_

    tuned_cv_r2 = cross_val_score(
        tuned_pipeline,
        X_train,
        y_train,
        cv=cross_validation,
        scoring="r2",
        n_jobs=-1,
    )
    tuned_cv_rmse = -cross_val_score(
        tuned_pipeline,
        X_train,
        y_train,
        cv=cross_validation,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    best_baseline_pipeline.fit(X_train, y_train)
    baseline_test_predictions = best_baseline_pipeline.predict(X_test)
    tuned_test_predictions = tuned_pipeline.predict(X_test)

    holdout_results = pd.DataFrame(
        [
            {
                "Model": f"{best_baseline_name} baseline",
                "R2": r2_score(y_test, baseline_test_predictions),
                "MAE": mean_absolute_error(y_test, baseline_test_predictions),
                "RMSE": np.sqrt(mean_squared_error(y_test, baseline_test_predictions)),
            },
            {
                "Model": f"{best_baseline_name} after GridSearchCV",
                "R2": r2_score(y_test, tuned_test_predictions),
                "MAE": mean_absolute_error(y_test, tuned_test_predictions),
                "RMSE": np.sqrt(mean_squared_error(y_test, tuned_test_predictions)),
            },
        ]
    )

    print(f"\nНайкраща базова модель: {best_baseline_name}")
    print(f"Найкращі параметри GridSearchCV: {grid_search.best_params_}")
    print(f"Найкращий середній R² усередині GridSearchCV: {grid_search.best_score_:.5f}")
    print(f"Середній tuned CV R² на train: {tuned_cv_r2.mean():.5f} ± {tuned_cv_r2.std():.5f}")
    print(f"Середній tuned CV RMSE на train: {tuned_cv_rmse.mean():.2f} ± {tuned_cv_rmse.std():.2f}")
    print_markdown("Фінальна оцінка на незалежній test-вибірці", holdout_results, floatfmt=".5f")

    baseline_results.to_csv(
        OUTPUT_DIRECTORY / "baseline_cv_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    holdout_results.to_csv(
        OUTPUT_DIRECTORY / "final_test_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(11, 7))
    plt.scatter(y_test, tuned_test_predictions, alpha=0.45, s=30)
    minimum_value = min(y_test.min(), tuned_test_predictions.min())
    maximum_value = max(y_test.max(), tuned_test_predictions.max())
    plt.plot(
        [minimum_value, maximum_value],
        [minimum_value, maximum_value],
        linestyle="--",
        linewidth=2,
    )
    plt.title("Фактична та прогнозована ціна найкращої моделі")
    plt.xlabel("Фактична ціна")
    plt.ylabel("Прогнозована ціна")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(compact_number))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(compact_number))
    save_plot("20_actual_vs_predicted.png")

    residuals = y_test.to_numpy() - tuned_test_predictions
    plt.figure(figsize=(11, 7))
    plt.scatter(tuned_test_predictions, residuals, alpha=0.45, s=30)
    plt.axhline(0, linestyle="--", linewidth=2)
    plt.title("Залишки найкращої моделі")
    plt.xlabel("Прогнозована ціна")
    plt.ylabel("Фактична ціна − прогнозована ціна")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(compact_number))
    save_plot("21_model_residuals.png")

    transformed_feature_names = tuned_pipeline.named_steps["preprocessor"].get_feature_names_out()
    final_estimator = tuned_pipeline.named_steps["model"]

    if hasattr(final_estimator, "feature_importances_"):
        raw_importances = final_estimator.feature_importances_
    elif hasattr(final_estimator, "coef_"):
        raw_importances = np.abs(np.ravel(final_estimator.coef_))
    else:
        raw_importances = np.zeros(len(transformed_feature_names))

    transformed_importance = pd.DataFrame(
        {
            "transformed_feature": transformed_feature_names,
            "importance": raw_importances,
        }
    )

    original_features = numeric_features + categorical_features

    def identify_original_feature(transformed_feature):
        clean_name = transformed_feature.split("__", 1)[-1]
        for feature in sorted(original_features, key=len, reverse=True):
            if clean_name == feature or clean_name.startswith(f"{feature}_"):
                return feature
        return clean_name

    transformed_importance.loc[:, "original_feature"] = transformed_importance[
        "transformed_feature"
    ].map(identify_original_feature)

    grouped_importance = (
        transformed_importance.groupby("original_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    total_importance = grouped_importance["importance"].sum()
    grouped_importance.loc[:, "importance_percent"] = np.where(
        total_importance > 0,
        grouped_importance["importance"] / total_importance * 100,
        0,
    )

    grouped_importance.to_csv(
        OUTPUT_DIRECTORY / "feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print_markdown("Важливість основних факторів моделі", grouped_importance, floatfmt=".3f")

    importance_plot = grouped_importance.sort_values("importance_percent")
    plt.figure(figsize=(10, 7))
    sns.barplot(
        data=importance_plot,
        x="importance_percent",
        y="original_feature",
        color="steelblue",
    )
    plt.title("Важливість факторів найкращої моделі")
    plt.xlabel("Важливість, %")
    plt.ylabel("Фактор")
    save_plot("22_feature_importance.png")

    cv_results = pd.DataFrame(
        {
            "Fold": np.arange(1, len(tuned_cv_r2) + 1),
            "Tuned R2": tuned_cv_r2,
            "Tuned RMSE": tuned_cv_rmse,
        }
    )

    print_markdown("Результати cross_val_score на train", cv_results, floatfmt=".5f")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=cv_results, x="Fold", y="Tuned R2", color="steelblue")
    plt.axhline(
        tuned_cv_r2.mean(),
        linestyle="--",
        linewidth=2,
        label=f"Середній R² = {tuned_cv_r2.mean():.3f}",
    )
    plt.title("R² найкращої моделі у п'яти train-фолдах")
    plt.xlabel("Фолд")
    plt.ylabel("R²")
    plt.legend()
    save_plot("23_cross_validation_scores.png")

    joblib.dump(tuned_pipeline, OUTPUT_DIRECTORY / "best_ikea_price_model.joblib")

    return baseline_results, holdout_results, grouped_importance, cv_results


def main():
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    FIGURES_DIRECTORY.mkdir(exist_ok=True)

    download_data()
    raw_data = pd.read_csv(FILE_NAME)

    print("\n# IKEA FINAL PROJECT")
    print(f"Початковий розмір набору: {raw_data.shape[0]} рядків і {raw_data.shape[1]} стовпців")
    print(f"Стовпці: {raw_data.columns.tolist()}")

    data, cleaning_summary = clean_data(raw_data)
    print_markdown("Результати очищення", cleaning_summary)

    data.to_csv(
        OUTPUT_DIRECTORY / "ikea_cleaned.csv",
        index=False,
        encoding="utf-8-sig",
    )

    create_eda(data)

    hypothesis_results = check_hypotheses(data)
    hypothesis_results.to_csv(
        OUTPUT_DIRECTORY / "hypothesis_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    baseline_results, holdout_results, feature_importance, cv_results = evaluate_models(data)

    print("\n# ПІДСУМОК")
    print(f"Очищений набір: {len(data)} товарів")
    print(f"Медіанна ціна: {data['price'].median():.2f}")
    print(f"Категорій: {data['category'].nunique()}")
    print(f"Очищених значень designer: {data['designer_clean'].nunique()}")
    print(f"Файли результатів збережено у: {OUTPUT_DIRECTORY.resolve()}")
    print(f"Візуалізації збережено у: {FIGURES_DIRECTORY.resolve()}")


if __name__ == "__main__":
    main()
