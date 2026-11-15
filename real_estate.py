# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 23:38:11 2025

@author: Aleksandra Kudriavtseva
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import os

# Все графики сохраняются в папку graphs (она создаётся автоматически), чтобы не засорять основную папку.
os.makedirs("graphs", exist_ok=True)


def save_fig(name):
    """Сохраняет текущий график в graphs/<name>.png."""
    plt.savefig(os.path.join("graphs", f"{name}.png"), dpi=150, bbox_inches="tight")


def log_step(name, before, after):
    """Печатает, сколько строк было до шага очистки, сколько стало и сколько убрано."""
    print(f"{name}: {before:,} -> {after:,} (removed {before - after:,})")


ds = pd.read_csv("realtor-data.csv")
print(f"Raw data: {len(ds):,} rows")

# Шаг 1. Удаляем строки, где хотя бы одно из нужных полей пустое (в том числе bed, bath и house_size).
# Строки без street (закодированный адрес) оставляем: они не мешают анализу, просто не участвуют в поиске дубликатов.
n = len(ds)
ds_c = ds.dropna(subset=['price', 'bed', 'bath', 'acre_lot', 'city', 'zip_code', 'house_size'])
log_step("Step 1. dropna", n, len(ds_c))

# (по желанию) Сохраняем в новый CSV
# ds_c.to_csv("cleaned_ds.csv", index=False)

# print(ds.head())

# print(ds_c.describe())

# # Shape : (rows, columns)
# print("Dataset shape: ", ds_c.shape)

# # Display all column names
# print("\nFeature Names:")
# print(ds.columns.tolist())

# understanding the datatypes
# print(ds.dtypes)

# Check missing values
# print("\nMissing Values Count:")
# print(ds_c.isnull().sum())

# print(ds_c.state.unique())
# print(ds_c.state.nunique())

us_states = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut',
    'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
    'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan',
    'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
    'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina',
    'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island',
    'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
    'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'
]

# Шаг 2. Удалим строки, где значение в колонке "state" не входит в список us_states
n = len(ds_c)
ds_c = ds_c[ds_c['state'].isin(us_states)]
log_step("Step 2. only 50 states + DC", n, len(ds_c))

# Шаг 3. Удаляем дубликаты: один и тот же дом (street, city, state, zip_code, price, bed, bath).
# Чаще всего это пары for_sale -> sold: тот же дом сначала в продаже, потом продан.
# Из каждой группы оставляем одну запись:
#   1) приоритет у 'sold' (итог сделки): for_sale < ready_to_build < sold;
#   2) при одинаковом статусе берём запись с самой поздней prev_sold_date.
# Сортируем по этим двум признакам и оставляем последнюю запись группы.
# Строки без street в поиске дубликатов не участвуют (иначе все пустые street считались бы "одним адресом").
n = len(ds_c)
status_rank = ds_c['status'].map({'for_sale': 0, 'ready_to_build': 1, 'sold': 2})
ds_c = ds_c.assign(status_rank=status_rank)
has_street = ds_c['street'].notna()
with_street = (ds_c[has_street]
               .sort_values(['status_rank', 'prev_sold_date'], na_position='first', kind='stable')
               .drop_duplicates(subset=['street', 'city', 'state', 'zip_code', 'price', 'bed', 'bath'], keep='last'))
ds_c = pd.concat([with_street, ds_c[~has_street]]).sort_index().drop(columns='status_rank')
log_step("Step 3. drop_duplicates", n, len(ds_c))

# street больше не нужен
ds_c = ds_c.drop(columns=['street'])

# print(ds_c.state.unique())
# print(ds_c.state.nunique())

dsc = ds_c.assign(price_per_sqft=round(ds_c['price'] / ds_c['house_size'], 2))

# Выбросы убираем двумя способами, и делаем это ДО графиков, чтобы EDA строился на чистых данных:
#   А) жёсткие границы и проверка согласованности (шаги 4-8): явные ошибки и сделки, которые не бывают в публичных объявлениях;
#   Б) статистика, IQR (шаг 9).
# Шаг 4. Цена: ниже $50 000 это в основном ошибки/заглушки (бывает даже $1) или участки земли,
# выше $10 млн это закрытые сделки, которые обычно не публикуются.
n = len(dsc)
dsc = dsc[(dsc['price'] >= 50_000) & (dsc['price'] <= 10_000_000)]
log_step("Step 4. price outside $50k-$10M", n, len(dsc))

# Шаг 5. Площадь дома: меньше 400 кв. футов или больше 20 000 это почти всегда ошибка ввода.
n = len(dsc)
dsc = dsc[(dsc['house_size'] >= 400) & (dsc['house_size'] <= 20_000)]
log_step("Step 5. house_size outside 400-20000 sqft", n, len(dsc))

# Таблица по размеру участка ДО шага 6: по ней выбран лимит в 10 акров. Сохраняется в acre_lot_table.png.
def money(x):
    """Форматирует цену: 370000 -> $370K, 1950000 -> $1.95M."""
    return f"${x / 1e6:.2f}M" if x >= 1e6 else f"${x / 1e3:,.0f}K" if x >= 1e3 else f"${x:,.0f}"

acre_bins = [-0.001, 1, 2, 5, 10, 20, 50, 100, 1000, float('inf')]
acre_labels = ['up to 1 acre', '1-2 acres', '2-5 acres', '5-10 acres', '10-20 acres',
               '20-50 acres', '50-100 acres', '100-1000 acres', 'over 1000 acres']
acre_group = pd.cut(dsc['acre_lot'], bins=acre_bins, labels=acre_labels)
price_per_acre = (dsc['price'] / dsc['acre_lot']).where(dsc['acre_lot'] > 0)  # участки 0 акров исключаем

acre_table = pd.DataFrame({
    'Share of listings': acre_group.value_counts(normalize=True, sort=False),
    'Listings': acre_group.value_counts(sort=False),
    'Median price': dsc.groupby(acre_group, observed=False)['price'].median(),
    'Median price per acre': price_per_acre.groupby(acre_group, observed=False).median(),
})
cell_text = [[label, f"{row['Share of listings']:.1%}", f"{int(row['Listings']):,}",
              money(row['Median price']), money(row['Median price per acre'])]
             for label, row in acre_table.iterrows()]

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
tbl = ax.table(cellText=cell_text,
               colLabels=['Lot size', 'Share of listings', 'Listings', 'Median price', 'Median price per acre'],
               cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.auto_set_column_width(list(range(5)))
tbl.scale(1, 1.6)
for j in range(5):
    tbl[0, j].set_facecolor('#dce6f1')
    tbl[0, j].set_text_props(weight='bold')
ax.set_title('Listings by lot size (before the acre_lot filter)', pad=10)
save_fig("acre_lot_table")
plt.show()

# Шаг 6. Участок: оставляем до 10 акров. Всё, что больше, это уже ферма/ранчо, а не типичный жилой дом.
# Значения больше 1000 акров почти наверняка ошибка (вероятно, площадь вписали в кв. футах).
n = len(dsc)
dsc = dsc[dsc['acre_lot'] <= 10]
log_step("Step 6. acre_lot > 10", n, len(dsc))

# Шаг 7. Спальни и ванные: в данных бывает 444 спальни и 222 ванны. Больше 10 это, как правило, многоквартирный дом.
n = len(dsc)
dsc = dsc[(dsc['bed'] <= 10) & (dsc['bath'] <= 10)]
log_step("Step 7. bed > 10 or bath > 10", n, len(dsc))

# Шаг 8. Согласованность bed, bath и площади: ванных не может быть намного больше, чем спален
# (например, 1 спальня и 10 ванных), и на одну спальню не бывает больше 4000 кв. футов. Это ошибки ввода.
n = len(dsc)
dsc = dsc[(dsc['bath'] < dsc['bed'] + 4) & (dsc['house_size'] / dsc['bed'] <= 4000)]
log_step("Step 8. inconsistent bed/bath/house_size", n, len(dsc))

# Шаг 9. IQR по price_per_sqft. Считаем в логарифмической шкале (у цены сильный правый хвост, обычный IQR даёт
# отрицательную нижнюю границу) и отдельно внутри каждого штата (иначе Гавайи и Калифорния стали бы "выбросами").
n = len(dsc)
log_ppsf = np.log10(dsc['price_per_sqft'])
q1 = log_ppsf.groupby(dsc['state']).transform(lambda s: s.quantile(0.25))
q3 = log_ppsf.groupby(dsc['state']).transform(lambda s: s.quantile(0.75))
iqr = q3 - q1
dsc = dsc[(log_ppsf >= q1 - 1.5 * iqr) & (log_ppsf <= q3 + 1.5 * iqr)]
log_step("Step 9. IQR of price_per_sqft (log scale, per state)", n, len(dsc))

print(f"Final dataset: {len(dsc):,} rows ({len(dsc) / len(ds):.1%} of raw data)")

# print(dsc.describe())
# print("Rows left:", len(dsc))

# print(dsc.head())

# crime_rate: общий уровень преступности в штате в 2025 году, число преступлений на 100 000 жителей.
# = насильственные (violent) + имущественные (property) преступления, т.е. все преступления, которые
# ФБР считает в основной статистике (Part I). Источник: FBI (UCR / Crime Data Explorer), 2025 год,
# таблицы USAFacts https://usafacts.org/articles/which-states-have-the-least-and-most-crime/ (все 50 штатов + DC).
# Оговорка: у Florida и North Dakota данные за 2025 неполные (меньше агентств передали отчёты).
crime_rate={
    'District of Columbia':3935.3,
    'New Mexico':2861.1,
    'Alaska':2622.1,
    'Colorado':2565.3,
    'Louisiana':2481.5,
    'Oregon':2429.2,
    'Nevada':2375.2,
    'Washington':2258.7,
    'Kansas':2253.7,
    'California':2224.9,
    'Oklahoma':2221.2,
    'Maryland':2177.7,
    'Arkansas':2176.7,
    'Texas':2130.9,
    'Tennessee':2128.9,
    'North Carolina':2110.2,
    'South Carolina':2063.8,
    'Missouri':2061.9,
    'New York':1943.2,
    'Delaware':1906.1,
    'Arizona':1891.4,
    'Montana':1867.6,
    'Alabama':1810.1,
    'Hawaii':1808.9,
    'Illinois':1758.5,
    'Georgia':1706.2,
    'Ohio':1704.4,
    'North Dakota':1696.5,
    'Michigan':1674.7,
    'Utah':1651.2,
    'Minnesota':1635.0,
    'Vermont':1633.7,
    'South Dakota':1619.8,
    'Nebraska':1597.0,
    'Virginia':1559.4,
    'Pennsylvania':1550.6,
    'Mississippi':1490.7,
    'Kentucky':1473.8,
    'New Jersey':1443.4,
    'Florida':1437.0,
    'Iowa':1405.6,
    'Massachusetts':1333.9,
    'Indiana':1306.6,
    'West Virginia':1295.0,
    'Wisconsin':1257.0,
    'Connecticut':1255.6,
    'Wyoming':1217.9,
    'Rhode Island':1054.6,
    'Maine':1032.3,
    'New Hampshire':935.7,
    'Idaho':822.3
}
dsc['crime_rate'] = dsc['state'].map(crime_rate)

# median_income: медианный доход домохозяйства по штатам в 2025 году, в долларах 2025 года.
# Источник: U.S. Census Bureau, American Community Survey 2025, 1-year estimates,
# ACS Brief ACSBR-028 "Household Income in States and Metropolitan Areas: 2025" (октябрь 2026), Appendix Table 1.
# https://www2.census.gov/library/publications/2026/demo/acsbr-028.pdf (все 50 штатов + DC).
median_income={
    'District of Columbia':109505,
    'New Jersey':107797,
    'Maryland':107700,
    'New Hampshire':106759,
    'Massachusetts':106539,
    'Hawaii':105013,
    'California':102159,
    'Washington':101946,
    'Connecticut':100056,
    'Colorado':99729,
    'Virginia':99322,
    'Utah':99311,
    'Alaska':95220,
    'Delaware':92520,
    'Minnesota':92297,
    'Rhode Island':91039,
    'New York':89210,
    'Oregon':86953,
    'Illinois':86676,
    'Vermont':86018,
    'Nevada':85198,
    'Idaho':84234,
    'Arizona':82940,
    'Texas':82625,
    'Georgia':81847,
    'Maine':81760,
    'Nebraska':81614,
    'North Dakota':81545,
    'Pennsylvania':81473,
    'Wyoming':80894,
    'Florida':80846,
    'South Dakota':80506,
    'Wisconsin':80213,
    'Kansas':77824,
    'North Carolina':77638,
    'Montana':77099,
    'Iowa':76483,
    'Tennessee':75875,
    'Michigan':75832,
    'Missouri':75485,
    'Ohio':74804,
    'Indiana':74516,
    'South Carolina':73672,
    'New Mexico':70342,
    'Oklahoma':68441,
    'Alabama':67663,
    'Kentucky':66447,
    'Arkansas':65588,
    'Louisiana':65139,
    'West Virginia':61340,
    'Mississippi':59346
}
dsc['median_income'] = dsc['state'].map(median_income)

# Flood risk score (0-100), все 50 штатов + DC.
# Метрика: частота наводнений в штате = число федеральных major disaster declarations (тип DR) за 2000-2025,
# связанных с наводнениями (incidentType Flood / Hurricane / Tropical Storm / Typhoon / Coastal Storm
# или слово FLOOD в названии). Максимум (Kentucky и West Virginia, по 39 объявлений) = 100.
# Источник: OpenFEMA, DisasterDeclarationsSummaries v2 (https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries)
flood_risk_score= {
    'Kentucky':100.0,
    'West Virginia':100.0,
    'Missouri':92.3,
    'Mississippi':87.2,
    'Florida':84.6,
    'Kansas':84.6,
    'New York':82.1,
    'Vermont':82.1,
    'Oklahoma':79.5,
    'Alaska':76.9,
    'California':76.9,
    'Tennessee':74.4,
    'Texas':74.4,
    'South Dakota':71.8,
    'Alabama':69.2,
    'Iowa':69.2,
    'Louisiana':69.2,
    'Nebraska':69.2,
    'Maine':66.7,
    'Arkansas':64.1,
    'Minnesota':64.1,
    'New Hampshire':64.1,
    'North Dakota':56.4,
    'Virginia':53.8,
    'North Carolina':51.3,
    'Washington':51.3,
    'Wisconsin':48.7,
    'New Mexico':46.2,
    'Illinois':43.6,
    'Ohio':43.6,
    'Pennsylvania':43.6,
    'Arizona':38.5,
    'Georgia':35.9,
    'Indiana':35.9,
    'Montana':35.9,
    'New Jersey':35.9,
    'Oregon':35.9,
    'Hawaii':33.3,
    'Massachusetts':33.3,
    'South Carolina':30.8,
    'Idaho':28.2,
    'Michigan':28.2,
    'Connecticut':23.1,
    'Maryland':23.1,
    'Delaware':20.5,
    'Nevada':17.9,
    'Rhode Island':17.9,
    'Utah':17.9,
    'Wyoming':12.8,
    'District of Columbia':10.3,
    'Colorado':7.7
}
dsc['flood_risk'] = dsc['state'].map(flood_risk_score)

# print(dsc.head())
# dsc.to_csv("cleaned_ds.csv", index=False)

# Define features for visualization (Choosing the numerical features)
num_features = ['price', 'bed', 'bath', 'acre_lot', 'house_size', 'price_per_sqft', 'crime_rate', 'median_income', 'flood_risk']

# Create a 2x3 grid for visualization
fig, ax = plt.subplots(3, 3, figsize=(18, 12))

# Iterate over features and plot
for i, feature in enumerate(num_features):
    row, col = divmod(i, 3)
    sns.histplot(dsc[feature], kde=True, bins=30, ax=ax[row, col])
    ax[row, col].set_title(f'Distribution of {feature}')
    
plt.tight_layout()
save_fig("distributions")
plt.show()

# Create boxplots for numerical variables
fig, ax = plt.subplots(3, 3, figsize=(18, 20))

for i, feature in enumerate(num_features):
    row, col = divmod(i, 3)
    sns.boxplot(y=dsc[feature], ax=ax[row, col])
    ax[row, col].set_title(f'Boxplot of {feature}')

plt.tight_layout()
save_fig("boxplots")
plt.show()

# Define categorical features for visualization
# (status убрали: распределение for_sale/sold мало что говорит)
cat_features = ['bed', 'bath', 'state']

# bed и bath сверху, state снизу на всю ширину (у него много категорий)
fig = plt.figure(figsize=(20, 14))
grid = fig.add_gridspec(2, 2)
axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]

# Iterate over categorical features and plot
for a, feature in zip(axes, cat_features):
    sns.countplot(data=dsc, x=feature, order=dsc[feature].value_counts().index, ax=a)
    a.set_title(f'Distribution of {feature}')
    a.tick_params(axis='x', rotation=90)  # Rotate x-axis labels for better readability

plt.tight_layout(pad=2)
save_fig("categorical_distributions")
plt.show()

# Scatter plot for numerical vs. numerical
sns.scatterplot(data=dsc, x='price', y='house_size')
plt.title("house_size vs. price")
save_fig("house_size_vs_price")
plt.show()

sns.scatterplot(data=dsc, x='bed', y='price')
plt.title("bed vs. price")
save_fig("bed_vs_price")
plt.show()

sns.scatterplot(data=dsc, x='acre_lot', y='price')
plt.title("acre_lot vs. price")
save_fig("acre_lot_vs_price")
plt.show()

sns.scatterplot(data=dsc, x='acre_lot', y='house_size')
plt.title("acre_lot vs. house_size")
save_fig("acre_lot_vs_house_size")
plt.show()

# Scatterplot: связь house_size с количеством спален и ванных (слева bed, справа bath).
# Точек больше миллиона, поэтому рисуем случайную выборку 100 000 строк и делаем точки полупрозрачными.
sample = dsc.sample(100_000, random_state=42)
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

for a, col in zip(ax, ['bed', 'bath']):
    sns.scatterplot(data=sample, x=col, y='house_size', alpha=0.15, s=12, ax=a)
    rho = dsc[col].corr(dsc['house_size'], method='spearman')
    a.set_title(f'{col} vs. house_size (Spearman = {rho:.2f})')

plt.tight_layout()
save_fig("bed_bath_vs_house_size")
plt.show()

# Scatterplot: price_per_sqft против количества спален и ванных. Красная линия это медиана по каждому значению.
# price_per_sqft честнее, чем price: цена растёт вместе с площадью, а цена за фут убирает эту зависимость.
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

for a, col in zip(ax, ['bed', 'bath']):
    sns.scatterplot(data=sample, x=col, y='price_per_sqft', alpha=0.15, s=12, ax=a)
    medians = dsc.groupby(col)['price_per_sqft'].median()
    a.plot(medians.index, medians.values, color='red', marker='o', label='median')
    rho = dsc[col].corr(dsc['price_per_sqft'], method='spearman')
    a.set_title(f'{col} vs. price_per_sqft (Spearman = {rho:.2f})')
    a.legend()

plt.tight_layout()
save_fig("bed_bath_vs_price_per_sqft")
plt.show()

# Словарь: полные названия → коды штатов (нужен для подписей на графиках и для карт ниже)
state_abbrev = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC'
}

# --- Графики по price_per_sqft ---

# 1. Boxplot: price_per_sqft по штатам, штаты отсортированы по медиане (слева самые дорогие).
state_order = dsc.groupby('state')['price_per_sqft'].median().sort_values(ascending=False).index
plt.figure(figsize=(18, 7))
sns.boxplot(data=dsc, x='state', y='price_per_sqft', order=state_order, showfliers=False)
plt.title("Boxplot: price_per_sqft by state (sorted by median)")
plt.xticks(rotation=90)
plt.tight_layout()
save_fig("ppsf_by_state_boxplot")
plt.show()

# 2. price_per_sqft vs. median_income: одна точка = один штат (медиана price_per_sqft и доход штата).
state_stats = dsc.groupby('state').agg(price_per_sqft=('price_per_sqft', 'median'),
                                       median_income=('median_income', 'first'))
rho = state_stats['median_income'].corr(state_stats['price_per_sqft'], method='spearman')

plt.figure(figsize=(10, 7))
sns.regplot(data=state_stats, x='median_income', y='price_per_sqft', ci=None,
            scatter_kws={'s': 40}, line_kws={'color': 'red'})
for state, row in state_stats.iterrows():
    plt.annotate(state_abbrev[state], (row['median_income'], row['price_per_sqft']),
                 xytext=(4, 4), textcoords='offset points', fontsize=8)
plt.title(f"State median income vs. median price_per_sqft (Spearman = {rho:.2f})")
plt.tight_layout()
save_fig("ppsf_vs_income_by_state")
plt.show()

# 3. Тепловая карта: bed x bath, цвет и число в клетке = медиана price_per_sqft.
# Клетки, где меньше 30 объявлений, скрываем: медиана по нескольким объектам ненадёжна.
ppsf_pivot = dsc.pivot_table(index='bed', columns='bath', values='price_per_sqft', aggfunc='median')
count_pivot = dsc.pivot_table(index='bed', columns='bath', values='price_per_sqft', aggfunc='size')
# подписи осей целыми числами (1, 2, 3, ... вместо 1.0, 2.0, ...)
ppsf_pivot.index, ppsf_pivot.columns = ppsf_pivot.index.astype(int), ppsf_pivot.columns.astype(int)
count_pivot.index, count_pivot.columns = count_pivot.index.astype(int), count_pivot.columns.astype(int)

plt.figure(figsize=(11, 8))
sns.heatmap(ppsf_pivot.where(count_pivot >= 30), annot=True, fmt='.0f', cmap='viridis',
            cbar_kws={'label': 'median price_per_sqft'})
plt.gca().invert_yaxis()  # bed идёт снизу вверх от 1 до 10, в левом нижнем углу bed = 1 и bath = 1
plt.title("Median price_per_sqft by bed and bath (cells with < 30 listings hidden)")
plt.tight_layout()
save_fig("ppsf_bed_bath_heatmap")
plt.show()

sns.scatterplot(data=dsc, x='median_income', y='price')
plt.title("income vs. price")
save_fig("income_vs_price")
plt.show()

sns.scatterplot(data=dsc, x='crime_rate', y='price')
plt.title("crime_rate vs. price")
save_fig("crime_rate_vs_price")
plt.show()

# Box plot for numerical vs. categorical
sns.boxplot(data=dsc, x='state', y='price')
plt.title("Boxplot: price by state")
plt.xticks(rotation=90)
save_fig("price_by_state_boxplot")
plt.show()

# Create a correlation heatmap
# brokered_by и zip_code числовые, но это идентификаторы, а не признаки, поэтому в матрицу их не берём
plt.figure(figsize=(8, 6))
sns.heatmap(dsc.select_dtypes(include=['number']).drop(columns=['brokered_by', 'zip_code']).corr(),
            annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
save_fig("correlation_heatmap")
plt.show()

# Create a price map
import plotly.io as pio

pio.renderers.default = 'notebook_connected' 

state_price = dsc.groupby('state')['price'].mean().reset_index()

# (словарь state_abbrev с кодами штатов теперь лежит выше, рядом с графиками по штатам)

# # Применим к датафрейму
# state_price['code'] = state_price['state'].map(state_abbrev)

# fig = px.choropleth(state_price,
#                     locations='code',
#                     locationmode='USA-states',
#                     color='price',
#                     scope='usa',
#                     color_continuous_scale='Viridis',
#                     title='Average House Price by State')

# fig.write_html("choropleth.html")

# # Create a crime rate map
# state_crime = dsc.groupby('state')['crime_rate'].mean().reset_index()
# state_crime['code'] = state_crime['state'].map(state_abbrev)

# fig1 = px.choropleth(state_crime,
#                     locations='code',
#                     locationmode='USA-states',
#                     color='crime_rate',
#                     scope='usa',
#                     color_continuous_scale='Reds',
#                     title='Average Crime Rate by State')
# fig1.write_html('crime_rate_map.html') 

# # Create income map
# state_income = dsc.groupby('state')['median_income'].mean().reset_index()
# state_income['code'] = state_income['state'].map(state_abbrev)

# fig2 = px.choropleth(state_income,
#                     locations='code',
#                     locationmode='USA-states',
#                     color='median_income',
#                     scope='usa',
#                     color_continuous_scale='Blues',
#                     title='Average Median Income by State')
# fig2.write_html('median_income_map.html')

# # Create flood risk map
# state_flood = dsc.groupby('state')['flood_risk'].mean().reset_index()
# state_flood['code'] = state_flood['state'].map(state_abbrev)

# fig3 = px.choropleth(state_flood,
#                     locations='code',
#                     locationmode='USA-states',
#                     color='flood_risk',
#                     scope='usa',
#                     color_continuous_scale='Viridis',
#                     title='Average Flood Risk by State')
# fig3.write_html('flood_risk_map.html')

# print("\nMissing Values Count:")
# print(dsc.isnull().sum())

# # Shows outliers
# sns.boxplot(x=dsc['price_per_sqft'])
# plt.show()

# # Проверка оставшихся максимальных значений
# print(dsc.sort_values('price_per_sqft', ascending=False).head(10))

# Сохраняем в сжатом виде (gzip): ~18 МБ вместо ~120 МБ, такой файл помещается на GitHub (лимит 100 МБ на файл),
# а pandas и приложение читают его напрямую.
dsc.to_csv("dsc.csv.gz", index=False, compression="gzip")












