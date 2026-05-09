import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

NULL_THRESHOLD = 0.4
CORRELATION_THRESHOLD = 0.9
N_PCA_COMPONENTS = 11

_LIGHTGBM_UNSAFE_CHARS = re.compile(r'[^A-Za-z0-9_]')
_CONSECUTIVE_UNDERSCORES = re.compile(r'_+')

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def adapt_sparkov_to_ieee_schema(df_sparkov: pd.DataFrame) -> pd.DataFrame:
    """Map Sparkov columns into IEEE-CIS-equivalent roles."""
    df = df_sparkov.copy()

    df = df.rename(columns={'is_fraud': 'isFraud'})

    if 'unix_time' in df.columns:
        df['TransactionDT'] = (df['unix_time'] - df['unix_time'].min()).astype(np.int64)
    else:
        ts = pd.to_datetime(df['trans_date_trans_time'])
        df['TransactionDT'] = ((ts - ts.min()).dt.total_seconds()).astype(np.int64)

    df = df.rename(columns={'amt': 'TransactionAmt'})

    df = df.rename(columns={
        'cc_num': 'card1',
        'category': 'ProductCD',
        'merchant': 'P_emaildomain',
        'zip': 'addr1',
    })

    if all(c in df.columns for c in ['lat', 'long', 'merch_lat', 'merch_long']):
        df['card_merch_dist_km'] = haversine_km(
            df['lat'], df['long'], df['merch_lat'], df['merch_long']
        )
    else:
        df['card_merch_dist_km'] = np.nan

    drop_cols = ['first', 'last', 'street', 'city', 'state', 'gender', 'dob',
                 'job', 'trans_num', 'unix_time', 'trans_date_trans_time',
                 'lat', 'long', 'merch_lat', 'merch_long']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    ieee_only_numeric = (
        ['card2', 'card3', 'card5'] +
        [f'C{i}' for i in range(1, 15)] +
        [f'D{i}' for i in range(1, 16)] +
        [f'V{i}' for i in range(1, 340)] +
        ['addr2']
    )
    ieee_only_categorical = (
        ['card4', 'card6', 'R_emaildomain'] +
        [f'M{i}' for i in range(1, 10)]
    )

    missing_numeric = [c for c in ieee_only_numeric if c not in df.columns]
    missing_categorical = [c for c in ieee_only_categorical if c not in df.columns]
    all_missing = missing_numeric + missing_categorical
    if all_missing:
        nan_df = pd.DataFrame(np.nan, index=df.index, columns=all_missing)
        df = pd.concat([df, nan_df], axis=1)

    df = df.copy()  # de-fragment in case anything else fragmented above
    df['TransactionID'] = np.arange(len(df))
    return df


def _make_lightgbm_safe_columns(df: pd.DataFrame) -> pd.DataFrame:
    sanitized_names = []
    name_counts = {}
    for original_name in df.columns:
        safe_name = _LIGHTGBM_UNSAFE_CHARS.sub('_', str(original_name))
        safe_name = _CONSECUTIVE_UNDERSCORES.sub('_', safe_name).strip('_') or 'col'
        if safe_name in name_counts:
            name_counts[safe_name] += 1
            safe_name = f'{safe_name}_{name_counts[safe_name]}'
        else:
            name_counts[safe_name] = 0
        sanitized_names.append(safe_name)
    df.columns = sanitized_names
    return df.reset_index(drop=True)


def run_phase4_pipeline(df_in: pd.DataFrame, fit_artifacts: dict | None = None):
    raw = df_in.copy()
    df = raw.copy()
    artifacts = fit_artifacts.copy() if fit_artifacts else {}

    # === Cleaning ===
    df = df.drop_duplicates()
    df['hour'] = (df['TransactionDT'] // 3600) % 24
    df['day_of_week'] = (df['TransactionDT'] // (3600 * 24)) % 7

    q1 = df['TransactionAmt'].quantile(0.25)
    q3 = df['TransactionAmt'].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 3 * iqr
    df['TransactionAmt'] = np.where(df['TransactionAmt'] > upper, upper, df['TransactionAmt'])

    cat_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
                'addr1', 'addr2'] + [f'M{i}' for i in range(1, 10)]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    if 'card6' in df.columns:
        df['card6'] = (df['card6'].astype(object)
                       .replace({'debit or credit': 'debit', 'charge card': 'credit'})
                       .infer_objects(copy=False)
                       .astype('category'))
    if 'P_emaildomain' in df.columns:
        df['P_emaildomain'] = (df['P_emaildomain'].astype(object)
                               .replace({'gmail': 'gmail.com', 'google': 'gmail.com',
                                         'outlook': 'outlook.com'})
                               .infer_objects(copy=False)
                               .astype('category'))

    # === Selection ===
    df = df.drop(columns=['TransactionID'])

    if 'null_cols' in artifacts:
        null_cols = [c for c in artifacts['null_cols'] if c in df.columns]
    else:
        null_cols = [c for c in df.columns if df[c].isnull().mean() > NULL_THRESHOLD]
        artifacts['null_cols'] = null_cols
    df = df.drop(columns=null_cols)

    if 'redundant_cols' in artifacts:
        redundant = [c for c in artifacts['redundant_cols'] if c in df.columns]
    else:
        num_corr = df.select_dtypes(include=[np.number]).corr().abs()
        upper_t = num_corr.where(np.triu(np.ones(num_corr.shape), k=1).astype(bool))
        redundant = [c for c in upper_t.columns if any(upper_t[c] > CORRELATION_THRESHOLD)]
        artifacts['redundant_cols'] = redundant
    df = df.drop(columns=redundant)

    # === Imputation ===
    for col in df.columns:
        if df[col].dtype.name == 'category':
            if 'unknown' not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories('unknown')
            df[col] = df[col].fillna('unknown')
        elif np.issubdtype(df[col].dtype, np.number):
            if f'median_{col}' not in artifacts:
                artifacts[f'median_{col}'] = df[col].median()
            df[col] = df[col].fillna(artifacts[f'median_{col}'])
        else:
            df[col] = df[col].fillna('unknown')

    # === Frequency encoding + PCA on V ===
    freq = {}
    for col in ['card1', 'card2', 'addr1', 'P_emaildomain']:
        if col not in df.columns:
            continue
        if f'freq_{col}' in artifacts:
            mapping = artifacts[f'freq_{col}']
        else:
            mapping = df[col].value_counts(dropna=False).to_dict()
            artifacts[f'freq_{col}'] = mapping
        freq[f'{col}_freq'] = df[col].astype(object).map(mapping).fillna(1).astype(float)

    v_cols = [c for c in df.columns if c.startswith('V')]
    if v_cols:
        v_filled = df[v_cols].fillna(-999)
        if 'pca_v' in artifacts:
            scaler, pca = artifacts['pca_v']
            v_pca = pca.transform(scaler.transform(v_filled))
        else:
            scaler = StandardScaler()
            pca = PCA(n_components=N_PCA_COMPONENTS)
            v_pca = pca.fit_transform(scaler.fit_transform(v_filled))
            artifacts['pca_v'] = (scaler, pca)
        for i in range(N_PCA_COMPONENTS):
            freq[f'pca_vesta_{i}'] = v_pca[:, i]
    else:
        for i in range(N_PCA_COMPONENTS):
            freq[f'pca_vesta_{i}'] = 0.0

    df = df.drop(columns=v_cols)
    df = pd.concat([df, pd.DataFrame(freq, index=df.index)], axis=1)

    if 'P_emaildomain' in df.columns:
        df['email_suffix'] = df['P_emaildomain'].astype(str).apply(
            lambda x: x.split('.')[-1] if '.' in x else x)
    else:
        df['email_suffix'] = 'unknown'

    # === Behavioral / log / interactions ===
    if 'card1' in df.columns:
        if 'card1_avg_amount_map' in artifacts:
            df['card1_avg_amount'] = df['card1'].map(artifacts['card1_avg_amount_map']).fillna(
                df['TransactionAmt'].mean())
        else:
            avg_map = df.groupby('card1')['TransactionAmt'].mean().to_dict()
            artifacts['card1_avg_amount_map'] = avg_map
            df['card1_avg_amount'] = df['card1'].map(avg_map)
        df['amount_vs_card_avg'] = df['TransactionAmt'] / df['card1_avg_amount']

    df['amount_log'] = np.log1p(df['TransactionAmt'])

    if 'card2' in df.columns and 'addr1' in df.columns:
        df['card2_x_addr1'] = df['card2'].astype(str) + '_' + df['addr1'].astype(str)
    if 'card1' in df.columns:
        df['card1_x_email'] = df['card1'].astype(str) + '_' + df.get(
            'P_emaildomain', pd.Series(['na'] * len(df))).astype(str)

    for col in df.select_dtypes('category').columns:
        df[col] = df[col].astype(str)

    # === Phase 4 features ===
    df['amount_cents'] = ((df['TransactionAmt'] * 100) % 100).astype(int)
    df['is_round_amount'] = (df['amount_cents'] == 0).astype(int)

    feat_cols = [c for c in raw.columns if c not in ['TransactionID', 'isFraud']]
    df['null_count'] = raw[feat_cols].isnull().sum(axis=1).values

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    if 'card1' in df.columns and 'addr1' in df.columns:
        df['card1_x_addr1'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str)
        if 'card_location_freq_map' in artifacts:
            df['card_location_freq'] = df['card1_x_addr1'].map(
                artifacts['card_location_freq_map']).fillna(1).astype(int)
        else:
            mp = df['card1_x_addr1'].value_counts(dropna=False).to_dict()
            artifacts['card_location_freq_map'] = mp
            df['card_location_freq'] = df['card1_x_addr1'].map(mp).astype(int)

    # === Encode interactions, drop strings, label-encode ===
    for col in ['card2_x_addr1', 'card1_x_email']:
        if col not in df.columns:
            continue
        if f'freq_{col}' in artifacts:
            mp = artifacts[f'freq_{col}']
        else:
            mp = df[col].value_counts(dropna=False).to_dict()
            artifacts[f'freq_{col}'] = mp
        df[f'{col}_freq'] = df[col].map(mp).fillna(1).astype(int)

    drop_str = ['P_emaildomain', 'R_emaildomain', 'addr1', 'addr2',
                'card2_x_addr1', 'card1_x_email', 'card1_x_addr1']
    df = df.drop(columns=[c for c in drop_str if c in df.columns])

    for col in df.select_dtypes(include=['object']).columns:
        if f'le_{col}' in artifacts:
            le = artifacts[f'le_{col}']
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col].astype(str))
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            artifacts[f'le_{col}'] = le

    df = _make_lightgbm_safe_columns(df)
    return df, artifacts