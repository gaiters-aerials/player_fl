import os
import sys
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # -> ./layer_pfl/code/evaluation
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR) # -> (...)/layer_pfl/code
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _CURRENT_DIR)

from configs import *
from helper import set_seeds

# Constants
set_seeds(seed_value=1)
BATCH_SIZE = 32
print(f"Using device: {DEVICE}")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ================= Utility Functions =================

def custom_sample(group, max_samples=5000):
    """Randomly sample a maximum of max_samples from a group."""
    n_samples = min(len(group), max_samples)
    return group.sample(n_samples, random_state=42)


def convert_ids_to_embedding_indices(input_ids, token_to_index, default_index=0):
    """Convert a tensor of token IDs to embedding indices using a mapping dictionary."""
    flat_input_ids = input_ids.flatten()
    flat_indices = [token_to_index.get(token_id.item(), default_index) for token_id in flat_input_ids]
    embedding_indices = torch.tensor(flat_indices, dtype=torch.long).view_as(input_ids)
    return embedding_indices


# ================= Benchmark Dataset =================
def _load_benchmark_images():
    """Handle torchvision datasets with download functionality"""
    print("\n================ Loading Benchmark Images ================")
    
    # Create the data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Define dataset classes with download=True to ensure datasets are available
    dataset_classes = {
        'EMNIST': lambda: EMNIST(f'{DATA_DIR}/EMNIST', split='byclass', download=True, train=True),
        'CIFAR': lambda: CIFAR10(f'{DATA_DIR}/CIFAR10', download=True, train=True),
        'FMNIST': lambda: FashionMNIST(f'{DATA_DIR}/FMNIST', download=True, train=True),
    }
    
    # Load each dataset
    datasets = {}
    for name, dataset_fn in dataset_classes.items():
        try:
            print(f"Loading {name} dataset...")
            datasets[name] = dataset_fn()
            print(f"Successfully loaded {name} dataset")
        except Exception as e:
            print(f"Error loading {name} dataset: {e}")
    
    print("Benchmark images loading complete.")
    return


# ================= ISIC Dataset Processing =================

def create_isic_dataset():
    # Load ground truth labels and assign category codes
    labels_path = os.path.join(DATA_DIR, 'ISIC', 'ISIC_2019_Training_GroundTruth.csv')
    labels = pd.read_csv(labels_path)
    labels['category'] = labels[['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']].idxmax(axis=1)
    labels['target'] = labels['category'].astype('category').cat.codes

    # Load metadata and merge labels
    metadata_path = os.path.join(DATA_DIR, 'ISIC', 'ISIC_2019_Training_Metadata_FL.csv')
    metadata = pd.read_csv(metadata_path)
    metadata = metadata.merge(labels[['image', 'target']], on='image')
    metadata = metadata[metadata['target'].isin([1, 2, 4, 5])]

    # Sample images per site and remap targets
    sampled_metadata = metadata.groupby('dataset').apply(custom_sample).reset_index(drop=True)
    sampled_metadata['target'] = sampled_metadata['target'].map({1: 0, 2: 1, 4: 2, 5: 3})
    prefix = os.path.join(DATA_DIR, 'ISIC', 'ISIC_2019_Training_Input_preprocessed')
    sampled_metadata['path'] = prefix + '/' + sampled_metadata['image'].astype(str) + '.jpg'

    # Save metadata for each unique site
    for i, dataset in enumerate(sampled_metadata['dataset'].unique()):
        df_site = sampled_metadata[sampled_metadata['dataset'] == dataset]
        out_path = os.path.join(DATA_DIR, 'ISIC', f'site_{i}_metadata.csv')
        df_site.to_csv(out_path, index=False)

    print("ISIC dataset metadata has been saved.")


# ---------------- Sentiment Dataset Processing ----------------

def create_sentiment_dataset():
    # Unzip the sentiment.zip file if it exists
    zip_path = os.path.join(DATA_DIR, 'Sentiment', 'sentiment.zip')
    sentiment_dir = os.path.join(DATA_DIR, 'Sentiment')
    
    # Create the Sentiment directory if it doesn't exist
    os.makedirs(sentiment_dir, exist_ok=True)
    
    if os.path.exists(zip_path):
        print("Unzipping sentiment.zip file...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(sentiment_dir)
        print("Sentiment dataset unzipped successfully.")
    else:
        print("sentiment.zip not found, assuming data is already extracted.")
    
    # Load tweet data and filter for top users
    data_path = os.path.join(DATA_DIR, 'Sentiment', 'training.1600000.processed.noemoticon.csv')
    with open(data_path, 'r', encoding='utf-8', errors='replace') as file:
        df = pd.read_csv(file, names= ['target','id', 'date', 'flag', 'user', 'tweet'], usecols=['target', 'user', 'tweet'])
    users = list(df['user'].value_counts().index)
    users_included = users[:50]
    df_used = df[df['user'].isin(users_included)]
    df_used['target'] = df_used['target'].map({0: 0, 4: 1}) # 4 is labelled as postive and 0 as negative, turn into 0,1
    tweets = df_used['tweet'].tolist()

    # Tokenise tweets using BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids = [tokenizer.encode(text, add_special_tokens=True) for text in tweets]
    input_ids = pad_sequence([torch.tensor(seq) for seq in input_ids], batch_first=True)
    attention_masks = (input_ids != tokenizer.pad_token_id).long()

    # Compute tweet embeddings using BERT model with GPU if available
    model = BertModel.from_pretrained('bert-base-uncased').to(DEVICE)
    model.eval()
    all_embeddings = []
    for i in range(0, len(input_ids), BATCH_SIZE):
        # Move input batch to the correct device
        input_chunk = input_ids[i:i + BATCH_SIZE].to(DEVICE)
        attention_chunk = attention_masks[i:i + BATCH_SIZE].to(DEVICE)
        with torch.no_grad():
            embeddings_chunk = model(input_chunk, attention_mask=attention_chunk)[0]
            # Move results back to CPU for concatenation
            all_embeddings.append(embeddings_chunk.cpu())
    all_embeddings = torch.cat(all_embeddings, dim=0)
    torch.save(all_embeddings, os.path.join(DATA_DIR, 'Sentiment', 'embeddings.pt'))

    # Create token embeddings dictionary for index mapping
    unique_token_ids = input_ids.unique().sort()[0]
    token_to_embedding_dict = {}
    for token_id in unique_token_ids:
        if token_id == tokenizer.pad_token_id:
            token_embedding = torch.zeros(model.config.hidden_size)
        else:
            token_embedding = model.embeddings.word_embeddings(torch.tensor([token_id], device=DEVICE))
        token_to_embedding_dict[token_id.item()] = token_embedding.squeeze(0).detach().cpu()

    token_ids = list(token_to_embedding_dict.keys())
    embeddings = [token_to_embedding_dict[token_id] for token_id in token_ids]
    embedding_tensor = torch.stack(embeddings)
    token_to_index = {token_id: index for index, token_id in enumerate(token_ids)}
    torch.save(
        {'token_to_index': token_to_index, 'embeddings': embedding_tensor},
        os.path.join(DATA_DIR, 'Sentiment', 'token_to_index_and_embeddings.pth')
    )

    # Convert token IDs to embedding indices
    token_indices = convert_ids_to_embedding_indices(input_ids, token_to_index, default_index=0)

    # Further refine: split data per user (e.g. users ranked 35-50 by tweet count)
    df_used.reset_index(drop=True, inplace=True)
    device_fl_users = df_used['user'].value_counts().index[-15:]
    df_device = df_used[df_used['user'].isin(device_fl_users)]
    for i, user in enumerate(device_fl_users):
        df_user = df_device[df_device['user'] == user]
        torch.save({
            'data': token_indices[df_user.index],
            'labels': torch.tensor(df_user['target'].values),
            'masks': attention_masks[df_user.index]
        }, os.path.join(DATA_DIR, 'Sentiment', f'data_device_{i}_indices.pth'))

    print("sentiment dataset embeddings and token indices have been saved.")


# ---------------- MIMIC Dataset Processing ----------------

class MimicEmbedDataset(Dataset):
    """Dataset for embedding clinical notes using ClinicalBERT."""
    def __init__(self, input_ids, attention_masks):
        self.input_ids = input_ids
        self.attention_masks = attention_masks

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx]


def create_mimic_dataset():
    # ----- Process Admissions -----
    admissions_path = os.path.join(DATA_DIR, 'mimic_iii', 'ADMISSIONS.csv')
    admissions = pd.read_csv(admissions_path)
    emergency_adm = admissions.loc[
        (admissions['HOSPITAL_EXPIRE_FLAG'].isin([0, 1])) &
        (admissions['ADMISSION_TYPE'] == 'EMERGENCY'),
        ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'HOSPITAL_EXPIRE_FLAG', 'DIAGNOSIS']
    ]

    # Define diagnosis groups
    infection_dx = ['PNEUMONIA','PNEUMONIA;CHRONIC OBST PULM DISEASE',
                    'RESPIRATORY FAILURE', 'SEPSIS',
                'URINARY TRACT INFECTION;PYELONEPHRITIS','SEPSIS;TELEMETRY',
                    'PNEUMONIA;TELEMETRY','URINARY TRACT INFECTION','UTI/PYELONEPHRITIS',
                'SEPTIC SHOCK']
    mi_dx = ['CORONARY ARTERY DISEASE', 'CHEST PAIN',
            'CORONARY ARTERY DISEASE\CATH', 'ACUTE CORONARY SYNDROME',
            'MYOCARDIAL INFARCTION','ACUTE MYOCARDIAL INFARCTION','CHEST PAIN;TELEMETRY',
            'MYOCARDIAL INFARCTION\CATH', 'UNSTABLE ANGINA', 'STEMI',
            'ST ELEVATED MYOCARDIAL INFARCTION','ACUTE MYOCARDIAL INFARCTION\CATH',
            'UNSTABLE ANGINA\CATH','NON-ST SEGMENT ELEVATION MYOCARDIAL INFARCTION']
    brain_dx = ['INTRACRANIAL HEMORRHAGE','SUBARACHNOID HEMORRHAGE',
                'SUBDURAL HEMATOMA','STROKE;TELEMETRY;TRANSIENT ISCHEMIC ATTACK',
                'ACUTE SUBDURAL HEMATOMA','CEREBROVASCULAR ACCIDENT', 'CARDIAC ARREST',
                'S/P CARDIAC ARREST','SUBDURAL HEMORRHAGE'
            'STROKE;TELEMETRY','INTRACRANIAL BLEED','STROKE','STROKE/TIA']
    gi_dx = ['GASTROINTESTINAL BLEED','UPPER GI BLEED','ABDOMINAL PAIN',
            'UPPER GASTROINTESTINAL BLEED','LOWER GASTROINTESTINAL BLEED']
    all_dx = infection_dx + mi_dx + brain_dx + gi_dx

    # Filter for emergency admissions with one of the diagnoses
    emergency_adm = emergency_adm[emergency_adm['DIAGNOSIS'].isin(all_dx)]

    # Assign diagnosis groups
    def assign_dx_group(dx):
        if dx in infection_dx:
            return 'infection'
        elif dx in mi_dx:
            return 'mi'
        elif dx in brain_dx:
            return 'brain'
        elif dx in gi_dx:
            return 'gi'
    emergency_adm['DX_GROUP'] = emergency_adm['DIAGNOSIS'].apply(assign_dx_group)

    # Create LOS bins (length of stay)
    emergency_adm['ADMITTIME'] = pd.to_datetime(emergency_adm['ADMITTIME'])
    emergency_adm['DISCHTIME'] = pd.to_datetime(emergency_adm['DISCHTIME'])
    emergency_adm['Length_of_Stay'] = (emergency_adm['DISCHTIME'] - emergency_adm['ADMITTIME']).dt.days
    bins = [-1, 7, float('inf')]
    emergency_adm['LOS'] = pd.cut(emergency_adm['Length_of_Stay'], bins=bins, labels=[0, 1])
    emergency_adm.dropna(inplace=True)
    emergency_adm['LOS'] = emergency_adm['LOS'].astype(int)

    # ----- Process Clinical Notes -----
    notes_cols = ['SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CHARTTIME', 'CATEGORY', 'TEXT']
    notes_path = os.path.join(DATA_DIR, 'mimic_iii', 'NOTEEVENTS.csv')
    notes = pd.read_csv(notes_path, usecols=notes_cols)
    notes = notes[notes['HADM_ID'].isin(emergency_adm['HADM_ID'].unique())]
    notes = notes[notes['CHARTTIME'].notna()]
    notes_date = notes.merge(
        emergency_adm[['HADM_ID', 'ADMITTIME', 'DX_GROUP', 'LOS', 'HOSPITAL_EXPIRE_FLAG']],
        on='HADM_ID'
    )
    notes_date['CHARTTIME'] = pd.to_datetime(notes_date['CHARTTIME'])
    time_diff = notes_date['CHARTTIME'] - notes_date['ADMITTIME']
    filtered_notes = notes_date[time_diff <= pd.Timedelta(hours=24)]
    filtered_notes = filtered_notes[filtered_notes['CATEGORY'].isin(['Nursing', 'Physician', 'Nursing/other'])]

    print(f"Total patients: {emergency_adm['HADM_ID'].nunique()}, "
          f"Patients with notes: {filtered_notes['HADM_ID'].nunique()}")

    # Tokenise clinical notes using ClinicalBERT
    tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
    model = AutoModel.from_pretrained("medicalai/ClinicalBERT")
    # Sort and drop duplicate notes for each patient
    features = filtered_notes.sort_values(by='ADMITTIME').drop_duplicates(['CHARTTIME', 'HADM_ID'])
    features['token_length'] = features['TEXT'].apply(lambda x: len(tokenizer.tokenize(x)))
    input_ids = [tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=512)
                 for text in features['TEXT']]
    input_ids = pad_sequence([torch.tensor(seq) for seq in input_ids], batch_first=True)
    attention_masks = (input_ids != tokenizer.pad_token_id).long()

    # Compute embeddings using DataLoader with GPU processing
    dataset = MimicEmbedDataset(input_ids, attention_masks)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = model.to(DEVICE)
    model.eval()
    all_embeddings_list = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating MIMIC Embeddings"):
            input_ids_batch, attention_masks_batch = batch
            input_ids_batch = input_ids_batch.to(DEVICE)
            attention_masks_batch = attention_masks_batch.to(DEVICE)
            embeddings_batch = model(input_ids_batch, attention_mask=attention_masks_batch)[0]
            all_embeddings_list.append(embeddings_batch.cpu())
    all_embeddings = torch.cat(all_embeddings_list, dim=0)
    torch.save(all_embeddings, os.path.join(DATA_DIR, 'mimic_iii', 'embeddings.pt'))

    # Create token embeddings dictionary for indices
    unique_token_ids = input_ids.unique().sort()[0]
    token_to_embedding_dict = {}
    for token_id in unique_token_ids:
        if token_id == tokenizer.pad_token_id:
            token_embedding = torch.zeros(model.config.hidden_size)
        else:
            token_embedding = model.embeddings.word_embeddings(torch.tensor([token_id], device=DEVICE))

        token_to_embedding_dict[token_id.item()] = token_embedding.squeeze(0).detach().cpu()

    token_ids = list(token_to_embedding_dict.keys())
    embeddings = [token_to_embedding_dict[token_id] for token_id in token_ids]
    embedding_tensor = torch.stack(embeddings)
    token_to_index = {token_id: index for index, token_id in enumerate(token_ids)}
    torch.save(
        {'token_to_index': token_to_index, 'embeddings': embedding_tensor},
        os.path.join(DATA_DIR, 'mimic_iii', 'token_to_index_and_embeddings.pth')
    )

    token_indices = convert_ids_to_embedding_indices(input_ids, token_to_index, default_index=0)

    # ----- Create Datasets for Each Diagnosis Group -----
    features.reset_index(drop=True, inplace=True)
    features['HADM_ID'] = features['HADM_ID'].astype(int)
    features['token_length'] = features['TEXT'].apply(lambda x: len(tokenizer.tokenize(x)))
    df_sorted = features.reset_index().sort_values(by='token_length', ascending=False)
    df_used = df_sorted.groupby('HADM_ID').first().reset_index()

    for dx in ['infection', 'gi', 'brain', 'mi']:
        df_dx = df_used[df_used['DX_GROUP'] == dx]
        pt_order = df_dx['index'].values
        out_file = os.path.join(DATA_DIR, 'mimic_iii', f'dataset_{dx}_indices.pt')
        torch.save({
            'data': token_indices[pt_order],
            'labels': {
                "LOS": torch.tensor(df_dx['LOS'].values),
                "Mortality": torch.tensor(df_dx['HOSPITAL_EXPIRE_FLAG'].values)
            },
            'masks': attention_masks[pt_order]
        }, out_file)

    print("MIMIC dataset embeddings and diagnosis datasets have been saved.")


def main(args):
    """Runs the processing functions for selected datasets based on args."""
    print("Starting dataset creation process...")

    # Determine which datasets to process based on arguments
    run_isic = args.isic
    run_sentiment = args.sentiment
    run_mimic = args.mimic
    run_benchmark = args.benchmark

    # --- Default behavior: If no specific dataset flag is true, run all ---
    run_all_default = not (run_isic or run_sentiment or run_mimic or run_benchmark)
    if run_all_default:
        print("No specific dataset selected via command line, processing ALL datasets by default.")
        run_isic = True
        run_sentiment = True
        run_mimic = True
        run_benchmark = True
    else:
        print("Processing datasets selected via command line arguments.")

    # Load benchmark images if requested
    if run_benchmark:
        try:
            benchmark_datasets = _load_benchmark_images()
            print(f"Successfully loaded benchmark datasets")
        except Exception as e:
            print(f"\n--- ERROR loading benchmark images: {e} ---")

    if run_isic:
        print("\n================ Processing ISIC ================")
        try:
            create_isic_dataset()
        except Exception as e:
            print(f"\n--- ERROR processing ISIC dataset: {e} ---")

    if run_sentiment:
        print("\n================ Processing sentiment ================")
        try:
            create_sentiment_dataset()
        except Exception as e:
            print(f"\n--- ERROR processing sentiment dataset: {e} ---")

    if run_mimic:
        print("\n================ Processing MIMIC-III ================")
        try:
            create_mimic_dataset()
        except Exception as e:
            print(f"\n--- ERROR processing MIMIC dataset: {e} ---")

    print("\nDataset creation process finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create datasets for ISIC, sentiment, MIMIC, and/or benchmark datasets. Runs all if no specific dataset is selected.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--isic',
        action='store_true',
        help='Flag to process the ISIC dataset.'
    )
    parser.add_argument(
        '--sentiment',
        action='store_true',
        help='Flag to process the sentiment dataset.'
    )
    parser.add_argument(
        '--mimic',
        action='store_true',
        help='Flag to process the MIMIC-III dataset.'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Flag to download and load benchmark image datasets (EMNIST, CIFAR10, FashionMNIST).'
    )

    args = parser.parse_args()

    main(args)


