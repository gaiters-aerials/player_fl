from configs import *

class UnifiedDataLoader:
    """
    Unified data loader that handles multiple data formats and prepares them
    for the DataPartitioner and DataPreprocessor pipeline
    """
    def __init__(self, root_dir, dataset_name):
        self.root_dir = root_dir
        self.data_dir = f'{root_dir}/data'
        self.dataset_name = dataset_name
        
    def load(self):
        """
        Load data and convert to a standardized DataFrame format
        Returns DataFrame with 'data' as numpy array, 'label' as numpy array, 
        and 'site' as integer
        """
        if self.dataset_name in ['EMNIST', 'CIFAR', 'FMNIST']:
            return self._load_benchmark_images()
        elif self.dataset_name == 'ISIC':
            return self._load_isic()
        elif self.dataset_name == 'Sentiment':
            return self._load_sentiment()
        elif self.dataset_name == 'mimic':
            return self._load_mimic()
        elif self.dataset_name == 'Heart':
            return self._load_heart()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def _load_benchmark_images(self):
        """Handle torchvision datasets"""
        dataset_classes = {
            'EMNIST': lambda: EMNIST(f'{self.data_dir}/EMNIST', split='byclass', download=False, train=True),
            'CIFAR': lambda: CIFAR10(f'{self.data_dir}/CIFAR10', download=False, train=True),
            'FMNIST': lambda: FashionMNIST(f'{self.data_dir}/FMNIST', download=False, train=True),
        }
        
        dataset = dataset_classes[self.dataset_name]()
        data = dataset.data
        labels = dataset.targets
        if isinstance(data, torch.Tensor):
            data = data.numpy()
            labels = labels.numpy()
        elif data.shape[-3:] == (32,32,3):
            data = data.transpose((0,3,1,2))
        
        return pd.DataFrame({
            'data': list(data),
            'label': labels,
            'site': np.zeros(len(labels)) 
        })

    def _load_isic(self):
        """Handle ISIC image dataset"""
        all_data = []
        
        for site in range(6):
            file_path = f'{self.data_dir}/ISIC/site_{site}_metadata.csv'
            files = pd.read_csv(file_path)
            # Store full image paths
            image_files = [f'{self.data_dir}/ISIC/ISIC_2019_Training_Input_preprocessed/{file}.jpg' 
                          for file in files['image']]
            
            df = pd.DataFrame({
                'data': image_files,
                'label': files['target'].values,
                'site': np.full(len(files), site)
            })
            all_data.append(df)
        return pd.concat(all_data, ignore_index=True)

    def _load_sentiment(self):
        """Handle Sentiment dataset with tensor dictionaries"""
        all_data = []
        
        for device in range(15):
            file_path = f'{self.data_dir}/Sentiment/data_device_{device}_indices.pth'
            site_data = torch.load(file_path)
            df = pd.DataFrame({
                'data': list(site_data['data'].numpy()),
                'label': site_data['labels'].numpy(),
                'mask': list(site_data['masks'].numpy()),
                'site': np.full(len(site_data['labels']), device)
            })
            all_data.append(df)
            
        return pd.concat(all_data, ignore_index=True)

    def _load_mimic(self):
        """Handle MIMIC dataset with tensor dictionaries"""
        all_data = []
        diagnoses = ['mi', 'gi', 'infection', 'brain']
        
        for i, dx in enumerate(diagnoses):
            file_path = f'{self.data_dir}/mimic_iii/dataset_concatenated_{dx}_indices.pt'
            site_data = torch.load(file_path)
            df = pd.DataFrame({
                'data': list(site_data['data'].numpy()),
                'label': site_data['labels']['Mortality'].numpy(),
                'mask': list(site_data['masks'].numpy()),
                'site': np.full(len(site_data['labels']['Mortality']), i)
            })
            all_data.append(df)
            
        return pd.concat(all_data, ignore_index=True)

    def _load_heart(self):
        """Handle Heart dataset from CSV files"""
        columns = ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 
                'sugar', 'ecg', 'max_hr', 'exercise_angina', 'exercise_ST_depression',
                    'slope_ST', 'number_major_vessels', 'thalassemia_hx', 'target']
        used_columns =  ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol',
                        'sugar', 'ecg', 'max_hr', 'exercise_angina', 'exercise_ST_depression',
                            'target']
            
        all_data = []
        sites = ['cleveland', 'hungarian', 'switzerland', 'va']
        
        for i, site in enumerate(sites):
            file_path = f'{self.data_dir}/Heart/processed.{site}.data'
            site_data = pd.read_csv(
                file_path,
                names=columns,
                na_values='?',
                 usecols=used_columns
            ).dropna()
            
            # Convert features to numpy arrays
            feature_cols = [col for col in used_columns if col != 'target']
            features = site_data[feature_cols].values
            
            df = pd.DataFrame({
                'data': list(features),  # Store each row as a numpy array
                'label': site_data['target'].values,
                'site': np.full(len(site_data), i)
            })
            all_data.append(df)
            
        return pd.concat(all_data, ignore_index=True)


class BaseDataset(Dataset):
    """Base class for all datasets"""
    def __init__(self, X, y, is_train=True):
        self.X = X
        self.y = y
        self.is_train = is_train
        self.transform = self.get_transform()
    
    def __len__(self):
        return len(self.y)
    
    def get_transform(self):
        """To be implemented by child classes"""
        raise NotImplementedError
    
    def __getitem__(self, idx):
        """To be implemented by child classes"""
        raise NotImplementedError


class EMNISTDataset(BaseDataset):
    """EMNIST dataset handler"""
    def get_transform(self):
        base_transform = [
            transforms.ToPILImage(), 
            transforms.Resize((28, 28)),
        ]
        
        base_transform_2 = [
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3317,))
        ]
        
        if self.is_train:
            augmentation = [
                transforms.RandomRotation((-15, 15)),
                transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            ]
            return transforms.Compose(base_transform + augmentation + base_transform_2)
        else:
            return transforms.Compose(base_transform + base_transform_2)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]

        image_tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor


class CIFARDataset(BaseDataset):
    """CIFAR-100 dataset handler"""
    def get_transform(self):
        base_transform = [
            transforms.ToPILImage(), 
            transforms.Resize(32),
        ]

        base_transform_2 = [
            transforms.ToTensor(),  
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ]

        if self.is_train:
            augmentation = [
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            ]
            return transforms.Compose(base_transform + augmentation + base_transform_2)
        else:
            return transforms.Compose(base_transform + base_transform_2)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]
        image = image.transpose(1, 2, 0)
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return image_tensor, label_tensor


class FMNISTDataset(BaseDataset):
    """Fashion MNIST dataset handler"""
    def get_transform(self):
        base_transform = [
            transforms.ToPILImage(), 
            transforms.Resize((28, 28)),
        ]
        
        base_transform_2 = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.2860],
                std=[0.3530]
            )
        ]

        if self.is_train:
            augmentation = [
                transforms.RandomRotation((-10, 10)),
                transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            ]
            return transforms.Compose(base_transform + augmentation + base_transform_2)
        else:
            return transforms.Compose(base_transform + base_transform_2)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]
        
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return image_tensor, label_tensor


class ISICDataset(BaseDataset):
    """ISIC dataset handler for skin lesion images"""
    def __init__(self, X, y, is_train=True):
        self.sz = 200  # Image size
        super().__init__(X, y, is_train)
    
    def get_transform(self):
        if self.is_train:
            return albumentations.Compose([
                albumentations.RandomScale(0.07),
                albumentations.Rotate(50),
                albumentations.RandomBrightnessContrast(0.15, 0.1),
                albumentations.Flip(p=0.5),
                albumentations.Affine(shear=0.1),
                albumentations.RandomCrop(self.sz, self.sz),
                albumentations.CoarseDropout(random.randint(1, 8), 16, 16),
                albumentations.Normalize(
                    mean=(0.585, 0.500, 0.486),
                    std=(0.229, 0.224, 0.225),
                    always_apply=True
                ),
            ])
        else:
            return albumentations.Compose([
                albumentations.CenterCrop(self.sz, self.sz),
                albumentations.Normalize(
                    mean=(0.585, 0.500, 0.486),
                    std=(0.229, 0.224, 0.225),
                    always_apply=True
                ),
            ])

    def __getitem__(self, idx):
        image_path = self.X[idx]
        label = self.y[idx]
        image = np.array(Image.open(image_path))
        
        transformed = self.transform(image=image)
        image = transformed['image']
        
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        label = torch.tensor(label, dtype=torch.int64)
        
        return image, label


class SentimentDataset(BaseDataset):
    """Sentiment dataset handler"""
    def __init__(self, X, y, masks, is_train=True):
        super().__init__(X, y, is_train)
        self.masks = masks
    
    def get_transform(self):
        return None

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        mask = self.masks[idx]
        return (x, mask), y


class MIMICDataset(BaseDataset):
    """MIMIC dataset handler"""
    def __init__(self, X, y, masks, is_train=True):
        super().__init__(X, y, is_train)
        self.masks = masks

    def get_transform(self):
        return None

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        mask = self.masks[idx]
        return (x, mask), y


class HeartDataset(BaseDataset):
    """Heart disease dataset handler"""
    # Define feature names
    FEATURE_NAMES = ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol',
                     'sugar', 'ecg', 'max_hr', 'exercise_angina', 'exercise_ST_depression']
    
    # These are the columns that should be scaled
    COLS_TO_SCALE = ['age', 'chest_pain_type', 'resting_bp', 'cholesterol',
                     'ecg', 'max_hr', 'exercise_ST_depression']
    
    def __init__(self, X, y, is_train=True, **kwargs):
        self.scaler = kwargs.get('scaler', None)
        
        # Get indices of columns to scale
        self.scale_indices = [self.FEATURE_NAMES.index(col) for col in self.COLS_TO_SCALE]
        
        if is_train:
            scaler = StandardScaler()
            
            # Initialize arrays for all features
            means = np.zeros(len(self.FEATURE_NAMES))
            variances = np.ones(len(self.FEATURE_NAMES))
            
            # Set the pre-computed values for columns that should be scaled
            scale_values = {
                'age': (53.0872973, 7.01459463e+01),
                'chest_pain_type': (3.23702703, 8.17756772e-01),
                'resting_bp': (132.74405405, 3.45493057e+02),
                'cholesterol': (220.23648649, 4.88430934e+03),
                'ecg': (0.64513514, 5.92069868e-01),
                'max_hr': (138.75459459, 5.29172208e+02),
                'exercise_ST_depression': (0.89532432, 1.11317517e+00)
            }
            
            # Update means and variances for scaled columns
            for col, (mean, var) in scale_values.items():
                idx = self.FEATURE_NAMES.index(col)
                means[idx] = mean
                variances[idx] = var
            
            scaler.mean_ = means
            scaler.var_ = variances
            scaler.scale_ = np.sqrt(variances)
            self.scaler = scaler
        
        super().__init__(X, y, is_train)
    
    def get_transform(self):
        return self.scaler
    
    def __getitem__(self, idx):
        features = self.X[idx].copy()
        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1)).flatten()
        
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return features, label
    
    def get_scalers(self):
        return {'scaler': self.scaler}


class DataPartitioner:
    """Handles partitioning of data across clients"""
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.num_sites, self.size_per_site, self.alpha = self._get_partition_params()

    def _get_partition_params(self):
        return DEFAULT_PARAMS[self.dataset_name]['num_clients'], DEFAULT_PARAMS[self.dataset_name]['sizes_per_client'], 0.5

    def partition_site_data(self, df):
        """Partition dataframe into client data dictionary"""
        if self.dataset_name in ['Heart', 'mimic', 'ISIC', 'Sentiment']:
            return self._natural_partition(df)
        else:
            return self._dirichlet_partition(df)

    def _natural_partition(self, df):
        """Split pre-existing sites into separate client data"""
        site_data = {}
        for site in range(self.num_sites):
            site_df = df[df['site'] == site]
            site_data[f'client_{site+1}'] = {
                'X': site_df['data'].values,
                'y': site_df['label'].values
            }
            if 'mask' in df.columns:
                site_data[f'client_{site+1}']['masks'] = site_df['mask'].values
        return site_data

    def _dirichlet_partition(self, df):
        """Split data into sites using Dirichlet distribution"""
        data = df['data'].values
        labels = df['label'].values
        
        total_size = self.size_per_site * self.num_sites
        data = data[:total_size]
        labels = labels[:total_size]

        split_data = self._split_by_dirichlet(data, labels)
        return {f'client_{i+1}': {'X': x, 'y': y} 
                for i, (x, y) in enumerate(split_data)}

    def _split_by_dirichlet(self, data, labels):
        """Split data using Dirichlet distribution"""
        unique_labels = np.unique(labels)
        client_data = [([], []) for _ in range(self.num_sites)]
        
        for label in unique_labels:
            label_mask = (labels == label)
            label_data = data[label_mask]
            label_count = len(label_data)
            
            proportions = np.random.dirichlet([self.alpha] * self.num_sites)
            client_sample_sizes = (proportions * label_count).astype(int)
            client_sample_sizes[-1] += label_count - client_sample_sizes.sum()
            
            start_idx = 0
            for client_idx, samples in enumerate(client_sample_sizes):
                end_idx = start_idx + samples
                client_data[client_idx][0].extend(label_data[start_idx:end_idx])
                client_data[client_idx][1].extend([label] * samples)
                start_idx = end_idx
                
        return client_data


class DataPreprocessor:
    """Handles dataset creation and preprocessing with support for masked data"""
    def __init__(self, dataset_name, batch_size):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.dataset_class = self._get_dataset_class()
        self.partitioner = DataPartitioner(self.dataset_name)
        
    def _get_dataset_class(self):
        dataset_classes = {
            'EMNIST': EMNISTDataset,
            'CIFAR': CIFARDataset,
            'FMNIST': FMNISTDataset,
            'ISIC': ISICDataset,
            'Sentiment': SentimentDataset,
            'mimic': MIMICDataset,
            'Heart': HeartDataset,
        }
        return dataset_classes[self.dataset_name]
    
    def process_client_data(self, df):
        """Process data for all clients"""
        partitioned_client_data = self.partitioner.partition_site_data(df)
        processed_data = {}
        
        for client_id, data in partitioned_client_data.items():
            train_loader, val_loader, test_loader = self._create_data_splits(data)
            processed_data[client_id] = (train_loader, val_loader, test_loader)
            
        return processed_data
    
    def _create_data_splits(self, data):
        """Create train/val/test splits handling both masked and unmasked data"""
        # Extract data components
        X, y = data['X'], data['y']
        masks = data.get('masks', None)
        
        # Split data while handling masks if present
        train_data, val_data, test_data = self._split_data(X, y, masks)

        # Create datasets with appropriate components
        train_dataset = self._create_dataset(train_data, is_train=True)
        scaler = getattr(train_dataset, 'get_scalers', lambda: {})()
        
        val_dataset = self._create_dataset(val_data, is_train=False, **scaler)
        test_dataset = self._create_dataset(test_data, is_train=False, **scaler)
        
        return (
            DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=2 ),
            DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=2 ),
            DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=2 )
        )
    
    def _create_dataset(self, data, is_train=True, **kwargs):
        """Create dataset instance handling presence/absence of masks"""
        if len(data) == 3:  # Data includes masks
            X, y, masks = data
            return self.dataset_class(X, y, masks=masks, is_train=is_train, **kwargs)
        else:  # Regular data without masks
            X, y = data
            return self.dataset_class(X, y, is_train=is_train, **kwargs)
    
    def _split_data(self, X, y, masks=None):
        """Split data into train/val/test sets, handling masks if present"""
        # Initial split for test set
        if masks is not None:
            X_temp, X_test, y_temp, y_test, masks_temp, masks_test = train_test_split(
                X, y, masks, test_size=0.2, random_state=42
            )
            # Split remaining data into train and validation
            X_train, X_val, y_train, y_val, masks_train, masks_val = train_test_split(
                X_temp, y_temp, masks_temp, test_size=0.2, random_state=42
            )
            return (
                (X_train, y_train, masks_train),
                (X_val, y_val, masks_val),
                (X_test, y_test, masks_test)
            )
        else:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.2, random_state=42
            )
            return (
                (X_train, y_train),
                (X_val, y_val),
                (X_test, y_test)
            )
    

