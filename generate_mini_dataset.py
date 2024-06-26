import os
import shutil
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

# Function to sample and copy images
def sample_and_copy_images(csv_path, images_path, output_images_path, output_csv_path, sample_size):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Sample the data
    sampled_df = df.sample(frac=sample_size, random_state=42)  # Use a fixed random state for reproducibility
    
    # Copy the sampled images to the new directory
    for idx, row in sampled_df.iterrows():
        image_name = row['image_file_name']  # Use the correct column name
        src_path = os.path.join(images_path, image_name)
        dest_path = os.path.join(output_images_path, image_name)
        
        # Check if the source file exists
        if not os.path.exists(src_path):
            print(f"Warning: Source file '{src_path}' does not exist. Skipping.")
            continue
        
        # Copy the file
        shutil.copyfile(src_path, dest_path)
    
    # Save the sampled CSV
    sampled_df.to_csv(output_csv_path, index=False)

# Function to create mini dataset with train-validation-test split
def create_mini_dataset(data_path, mini_dataset_path, sample_size, train_val_split_ratio=0.1):
    # Paths for images and CSV files
    train_images_path = os.path.join(data_path, 'images_train')
    test_images_path = os.path.join(data_path, 'images_test')
    train_csv_path = os.path.join(data_path, 'train.csv')
    test_csv_path = os.path.join(data_path, 'test.csv')
    
    # Output paths for train, validation, and test
    mini_train_images_path = os.path.join(mini_dataset_path, 'images_train')
    mini_val_images_path = os.path.join(mini_dataset_path, 'images_val')
    mini_test_images_path = os.path.join(mini_dataset_path, 'images_test')
    
    mini_train_csv_path = os.path.join(mini_dataset_path, 'train.csv')
    mini_val_csv_path = os.path.join(mini_dataset_path, 'val.csv')
    mini_test_csv_path = os.path.join(mini_dataset_path, 'test.csv')
    
    # Create directories for the mini dataset
    os.makedirs(mini_train_images_path, exist_ok=True)
    os.makedirs(mini_val_images_path, exist_ok=True)
    os.makedirs(mini_test_images_path, exist_ok=True)
    
    # Sample and copy images for test set
    sample_and_copy_images(test_csv_path, test_images_path, mini_test_images_path, mini_test_csv_path, sample_size)
    
    # Sample and copy images for training set
    sample_and_copy_images(train_csv_path, train_images_path, mini_train_images_path, mini_train_csv_path, sample_size)
    
    # Split mini training set into training and validation
    mini_train_df = pd.read_csv(mini_train_csv_path)
    mini_train_images = os.listdir(mini_train_images_path)
    
    # Split the remaining training data into training and validation sets
    train_images, val_images = train_test_split(mini_train_images, test_size=train_val_split_ratio, random_state=42)
    
    # Copy validation images
    for image_name in val_images:
        src_path = os.path.join(mini_train_images_path, image_name)
        dest_path = os.path.join(mini_val_images_path, image_name)
        shutil.copyfile(src_path, dest_path)
    
    # Filter DataFrame to match the selected images for validation set
    mini_val_df = mini_train_df[mini_train_df['image_file_name'].isin(val_images)]
    
    # Filter DataFrame to match the selected images for training set
    mini_train_df = mini_train_df[mini_train_df['image_file_name'].isin(train_images)]
    
    # Save mini training CSV
    mini_train_df.to_csv(mini_train_csv_path, index=False)
    
    # Save mini validation CSV
    mini_val_df.to_csv(mini_val_csv_path, index=False)

# Main function
def main():
    parser = argparse.ArgumentParser(description="Create mini dataset")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset (e.g., flowers, fashion)")
    parser.add_argument('--sample_size', type=float, default=0.02, help="Fraction of the dataset to sample")
    args = parser.parse_args()

    data_path = f"data/{args.dataset}"
    mini_dataset_path = f"data/{args.dataset}_mini"
    
    create_mini_dataset(data_path, mini_dataset_path, args.sample_size, train_val_split_ratio=0.1)
    print(f"Mini dataset for {args.dataset} created successfully.")

if __name__ == "__main__":
    main()
