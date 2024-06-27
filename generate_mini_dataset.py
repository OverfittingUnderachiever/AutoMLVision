import os
import shutil
import pandas as pd
import argparse

#FIRST DOWNLOAD THE FULL DATASET!
#etc
#py run.py --dataset flowers



# Function to sample and copy images
def sample_and_copy_images(csv_path, images_path, output_images_path, output_csv_path, sample_size):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Sample the data
    sampled_df = df.sample(frac=sample_size)
    
    # Copy the sampled images to the new directory
    for idx, row in sampled_df.iterrows():
        image_name = row['image_file_name']  # Use the correct column name
        src_path = os.path.join(images_path, image_name)
        dest_path = os.path.join(output_images_path, image_name)
        shutil.copyfile(src_path, dest_path)
    
    # Save the sampled CSV
    sampled_df.to_csv(output_csv_path, index=False)

# Function to create mini dataset
def create_mini_dataset(data_path, mini_dataset_path, sample_size):
    # Paths for images and CSV files
    train_images_path = os.path.join(data_path, 'images_train')
    test_images_path = os.path.join(data_path, 'images_test')
    train_csv_path = os.path.join(data_path, 'train.csv')
    test_csv_path = os.path.join(data_path, 'test.csv')
    
    # Output paths
    mini_train_images_path = os.path.join(mini_dataset_path, 'images_train')
    mini_test_images_path = os.path.join(mini_dataset_path, 'images_test')
    mini_train_csv_path = os.path.join(mini_dataset_path, 'train.csv')
    mini_test_csv_path = os.path.join(mini_dataset_path, 'test.csv')
    
    # Create directories for the mini dataset
    os.makedirs(mini_train_images_path, exist_ok=True)
    os.makedirs(mini_test_images_path, exist_ok=True)
    
    # Sample and copy images for train and test sets
    sample_and_copy_images(train_csv_path, train_images_path, mini_train_images_path, mini_train_csv_path, sample_size)
    sample_and_copy_images(test_csv_path, test_images_path, mini_test_images_path, mini_test_csv_path, sample_size)

# Main function
def main():
    parser = argparse.ArgumentParser(description="Create mini dataset")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset (e.g., flowers, fashion)")
    parser.add_argument('--sample_size', type=float, default=0.01, help="Fraction of the dataset to sample")
    args = parser.parse_args()

    data_path = f"data/{args.dataset}"
    mini_dataset_path = f"data/{args.dataset}_mini"
    
    create_mini_dataset(data_path, mini_dataset_path, args.sample_size)
    print(f"Mini dataset for {args.dataset} created successfully.")

if __name__ == "__main__":
    main()
