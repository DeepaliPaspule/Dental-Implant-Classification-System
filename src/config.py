import os

class Config:
    # Base paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'EntireDataset')
    
    # Model parameters
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # Categories
    CATEGORIES = ['endosteal', 'subperiosteal', 'transosteal', 'zygomatic']
    
    @classmethod
    def check_data_directory(cls):
        """Verify data directory structure"""
        print(f"\nChecking data directory structure:")
        print(f"Base directory: {cls.DATA_DIR}")
        
        if not os.path.exists(cls.DATA_DIR):
            print(f"❌ Data directory not found")
            return False
            
        valid_structure = True
        for category in cls.CATEGORIES:
            images_path = os.path.join(cls.DATA_DIR, category, 'images')
            print(f"\nChecking {category}:")
            print(f"Path: {images_path}")
            
            if not os.path.exists(images_path):
                print(f"❌ Images directory not found")
                valid_structure = False
                continue
                
            # Count images recursively
            image_count = 0
            for root, _, files in os.walk(images_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_count += 1
                        print(f"Found: {os.path.join(root, file)}")
            
            if image_count == 0:
                print(f"❌ No images found")
                valid_structure = False
            else:
                print(f"✓ Found {image_count} images")
                
        return valid_structure

    @classmethod
    def print_paths(cls):
        print("\nConfiguration paths:")
        print(f"BASE_DIR: {cls.BASE_DIR}")
        print(f"DATA_DIR: {cls.DATA_DIR}")
        print("\nVerifying paths:")
        print(f"DATA_DIR exists: {os.path.exists(cls.DATA_DIR)}")
        if os.path.exists(cls.DATA_DIR):
            print("Contents of DATA_DIR:")
            for item in os.listdir(cls.DATA_DIR):
                print(f"  - {item}")