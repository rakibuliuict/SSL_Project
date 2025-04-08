import os
import shutil

# Base directory containing all patient folders
base_dir = r"E:\Dataset\PICAI\workdir\workdir\csPCa_patients\PICAI_dataset"

# Loop through all items in the base directory
for patient_id in os.listdir(base_dir):
    patient_path = os.path.join(base_dir, patient_id)
    
    # Skip non-directories
    if not os.path.isdir(patient_path) or not patient_id.isdigit():
        continue

    print(f"Processing patient: {patient_id}")
    
    # Modality folder names and final file names
    modalities = ['t2w', 'adc', 'hbv', 'seg']
    
    for mod in modalities:
        mod_folder = os.path.join(patient_path, mod)
        expected_file = f"{patient_id}_{mod}.nii.gz"
        source_file = os.path.join(mod_folder, expected_file)
        dest_file = os.path.join(patient_path, f"{mod}.nii.gz")

        if os.path.exists(source_file):
            try:
                shutil.move(source_file, dest_file)
                print(f" Moved {expected_file} to {dest_file}")
                os.rmdir(mod_folder)  # Remove empty modality folder
            except Exception as e:
                print(f" Error moving {expected_file}: {e}")
        else:
            print(f" File not found: {source_file}")

print("\n All patient folders flattened successfully.")
