import os
import h5py
import nibabel as nib
import numpy as np

# Base path where patient folders are organized
organized_base = r"E:\Dataset\PICAI\workdir\workdir\csPCa_patients\PICAI_dataset"

# List all patient IDs (folder names that are digits)
patient_ids = [name for name in os.listdir(organized_base)
               if os.path.isdir(os.path.join(organized_base, name)) and name.isdigit()]

for pid in patient_ids:
    try:
        patient_folder = os.path.join(organized_base, pid)
        t2w_path = os.path.join(patient_folder, "t2w", f"{pid}_t2w.nii.gz")
        adc_path = os.path.join(patient_folder, "adc", f"{pid}_adc.nii.gz")
        hbv_path = os.path.join(patient_folder, "hbv", f"{pid}_hbv.nii.gz")
        seg_path = os.path.join(patient_folder, "seg", f"{pid}_seg.nii.gz")

        # Load images using nibabel
        t2w = nib.load(t2w_path).get_fdata()
        adc = nib.load(adc_path).get_fdata()
        hbv = nib.load(hbv_path).get_fdata()
        seg = nib.load(seg_path).get_fdata()

        # Normalize image modalities
        def norm(img): return (img - np.mean(img)) / np.std(img) if np.std(img) != 0 else img
        t2w, adc, hbv = map(norm, [t2w, adc, hbv])

        # Save the HDF5 file in the patient's folder
        h5_path = os.path.join(patient_folder, f"{pid}.h5")
        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset("image/t2w", data=t2w, compression="gzip")
            hf.create_dataset("image/adc", data=adc, compression="gzip")
            hf.create_dataset("image/hbv", data=hbv, compression="gzip")
            hf.create_dataset("label/seg", data=seg.astype(np.uint8), compression="gzip")

        print(f"  Converted {pid} → saved to {h5_path}")

    except Exception as e:
        print(f"  Error with patient {pid}: {e}")

print("\n All valid patient folders converted to .h5 format inside their respective folders.")
