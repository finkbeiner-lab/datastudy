#Batchcopy & save
import os
import pdb
import shutil
import glob

# Source and destination directories
source_dir = "/gladstone/finkbeiner/robodata/IXM4Galaxy/Una/JAK-GSGT30-Plate1-GEDI"
dest_mor4_dir = "/gladstone/finkbeiner/robodata/IXM4Galaxy/Heather/GEDI/JAK-MOR4-GEDI"
dest_mor5_dir = "/gladstone/finkbeiner/robodata/IXM4Galaxy/Heather/GEDI/JAK-MOR5-GEDI"


# Function to extract well number from filename
def extract_well_number(filename):
    # pdb.set_trace()


    return filename.split("_")[4] # For IXM files


# Function to move files based on their filenames
def move_files(source_dir, dest_dir, keyword):

    for filename in glob.glob(os.path.join(source_dir,"*/")):
        well_images = glob.glob(os.path.join(filename,"*.tif"))

        for well_image in well_images:
            if keyword in well_image:
                print('-----Keyword')
                print('\n-----well_image')
                well_number = extract_well_number(well_image)
                # pdb.set_trace()


                source_file_path = os.path.join(source_dir, filename)
                dest_subfolder = os.path.join(dest_dir,  well_number)
                if not os.path.exists(dest_subfolder):
                    os.makedirs(dest_subfolder)
                dest_file_path = os.path.join(dest_subfolder, os.path.basename(source_file_path))
                print("Moving file from:", source_file_path)

                print("Moving file to:", dest_file_path)
                try:

                    shutil.move(well_image, dest_file_path)
                    print("File moved successfully.")

                except Exception as e:
                    print("Error moving file:", e)
            else:
                print("Skipping file:", well_image, "as it does not contain the keyword.")

            # shutil.move(source_file_path, dest_file_path)


# Move files to MOR4 destination # Eg. For Heather's dataset
move_files(source_dir, dest_mor4_dir, "JAK-MOR4")

# Move files to MOR5 destination #Eg. For Heather's dataset
move_files(source_dir, dest_mor5_dir, "JAK-MOR5")


