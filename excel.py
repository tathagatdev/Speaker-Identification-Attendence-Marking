import pandas as pd
import os

modelpath = os.path.join("C:\\Users\\DELL\\Desktop\\Mini-Project\\Speaker-identification-using-GMMs", "dest_models")

# Debugging output to check files in the directory
print("Files in 'dest_models':", os.listdir(modelpath))
speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname in os.listdir(modelpath) if fname.endswith('.gmm')]

def update_excel(file_path, detected_speaker):
    # Read the existing Excel sheet or create a new one
    excel_file = "C:\\Users\\DELL\\Desktop\\Mini-Project\\identified_speakers.xlsx"
    try:
        df = pd.read_excel(excel_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["File", "Detected Speaker"])

    # Check if the detected_speaker index is within the range of speakers
    if detected_speaker < len(speakers):
        speaker_name = speakers[detected_speaker]
        # Append the new information
        new_row = {"File": file_path, "Detected Speaker": speaker_name}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # Save the updated DataFrame to the Excel file
        df.to_excel(excel_file, index=False)
    else:
        print(f"Detected speaker index {detected_speaker} is out of range.")

if __name__ == "__main__":
    # Example usage:
    # You can call this function in your main application after detecting the speaker.
    # Replace "example_file.wav" and 0 with your actual file path and detected speaker index.
    update_excel("example_file.wav", 0)
