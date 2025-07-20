# High-Performance, Flexible Beat Extraction Script
# Processes a whole folder in parallel or a single file on demand.

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import os
import argparse
from alive_progress import alive_bar
from multiprocessing import Pool, cpu_count

# --- Library Imports ---
try:
    from BeatNet.BeatNet import BeatNet
except ImportError:
    BeatNet = None
try:
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
except ImportError:
    RNNBeatProcessor = DBNBeatTrackingProcessor = None

# --- Configuration for Folder Mode ---
MUSIC_FOLDER = 'music'
REPORTS_FOLDER = 'reports'

# --- Globals for Worker Processes ---
beat_tracker_pipeline = None
engine_in_worker = None

def init_worker(engine_name):
    """Initializer for each worker: loads the selected beat tracking model/pipeline."""
    global beat_tracker_pipeline, engine_in_worker
    engine_in_worker = engine_name
    
    if engine_in_worker == 'beatnet':
        if BeatNet is None:
            raise ImportError("BeatNet library is not installed. Please run 'pip install beatnet-v2'.")
        beat_tracker_pipeline = BeatNet(1, 'offline', 'DBN', None, False, 'cpu')
    elif engine_in_worker == 'madmom':
        if DBNBeatTrackingProcessor is None:
            raise ImportError("Madmom library is not installed correctly. Please ensure it's installed in your environment.")
        # Madmom uses a two-step pipeline, which we create here.
        beat_tracker_pipeline = (RNNBeatProcessor(), DBNBeatTrackingProcessor(fps=100))

def detect_beats(audio_path):
    """Detects beats using the globally loaded model, handling different library APIs."""
    try:
        if engine_in_worker == 'beatnet':
            predictions = beat_tracker_pipeline.process(audio_path)
            return predictions[:, 0] if isinstance(predictions, np.ndarray) and predictions.size > 0 else np.array([])
        elif engine_in_worker == 'madmom':
            # Unpack the two-step pipeline
            act_proc, beat_proc = beat_tracker_pipeline
            activations = act_proc(audio_path)
            return beat_proc(activations)
    except Exception as e:
        return f"Error during beat detection for {os.path.basename(audio_path)}: {e}"

def folder_mode_process_file(audio_filename):
    """Worker function for folder mode."""
    audio_path = os.path.join(MUSIC_FOLDER, audio_filename)
    base_name = os.path.splitext(audio_filename)[0]
    report_path = os.path.join(REPORTS_FOLDER, f"{base_name}_{engine_in_worker}_beats.txt")
    
    beat_times = detect_beats(audio_path)
    
    if isinstance(beat_times, str): return beat_times
    if len(beat_times) == 0: return f"Warning: No beats detected for '{audio_filename}'."

    try:
        np.savetxt(report_path, beat_times, fmt='%.4f')
        return "processed"
    except Exception as e:
        return f"Error writing report for '{audio_filename}': {e}"

def single_file_mode(filepath, engine):
    """Processes a single file without parallelization."""
    print(f"Initializing {engine} engine for single file processing...")
    
    try:
        if engine == 'beatnet':
            if BeatNet is None: raise ImportError("BeatNet not installed.")
            tracker = BeatNet(1, 'offline', 'DBN', None, False, 'cpu')
            print(f"Processing '{os.path.basename(filepath)}'...")
            predictions = tracker.process(filepath)
            beat_times = predictions[:, 0] if isinstance(predictions, np.ndarray) and predictions.size > 0 else np.array([])
        elif engine == 'madmom':
            if DBNBeatTrackingProcessor is None: raise ImportError("Madmom not installed.")
            act_proc = RNNBeatProcessor()
            beat_proc = DBNBeatTrackingProcessor(fps=100)
            print(f"Processing '{os.path.basename(filepath)}' (step 1 of 2)...")
            activations = act_proc(filepath)
            print(f"Processing '{os.path.basename(filepath)}' (step 2 of 2)...")
            beat_times = beat_proc(activations)
        else:
            beat_times = np.array([])
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return

    if len(beat_times) == 0:
        print("Warning: No beats were detected.")
        return

    base_name = os.path.splitext(os.path.basename(filepath))[0]
    output_filename = f"{base_name}_{engine}_beats.txt"

    try:
        np.savetxt(output_filename, beat_times, fmt='%.4f')
        print(f"\n✨ Successfully saved beat report to '{output_filename}'")
    except Exception as e:
        print(f"Error: Could not write report to file: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract beat timestamps from audio files.")
    parser.add_argument('-f', '--file', type=str, default=None, help="Path to a single audio file to process.")
    parser.add_argument('--engine', type=str, default='madmom', choices=['beatnet', 'madmom'], help="The beat detection engine to use.")
    args = parser.parse_args()

    # --- SINGLE FILE MODE ---
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: Input file not found at '{args.file}'")
            exit()
        single_file_mode(args.file, args.engine)
    
    # --- FOLDER (BATCH) MODE ---
    else:
        os.makedirs(MUSIC_FOLDER, exist_ok=True)
        os.makedirs(REPORTS_FOLDER, exist_ok=True)
        try:
            audio_files = [f for f in os.listdir(MUSIC_FOLDER) if f.lower().endswith(('.mp3', '.wav', '.opus'))]
        except FileNotFoundError:
            print(f"Error: The '{MUSIC_FOLDER}' directory was not found.")
            audio_files = []

        if not audio_files:
            print(f"No audio files found in the '{MUSIC_FOLDER}' folder.")
        else:
            num_files = len(audio_files)
            num_workers = min(cpu_count(), num_files)
            
            print(f"Found {num_files} file(s). Using '{args.engine}' engine with {num_workers} processes...")
            
            init_args = (args.engine,)
            
            try:
                with Pool(processes=num_workers, initializer=init_worker, initargs=init_args) as pool:
                    results_iterator = pool.imap_unordered(folder_mode_process_file, audio_files)
                    
                    with alive_bar(num_files, title=f"🎵 {args.engine.capitalize()} Beat Detection", bar='smooth', spinner='notes') as bar:
                        for result in results_iterator:
                            if result and result != "processed":
                                print(f"\n{result}")
                            bar()
                print(f"\n✨ Beat extraction complete! Reports saved in '{REPORTS_FOLDER}'. ✨")
            except Exception as e:
                print(f"\nA critical error occurred: {e}")