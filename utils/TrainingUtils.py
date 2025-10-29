from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

def CreateTensorBoardCallback(name:str):
    # Create a log directory for storing TensorBoard logs
    logdir = Path(f"logs/{name}/{datetime.now().strftime("%Y-%m-%d-%H%M%S")}")
    return TensorBoard(logdir)

def CreateCircuitBreakerCallback(monitor:str="accuracy", mode:str='max', patience:int = 3):
    """
    Early-stopping callback which could help prevent overfitting by stopping the training once certain evaluation metrics stop improving / start to plateau.
    """
    return EarlyStopping(monitor=monitor, mode=mode, patience=patience)