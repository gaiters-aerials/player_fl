"""
Sets up and manages logging for the project.

Provides a centralized `PerformanceLogger` class to create and access
loggers for different components (server, clients) and datasets,
directing output to both console and timestamped files. Includes
decorators for logging execution time and GPU stats.
"""
from helper import *  # Imports cleanup_gpu, etc. if needed, but primarily configs
from configs import * # Imports ROOT_DIR, DEVICE, etc.


class PerformanceLogger:
    """
    Manages the creation and retrieval of logger instances.

    Ensures that logs are directed to both the console and specific,
    timestamped log files based on dataset and component name.
    """
    def __init__(self, log_dir: str = 'code/evaluation/logs/python_logs'):
        """
        Initializes the PerformanceLogger.

        Args:
            log_dir (str): The relative path within the project root where
                           log files should be stored. Defaults to
                           'code/evaluation/logs/python_logs'.
        """
        self.log_dir = os.path.join(ROOT_DIR, log_dir) # Construct full path
        os.makedirs(self.log_dir, exist_ok=True) # Create log directory if it doesn't exist
        self.loggers = {} # Cache for created loggers

    def get_logger(self, dataset: str, name: str) -> logging.Logger:
        """
        Retrieves or creates a logger instance for a specific dataset and component name.

        Configures the logger with handlers for console output and file output.
        Log files are named using the dataset, component name, and timestamp.

        Args:
            dataset (str): The name of the dataset context (e.g., 'FMNIST', 'server', 'client_1').
                           Used for differentiating log files.
            name (str): A specific name for the logger (e.g., 'evaluation', 'lr_tuning',
                        algorithm name). Used in the logger name and filename.

        Returns:
            logging.Logger: The configured logger instance.
        """
        # Create a unique identifier for the logger instance
        logger_name = f"{dataset}_{name}"

        if logger_name not in self.loggers:
            # Create a new logger if it doesn't exist
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.INFO) # Set default logging level (e.g., INFO, DEBUG)

            # Prevent duplicate handlers if logger is somehow retrieved multiple times externally
            if logger.hasHandlers():
                logger.handlers.clear()

            # --- File Handler ---
            # Create a timestamp for the log file name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f'{logger_name}_{timestamp}.log'
            log_filepath = os.path.join(self.log_dir, log_filename)
            # Use 'a' for append mode or 'w' for write mode (overwrites existing)
            file_handler = logging.FileHandler(log_filepath, mode='w')
            file_handler.setLevel(logging.INFO) # Log INFO and above to file

            # --- Console Handler ---
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO) # Log INFO and above to console

            # --- Formatter ---
            # Define the format for log messages
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S' # Define date format
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # --- Add Handlers ---
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

            # Prevent propagation to root logger if desired
            logger.propagate = False

            # Cache the logger instance
            self.loggers[logger_name] = logger
            # print(f"Initialized logger '{logger_name}' logging to {log_filepath}") # Debug message

        # Return the existing or newly created logger
        return self.loggers[logger_name]


# --- Decorators ---

def log_execution_time(func):
    """
    Decorator to log the execution time of a function.

    Must be applied to a method of a class that has a `self.logger` attribute
    (which should be a configured `logging.Logger` instance).

    Args:
        func (Callable): The function or method to wrap.

    Returns:
        Callable: The wrapped function/method.
    """
    @wraps(func) # Preserves original function metadata (name, docstring)
    def wrapper(self, *args, **kwargs):
        # Check if the logger exists on the instance
        if not hasattr(self, 'logger') or not isinstance(self.logger, logging.Logger):
             print(f"Warning: @log_execution_time used on {func.__name__} but 'self.logger' is not a valid Logger instance.")
             return func(self, *args, **kwargs) # Execute function without logging time

        start_time = time.time()
        # Execute the original function
        result = func(self, *args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        # Log the execution time using the instance's logger
        self.logger.info(f"Method '{func.__name__}' completed in {execution_time:.2f} seconds")
        return result
    return wrapper


def log_gpu_stats(logger: logging.Logger, prefix: str = ""):
    """
    Logs current GPU memory usage if CUDA is available.

    Args:
        logger (logging.Logger): The logger instance to use for output.
        prefix (str): An optional string to prepend to the log message
                      (e.g., "Before training: "). Defaults to "".
    """
    if torch.cuda.is_available():
        # Get memory stats in bytes, convert to megabytes (MB)
        allocated_bytes = torch.cuda.memory_allocated()
        reserved_bytes = torch.cuda.memory_reserved() # Total memory reserved by allocator
        allocated_mb = allocated_bytes / (1024**2)
        reserved_mb = reserved_bytes / (1024**2)
        # Log using DEBUG level as it can be verbose
        logger.debug(
            f"{prefix}GPU Memory: "
            f"Allocated={allocated_mb:.1f}MB, "
            f"Reserved={reserved_mb:.1f}MB"
        )
    else:
        logger.debug(f"{prefix}GPU not available, skipping memory stats.")