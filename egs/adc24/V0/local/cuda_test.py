import torch
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



print(torch.cuda.is_available())  # Should return True
print(torch.version.cuda)


def check_nccl_availability():
    try:
        # Check if NCCL is available
        is_nccl_available = torch.cuda.nccl.is_available(torch.randn(1).cuda())
        logging.info(f"NCCL Available: {is_nccl_available}")
    except Exception as e:
        logging.error(f"Error checking NCCL availability: {e}")

def get_nccl_version():
    try:
        # Retrieve NCCL version
        nccl_version = torch.cuda.nccl.version()
        logging.info(f"NCCL Version: {nccl_version}")
    except Exception as e:
        logging.error(f"Error retrieving NCCL version: {e}")

if __name__ == "__main__":
    check_nccl_availability()
    get_nccl_version()
